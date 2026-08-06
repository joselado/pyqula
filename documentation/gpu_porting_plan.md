# GPU porting plan

Status: **not started**. This is a roadmap for future work, written after surveying the
codebase for GPU-portable hot spots. It complements, and is independent of, the CPU-side
`perf_optimization_plan` work (numba batching of dense diagonalization and KPM moments,
already landed; SCF-loop redundancy still pending). Nothing here should be implemented
without explicit user sign-off per tier, same as that plan.

**No GPU hardware is available in the current dev environment** (`nvidia-smi` is absent,
`jax.devices()` returns only a `CpuDevice`). Every tier below must therefore be validated
two ways: (1) numerical correctness against the existing CPU reference, checked here via
jax's transparent CPU fallback, and (2) actual speedup, which can only be measured on real
GPU hardware and is out of scope until that's available. Don't claim a speedup number that
wasn't measured on a GPU.

## Why jax, not torch/cupy

`jax` is already a hard runtime dependency (`pyproject.toml`), not optional, and several
modules already run through it: `selfconsistency/densitydensity_jax.py`,
`selfconsistency/vjinteraction_jax.py`, `keldyshtk/current_jax.py`,
`transporttk/kappa_jax.py`, `kpmtk/kpmjax.py`. jax transparently targets GPU when one is
present and `jaxlib` is built with CUDA/ROCm support, and falls back to CPU otherwise — no
code changes needed at the call site, only at the environment level. `kpmtk/kpmjax.py`
already encodes the right pattern for this:

```python
def is_gpu_available():
    import jax
    try:
        jax.devices("gpu")
        return True
    except Exception: return False

if is_gpu_available():
    pass
else:
    import os
    os.environ['JAX_PLATFORMS'] = 'cpu'
```

Any new GPU work should reuse this pattern rather than inventing a second one.

`parallelmpi.py` has a commented-out `pcallgpu` using MPI + `torch.cuda` — an abandoned,
never-wired-up prior attempt (zero callers, `torch` isn't even a dependency). Don't
resurrect it. Introducing torch or cupy as a second GPU backend alongside jax would split
the codebase's GPU story in two for no benefit; jax already covers dense linear algebra,
sparse matvecs, and batching (`vmap`) well enough for every candidate identified below.

## Candidate hot spots, ranked

### 1. Batched dense diagonalization — `htk/eigenvectors.py` (`peigh`/`peigvalsh`)

The single most reused hot path from the CPU perf work: a numba `prange`-parallel batched
`eigh` over many k-point Hamiltonians, called from `dos.py`, `spectrum.py`,
`bandstructure.py`, and (per the original CPU survey, not yet migrated to the batched
helper) `topology.py`, `ldos.py`, `gap.py`, `ipr.py`, `dostk/adaptivedos.py`,
`selfconsistency/hubbard.py`/`coulomb.py`. This is bounded by `limits.densedimension`
(currently 10000) below which dense diagonalization is used at all; above it the code goes
sparse (ARPACK via `scipy.sparse.linalg`), which is a different problem (see item 3).

Batched `eigh` over many independent small-to-medium matrices is a textbook GPU batching
win in principle (`jax.vmap(jnp.linalg.eigh)` or `jax.numpy.linalg.eigh` with a leading
batch axis) — but GPU kernel-launch and host-device transfer overhead can dominate for
small matrices/small k-meshes, where the existing numba CPU path may already be faster.
**This needs a benchmarking sweep across matrix size × batch size (k-mesh density) to find
the crossover point** before deciding whether to add a GPU path at all, and if so, whether
it replaces or supplements the numba path.

### 2. KPM moments, batched GPU path — `kpmtk/kpmnumba.py` / `kpmtk/kpmjax.py`

A GPU switch already exists (`kpm_cpugpu="GPU"` in `kpm_moments_v`/`kpm_moments_batch`),
but it's unfinished: `kpm_moments_batch`'s GPU branch loops in plain Python over
`kpm_moments_gpu` once per vector —

```python
elif kpm_cpugpu=="GPU": # no batched GPU path, loop over the single-vector one
    from .kpmjax import kpm_moments_gpu
    mus = np.array([kpm_moments_gpu(vs[iv],data,mo.row,mo.col,n=n)
            for iv in range(vs.shape[0])])
```

— unlike the CPU branch, which batches all vectors into one `prange`-parallel numba kernel
(`python_kpm_moments_batch_complex`/`_real`). This is the most scoped, lowest-risk item
here: replace the Python loop with a `jax.vmap` over `_kpm_moments_sparse_jit` (or an
explicit `jax.lax.map`/batched `sparse.BCOO` matmul) so all vectors are dispatched to the
device together, mirroring the CPU batching work already done for
`kpm_moments_A_batch`/`kpm_moments_batch` (see `perf_optimization_plan` Tier 2). Ground
truth for correctness already exists — the CPU batched kernels — so this can be verified
numerically on this GPU-less machine (via jax's CPU fallback) even before real GPU timing
is possible.

### 3. Sparse / Green's-function work — lower priority, needs its own research spike

`green.py`, `embedding.py`, `ldos.py`, `chitk/*`, `transporttk/*` (~30 `parallel.pcall`
sites, per the CPU survey) call `scipy.sparse.linalg` (ARPACK) or dense per-energy-point
Green's function inversions. The CPU survey already concluded these are *not* good numba
batching candidates (non-jittable SciPy calls per iteration), and the same reasoning likely
extends to GPU: there's no drop-in jax/cupy equivalent of ARPACK's shift-invert Lanczos, and
much of this work is single-shot (one energy/one defect configuration) rather than
embarrassingly parallel over many independent same-shape instances, which is what GPU
batching needs to pay off. Worth a dedicated feasibility spike later (e.g. cupy's sparse
eigensolvers, or reformulating as dense-batched recursive Green's function where
`densedimension` allows), but don't bundle it with items 1–2.

### 4. Modules that force jax onto CPU — needs investigation before touching

`classicalspin.py` and `symmetrytk/localsymmetry.py` both call
`jax.config.update('jax_platform_name', 'cpu')` unconditionally, overriding whatever GPU
would otherwise be picked up. The reason isn't yet understood from reading the code alone —
possibilities include small problem sizes where GPU dispatch overhead isn't worth it,
numerical-stability requirements, or avoiding GPU contention when these run under
`parallel.pcall`'s multiprocess pool (multiple processes fighting over one GPU context can
deadlock or serialize badly). Ask before lifting this restriction; it may be intentional.

## Proposed phased plan

Each tier is independent and needs explicit user confirmation before implementation, same
process as the CPU perf plan.

- **Tier 1 — finish the KPM batched GPU path** (item 2 above). Scoped, mirrors an existing
  CPU pattern, has a ready-made correctness reference. Best starting point.
- **Tier 2 — benchmark batched `eigh` on GPU** (item 1 above). Requires the size/batch
  sweep described above before committing to an implementation; likely the highest
  eventual payoff given how many call sites feed off `htk/eigenvectors.py`, but also the
  most engineering (deciding the CPU/GPU crossover, wiring a backend switch analogous to
  `kpm_cpugpu` through `dos.py`/`spectrum.py`/`bandstructure.py`/etc.).
- **Tier 3 — audit the forced-CPU jax modules** (item 4 above). Find out why before
  deciding whether to add a GPU option there too.
- **Tier 4 — research spike only, not committed work**: sparse/Green's-function GPU
  feasibility (item 3 above). Write up findings before proposing an implementation tier.

## Process notes

- This repo has no CI (no GitHub Actions, no lint config — see `CLAUDE.md`). GPU-path
  correctness must be verified manually, the same way Tier 1/2 of the CPU perf plan were:
  new tests under `tests/` following the existing numerical-equivalence pattern (e.g.
  `tests/kpm/test_kpm_moments_A.py`), run against the CPU reference implementation.
- Keep GPU dispatch **explicit**, not silently automatic (`kpm_cpugpu="CPU"|"GPU"` is the
  established convention) — this keeps every GPU path benchmarkable against a known-correct
  CPU baseline on demand, and keeps this GPU-less dev machine's test suite deterministic.
- Preserve the CPU-fallback pattern from `kpmtk/kpmjax.py` (`is_gpu_available()` /
  `JAX_PLATFORMS=cpu`) in any new GPU code path, so the package keeps working unmodified on
  machines without a GPU.
