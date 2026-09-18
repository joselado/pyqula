# GPU porting plan

Status: **Tiers 1 and 3 done** (KPM batched GPU path; the forced-CPU modules no longer
switch jax's platform globally); Tiers 2 and 4 not started. A
separate tier for the RPA response kernel (`chi_cpugpu`, with `chi_prec` single precision
as the GPU default) is implemented and measured on a V100 and a GTX 1060 -- see
`future_development/gpu_rpa_spin_response.md`.
This is a roadmap for future work, written after surveying the codebase for GPU-portable hot spots.
It complements, and is independent of, the CPU-side `perf_optimization_plan` work (numba
batching of dense diagonalization and KPM moments, already landed; SCF-loop redundancy
still pending). Nothing here should be implemented without explicit user sign-off per tier,
same as that plan.

**The dev workstation has a consumer GPU** (GeForce GTX 1060 6GB, Pascal) with a
CUDA-built jax in a separate environment; the default environment's jax is CPU-only.
Earlier tiers were written on a GPU-less machine and validated only through jax's CPU
fallback, so every tier still needs checking two ways: (1) numerical correctness against
the existing CPU reference, and (2) actual speedup, measured on a device. Don't claim a
speedup number that wasn't measured on a GPU, and state the card with every number: a
consumer card's FP64 is an order of magnitude behind its FP32 (21x on the GTX 1060, see
`future_development/gpu_rpa_spin_response.md`), so data-centre and consumer ratios do not
transfer.

## Why jax, not torch/cupy

`jax` is already a hard runtime dependency (`pyproject.toml`), not optional, and several
modules already run through it: `scftk/densitydensity_jax.py`,
`scftk/vjinteraction_jax.py`, `keldyshtk/current_jax.py`,
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

A prior, never-wired-up attempt at an MPI + `torch.cuda` `pcallgpu` (zero callers, `torch`
isn't even a dependency) lived in `parallelmpi.py` and has since been removed as dead code.
Don't resurrect that approach. Introducing torch or cupy as a second GPU backend alongside
jax would split the codebase's GPU story in two for no benefit; jax already covers dense
linear algebra, sparse matvecs, and batching (`vmap`) well enough for every candidate
identified below.

## Candidate hot spots, ranked

### 1. Batched dense diagonalization — `htk/eigenvectors.py` (`peigh`/`peigvalsh`)

The single most reused hot path from the CPU perf work: a numba `prange`-parallel batched
`eigh` over many k-point Hamiltonians, called from `dos.py`, `spectrum.py`,
`bandstructure.py`, and (per the original CPU survey, not yet migrated to the batched
helper) `topology.py`, `ldos.py`, `gap.py`, `ipr.py`, `dostk/adaptivedos.py`,
`scftk/hubbard.py`/`coulomb.py`. This is bounded by `limits.densedimension`
(currently 10000) below which dense diagonalization is used at all; above it the code goes
sparse (ARPACK via `scipy.sparse.linalg`), which is a different problem (see item 3).

Batched `eigh` over many independent small-to-medium matrices is a textbook GPU batching
win in principle (`jax.vmap(jnp.linalg.eigh)` or `jax.numpy.linalg.eigh` with a leading
batch axis) — but GPU kernel-launch and host-device transfer overhead can dominate for
small matrices/small k-meshes, where the existing numba CPU path may already be faster.
**This needs a benchmarking sweep across matrix size × batch size (k-mesh density) to find
the crossover point** before deciding whether to add a GPU path at all, and if so, whether
it replaces or supplements the numba path.

**Sweep on the GTX 1060 (2026-09-17/18).** Random dense Hermitian batches, warm times in
seconds, numba `peigvalsh`/`parallel_diagonalization` on 8 threads against
`jax.jit(jnp.linalg.eigvalsh/eigh)` on the device, host-device transfers included (what a
drop-in replacement would pay). `err32` is the single-precision eigenvalue error relative
to the spectral scale; double agrees with numba to <1e-10 everywhere.

| n | batch | numba vals | GPU double vals | GPU single vals | numba vecs | GPU double vecs | GPU single vecs | err32 |
|---|---|---|---|---|---|---|---|---|
| 8 | 4096 | 0.029 | 0.067 | 0.006 | 0.029 | 0.070 | 0.007 | 2e-6 |
| 16 | 4096 | 0.076 | 0.152 | 0.011 | 0.077 | 0.157 | 0.018 | 3e-6 |
| 32 | 256 | 0.055 | 0.086 | 0.006 | 0.057 | 0.090 | 0.007 | 6e-6 |
| 32 | 4096 | 1.70 | 1.34 | 0.090 | 1.83 | 1.38 | 0.111 | 6e-6 |
| 64 | 16 | 0.019 | 0.009 | 0.002 | 0.022 | 0.010 | 0.003 | 4e-7 |
| 64 | 4096 | 7.68 | 1.88 | 0.365 | 7.73 | 2.04 | 0.446 | 1e-6 |
| 128 | 256 | 2.10 | 0.60 | 0.088 | 2.14 | 0.50 | 0.083 | 1e-6 |
| 128 | 1024 | 4.10 | 1.75 | 0.320 | 3.99 | 1.93 | 0.389 | 2e-6 |
| 256 | 16 | 0.48 | 0.14 | 0.030 | 0.66 | 0.15 | 0.033 | 1e-6 |
| 256 | 64 | 1.33 | 0.49 | 0.114 | 1.40 | 0.57 | 0.134 | 1e-6 |
| 256 | 256 | 6.38 | 2.00 | 0.483 | 6.59 | 2.29 | 0.588 | 2e-6 |
| 512 | 16 | 2.53 | 0.68 | 0.150 | 2.77 | 0.72 | 0.164 | 3e-6 |
| 512 | 64 | 11.3 | 2.70 | 0.608 | 13.1 | 3.02 | 0.662 | 2e-6 |
| 1024 | 8 | 20.0 | 2.18 | 0.422 | 20.8 | 2.84 | 0.520 | 1e-6 |
| 1024 | 16 | 39.9 | 5.56 | 0.989 | 40.1 | 5.74 | 1.04 | 1e-6 |

The crossover is in the matrix size, not the batch size. Below n~32 double precision on the
device loses to numba however large the batch, because the per-matrix solve is too small to
fill the card and the transfers are not amortized. It breaks even near n=32, wins 2-4x at
n=64-256, and 7-9x at n=512-1024 -- eigenvalues and eigenvectors behave the same, so
`peigh` and `peigvalsh` would both benefit and neither needs its own crossover. Single
precision wins everywhere, 5-40x, at ~1e-6 relative eigenvalue error: fine for DOS and
bands, not for anything differencing eigenvectors, which is the same trade `chi_prec`
already exposes for the RPA kernel.

Note that n here is `limits.densedimension`-bounded (10,000) but realistically a few
hundred, and a k-mesh gives batches of hundreds to thousands, so the wide-and-shallow
corner (small n, huge batch) is the common one for bands/DOS and is exactly where the
device does not help in double precision. A GPU option should therefore not be the
default, and should probably not even be offered below a size threshold.

One hard constraint surfaced: cuSOLVER's batched Jacobi solver needs ~500 bytes of
workspace per matrix entry, so a 4096 x 64 x 64 batch asked for 7.75 GiB in one dispatch
and failed on the 6 GB card. A GPU path must chunk by workspace, not by the size of the
matrices themselves (2^21 entries per dispatch worked for every row above).

### 2. KPM moments, batched GPU path — `kpmtk/kpmnumba.py` / `kpmtk/kpmjax.py` — **done**

`kpm_moments_batch`'s GPU branch used to loop in plain Python over the single-vector
`kpm_moments_gpu` kernel, one dispatch per vector. It now calls
`kpmjax.kpm_moments_batch_gpu`, which builds the sparse `BCOO` matrix once and dispatches
the batch via `jax.lax.map(..., batch_size=gpu_batch_size)` (default 256, jitted as
`_kpm_moments_sparse_batch_jit`) instead of one `jax.vmap` over the whole batch — a plain
vmap would materialize `nvec` copies of the per-vector recursion state on-device at once,
which for a full-space trace (`kpm.full_trace`/`full_trace_A`, one basis vector per site,
so `nvec=nsites`) can reach `limits.densedimension` (10,000) and plausibly exceed real GPU
memory; chunking bounds device memory independent of `nvec`. The same treatment was
extended to `kpm_moments_A_batch` (the operator-weighted moments used by
`random_trace_A`/`full_trace_A`), which previously had no GPU path at all
(`kpm_cpugpu="GPU"` raised `ValueError`) — it now calls `kpmjax.kpm_momentsA_batch_gpu`,
built the same way around a new `_kpm_momentsA_sparse` recursion.

Separately, `kpm_cpugpu` used to be unreachable from the actual user-facing entry points:
`random_trace`, `random_trace_A`, `full_trace_A`, and `tdos` (which `pdos`/`ldos` build on)
took no `**kwargs` and never forwarded a backend choice down to
`get_moments_batch`/`get_moments_A_batch`, so `kpm.tdos(..., kpm_cpugpu="GPU")` silently
ran on the CPU regardless. All four now accept and forward `**kwargs` (including
`kpm_cpugpu`, `kpm_prec`, and the new `gpu_batch_size`).

Both the chunked dispatch and the reachability fix are verified in
`tests/kpm/test_kpm_gpu_batch.py`: batched moments and A-batched moments each against the
CPU reference (real/complex input, single/double precision), an explicit multi-chunk check
(batch size > the default chunk, plus a small custom `gpu_batch_size` forcing many uneven
chunks), and an end-to-end check through `kpm.tdos`/`kpm.ldos`/`kpm.full_trace`/
`kpm.full_trace_A`. All of this is exercised transparently through jax's CPU fallback on
this GPU-less machine at the time. On the GTX 1060 (2026-09-17) the same tests pass on the
device, and the batched moments (64 vectors, 200 moments, 8-thread numba as the CPU side)
measure:

| sites | input | numba double | GPU double | GPU single |
|---|---|---|---|---|
| 10,000 | real | 0.25 s | 0.16 s | 0.12 s |
| 10,000 | complex | 0.58 s | 0.35 s | 0.23 s |
| 160,000 | real | 7.6 s | 4.4 s | 2.5 s |
| 160,000 | complex | 15.6 s | 11.5 s | 7.8 s |

Double precision on the device agrees with numba to ~1e-14 relative, single to ~3e-7.
A 1.4-3x gain is modest because a sparse matvec is memory-bandwidth bound rather than
FLOP bound, which is not where a consumer card beats a CPU by much. Real GPU memory
behavior at `nvec=nsites` is still untested, so the chunk size is a reasoned default, not
a benchmarked one.

The same measurement exposed a CPU-side bug, since fixed: the numba kernels computed
`2.*data[k]*a[col[k]]`, and the float64 literal promoted every single-precision product
to double, so `kpm_prec="single"` on the CPU was *slower* than double at 10,000 sites.
With `2*data` precomputed in the data's own dtype, CPU single is now 1.4-1.6x faster than
double at 160,000 sites. One difference remains between the backends: numba accumulates
the inner products sequentially in float32, so its single-precision error grows with
system size (~8e-6 relative at 10,000 sites, ~4e-5 at 160,000), while jax's tree
reduction stays at ~3e-7.

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

### 4. Modules that forced jax onto CPU — **done**

`classicalspin.py` and `symmetrytk/localsymmetry.py` used to call
`jax.config.update('jax_platform_name', 'cpu')` at import time. The commits that added it
gave no reason. The switch was global: in a script importing either module before any jax
array existed (the normal order, imports at the top), every later `kpm_cpugpu="GPU"` or
`chi_cpugpu="GPU"` call ran on the CPU while still printing "GPU available". Imported after
jax had already placed an array, it did nothing, so the modules then ran on the GPU.

Measured on the GTX 1060 (2026-09-17) with the platform switched either way, the GPU buys
these modules nothing: both drive `scipy.optimize` from the host with one small jitted call
per step, so per-call transfer dominates. A 400-spin `minimize_energy` takes 8.3 s on the
CPU and 8.4 s on the GPU; `all_permutations(n=4)` on bilayer graphene 2.3 s against 5.1 s.
The CPU placement was therefore kept but scoped: `classicalspin.jit_on_cpu` jits a function
and `jax.device_put`s its inputs on the CPU (`jax.jit(device=...)` is deprecated in jax
0.11), and `localsymmetry` and `classicalspintk/align.py` (same shape, previously on
whatever device was default) use it too. jax's default device is no longer touched;
`tests/classicalspin/test_classicalspin.py` checks that in a fresh interpreter.

### 5. jax modules that already run on the GPU implicitly

Without any switch, `scftk/densitydensity_jax.py`, `scftk/vjinteraction_jax.py`,
`graphenetk/relax.py`, `transporttk/kappa_jax.py` (called from `transporttk/kappa.py`),
`keldyshtk/current_jax.py` and `fermisurfacetk/swarmfs.py` place their arrays on the GPU
whenever a CUDA jax is installed, which is not the explicit-dispatch convention below.
Their test files (a proxy for realistic sizes) take the same time either way on the
GTX 1060: GPU/CPU wall-time ratios 0.95-1.14, 2026-09-17. Nothing is gained, and whether
to pin them to the CPU or give them a switch is still open.

Running them on the device did expose a real bug, now fixed: the jax Newton SCF
differentiated the density matrix through `eigh`'s eigenvector tangent, which is NaN at an
exact degeneracy. cuSOLVER returns degenerate levels bit-identical where LAPACK splits
them by ~1e-16, so the Newton SCF stalled only on the GPU.
`densitydensity_jax.fermi_projector` now carries a Daleckii-Krein custom JVP.

## Proposed phased plan

Each tier is independent and needs explicit user confirmation before implementation, same
process as the CPU perf plan.

- **Tier 1 — finish the KPM batched GPU path** (item 2 above). **Done.**
- **Tier 2 — benchmark batched `eigh` on GPU** (item 1 above). Requires the size/batch
  sweep described above before committing to an implementation; likely the highest
  eventual payoff given how many call sites feed off `htk/eigenvectors.py`, but also the
  most engineering (deciding the CPU/GPU crossover, wiring a backend switch analogous to
  `kpm_cpugpu` through `dos.py`/`spectrum.py`/`bandstructure.py`/etc.).
- **Tier 3 — audit the forced-CPU jax modules** (item 4 above). **Done**: no GPU option,
  the CPU placement is now local to those modules.
- **Tier 4 — research spike only, not committed work**: sparse/Green's-function GPU
  feasibility (item 3 above). Write up findings before proposing an implementation tier.
- **The RPA response kernel** (`chitk/chiAB.py::chiAB_matrix`, the `N^4` Lindhard
  contraction behind every spin/charge RPA entry point) has its own plan and its own
  measurements in `future_development/gpu_rpa_spin_response.md`. It follows the conventions
  of this file (`chi_cpugpu="CPU"|"GPU"`, the `kpmjax` fallback pattern, an import kept
  inside the GPU branch) and is implemented through its Tier 2; what is missing is the
  speedup itself, which needs a GPU node.

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
