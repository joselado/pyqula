# GPU port of the RPA spin response

Status: **Tiers 0-2 implemented and verified on the CPU fallback; the
device measurement is the open item.** Target: good performance for
1d/2d systems with ~100 sites in the unit cell. Development on the local
workstation (no GPU), timing and acceptance on a Triton GPU node. See
section 11 for exactly what landed and what is still open.

This is a concrete tier for `documentation/gpu_porting_plan.md`, which
already sets the conventions this plan follows (jax, not torch/cupy; an
explicit `*_cpugpu` switch rather than silent dispatch; the
`kpmtk/kpmjax.py` CPU-fallback pattern). Read that file first; nothing
here overrides it.

## 1. Scope

"The RPA spin response" is the **site-basis** route of
`future_development/magnons_tdhf.md`: the N x N (really 3N x 3N)
site-resolved spin susceptibility dressed as `chi@(1-V@chi)^-1`. Concretely
the entry points

| entry point | implementation |
|---|---|
| `h.get_spinchi_full` | `chitk/spinchi.py::spinchi_full` -> `chitk/rpa.py::chi_ops_RPA` |
| `h.get_spinchi_ladder` | `chitk/spinchi.py::spinchi_ladder` -> `chitk/rpa.py::chi_AB_RPA` |
| `h.get_magnon_bands(method="rpa")` | `chitk/spinchi.py::magnon_bands` -> `rpa_kernel_poles_ops` |
| `h.get_rpa_kernel_poles` | `chitk/rpa.py::rpa_kernel_poles` |
| `h.get_iets_ldos`, `get_qdos_iets` | both call `get_spinchi_full` |
| `chitk/densitychi.py` (charge channel) | `chi_AB_RPA` with the default identity operator |

all of which bottom out in **one** function:
`chitk/chiAB.py::chiAB_matrix`, the numba Lindhard kernel -- i.e. `chiAB`'s
`mode="matrix"` branch, and only that branch. `mode="trace"` and
`mode="diagonal"` (`h.get_chiAB_trace`) route through `getAB`/`chiAB_jit`
once per projector instead and are **not** part of this port; they stay on
numba, and the two must not be allowed to drift apart.

Everything above that kernel -- the RPA dressing, the pole tracking, the
q-path loop -- is cheap by comparison. So this is a port of one kernel,
plus the plumbing needed to keep data on the device around it.

**Out of scope**, deliberately:

- `chitk/pairchi.py` (the pair-basis ladder) and `bsetk/spinflip.py`
  (TDHF). Both are also "the spin response", but their cost is set by the
  interaction's support (`N(z+1)`, *linear* in N -- eight pairs for a
  honeycomb cell with V1) or by a Casida eigenproblem, not by the N^4
  contraction below. They are not the thing that hurts at 100 sites. If
  they turn out to matter, they get their own tier, not this one.
- `chiAB`'s `ij_mode="accelerated"` path
  (`chiAB_full_matrix_jit_kmesh`), which is a second, spinful-only
  implementation of the same quantity. It is not the default and is not
  what the spin entry points use; port the default path, leave that one
  on numba, and do not let the two drift.
- `imode="adaptive"`. Its integrator calls back into Python per point;
  it does not batch. Leave it on the CPU path.

## 2. Where the time actually goes (measured)

`chiAB_matrix`'s inner loop is

```
out[i,j,w] += MA[i,a,b] * MB[j,b,a] * (f_a - f_b) / (e_a - e'_b - w + i*delta)
```

over `i,j in [0,ni)`, `a,b in [0,n)`, `w in [0,nw)`, with

- `n  = 2N`  spin-orbitals (`N` sites, spinful),
- `ni = nj = 3N` for the full (Sx,Sy,Sz) response (`build_ops_projectors`
  makes one projected operator per site *per operator*); `ni = nj = N` for
  the S+/S- ladder,
- `nw` frequencies, and one such contraction **per k-point** in the mesh,
  and the whole mesh **per q-point** of a dispersion scan.

Cost is therefore `ni*nj*n^2*nw` = **`36 N^4 nw` complex FMA per k-point**
for the full spin response. Measured on this workstation (1 BLAS thread,
`nw=100`, synthetic operators with the same site-block structure
`build_ops_projectors` produces):

| N | n | ni=nj | `chiAB_matrix`, one k-point |
|---|---|---|---|
| 10 | 20 | 30 | 0.035 s |
| 20 | 40 | 60 | 0.353 s |
| 30 | 60 | 90 | 1.77 s |
| 40 | 80 | 120 | 5.86 s |

The ratios (5.0x, 3.3x for 20->30->40) match `N^4` to within a few
percent, so extrapolation is safe: **N=100 is ~230 s per k-point**, which
is `2.9 TFLOP` of complex arithmetic. A modest 1d run -- nk=20, a 20-point
q-path, nw=100 -- is then ~25 CPU-hours for one dispersion. That is the
problem this plan exists to solve.

### What is *not* the problem

Two things were checked so the plan does not chase them:

- **The eigendecompositions are not the bottleneck.** Two `eigh` of a
  200x200 per k-point is milliseconds against 230 s.
- **The numba kernel is not badly written.** Rewriting the contraction as
  BLAS ZGEMM (`out[w] = (TA*D[:,w]) @ TB.T`, see below) is only **1.6x**
  faster single-threaded, and multi-threaded BLAS adds nothing at these
  shapes (M=N=120, K=6400 is too small to thread). The measured single-core
  ZGEMM rate is ~22 GFLOPS, i.e. already near this CPU's roofline. There is
  no large CPU-side win being left on the table; the win has to come from
  hardware with more FP64 throughput.

So the expected payoff is essentially the FP64 GEMM throughput ratio:
~20 GFLOPS/core (and ~12 GFLOPS effective for the threaded numba kernel)
against an H200's tens of TFLOPS. **Order 100x is the expectation; the
number goes in this file only once measured on the device.**

## 3. The reformulation (exact, already verified)

The kernel is a GEMM in disguise. With `p = (a,b)` a single flattened
index of length `n^2`:

```
TA[i,p] = MA[i,a,b]                        (ni x n^2)
TB[j,p] = MB[j,b,a]                        (nj x n^2)   <- note the swap
D[p,w]  = (f_a - f_b) / (e_a - e'_b - w + i*delta)
out[w]  = (TA * D[:,w]) @ TB.T             (ni x nj), one GEMM per w
```

`MA[i] = conj(ws1) @ (A_i @ ws2.T)` and `MB[j] = conj(ws2) @ (B_j @ ws1.T)`
exactly as the current code builds them.

This was checked against `chiAB_matrix` directly on random input with the
site-block operator structure: **max absolute difference 4.6e-13 on
elements of magnitude 3.7e3**, i.e. agreement at the last bit of
complex128. The reformulation is not an approximation, and this check is
the template for the acceptance test in Tier 1.

### The one subtlety: the occupation cutoff

`chiAB_matrix` does `if abs(f_a-f_b) < delta/100: continue`. At low
temperature that skips every occupied-occupied and empty-empty pair. The
surviving fraction is `2f(1-f)` with `f` the filling (occupied->empty plus
empty->occupied), so the saving is **2x at half filling** and grows away
from it -- ~5.5x at `f=0.1`. It is a real saving that the numba loop
already collects and that a naive dense GEMM throws away by multiplying
zeros, but it is filling-dependent: quote it per case, and do not expect
the 100-site metallic runs near half filling to see more than ~2x from
this alone.

The device version must therefore **gather** the contributing pairs into a
compact `p` list rather than mask them. The complication is that the
number of surviving pairs varies with k, q and filling, and a varying
shape makes jax retrace the whole kernel. The fix is the one the dmrgpy
GPU port already paid for: **pad the gathered list to a fixed length**
(next power of two, or a per-run constant derived from the band structure)
and zero the padding. Padding plus a stable shape was worth 6.4-12.1x on
cold-start there, against 2.4-3.8x for padding alone -- the two knobs are
not independent, and neither is optional.

## 4. Design

### New module: `src/pyqula/chitk/chijax.py`

Mirrors `kpmtk/kpmjax.py` in structure and in header conventions
(`is_gpu_available()`, the `JAX_PLATFORMS=cpu` fallback,
`jax.config.update("jax_enable_x64",True)` -- without which every
"complex128" request is silently truncated to complex64).

Public surface, one function, deliberately a drop-in for the numba kernel
so both can be diffed on identical input:

```python
def chiAB_matrix_gpu(ws1, es1, ws2, es2, energies, Ais, Bjs, temp, delta,
                     w_batch_size=..., pair_pad=...):
    """Same contract and same return value as chiAB.chiAB_matrix."""
```

plus a k-batched entry point that keeps the eigenvectors on the device:

```python
def chi_matrix_kmesh_gpu(hks1, hks2, energies, Ais, Bjs, temp, delta, ...):
    """Batched eigh over the whole k-mesh, then accumulate chi over k
    without returning to the host."""
```

The second one is where most of the real speedup lives once the kernel is
fast: at N=100 a single k-point's `TA`/`TB` are 192 MB each, and shipping
them across PCIe per k-point would dominate the 0.2 s of arithmetic.

### Import discipline (not optional)

`chijax` must be imported **inside** the `chi_cpugpu=="GPU"` branch, never
at `chiAB.py` module scope. Three reasons, all visible in the existing
code: `kpmjax` prints an availability banner at import time (which would
then greet every CPU-only user of `get_spinchi_full`); it flips
process-global jax configuration at import (`JAX_PLATFORMS`,
`jax_enable_x64`); and the default CPU path still runs under
`parallel.pcall`'s fork-based pool, so initializing jax state in the
parent before a fork is precisely the hazard class the `workqueue`
comment at the top of `parallel.py` exists to document.

### Dispatch

Follow `kpm_cpugpu` exactly: a `chi_cpugpu="CPU"|"GPU"` kwarg, defaulting
to `"CPU"`, threaded down through `chiAB` -> `chiAB_q` -> the kernel.

**The lesson from Tier 1 of the KPM port applies here verbatim and is the
most likely way this lands broken:** there, `kpm_cpugpu` was unreachable
from the user-facing entry points because `random_trace`, `tdos` and
friends took no `**kwargs` and never forwarded the choice down, so
`kpm.tdos(..., kpm_cpugpu="GPU")` silently ran on the CPU. The spin chain
here is longer than KPM's -- `get_spinchi_full` -> `chi_ops_RPA` ->
`_chi_ops_matrix_vectorized` -> `chiAB` -> `chiAB_q` -> kernel, and
separately `magnon_bands` -> `rpa_kernel_poles_ops` -> the same. Every
link must forward `**kwargs`, and the test suite must assert reachability
end to end (a test that only exercises `chijax` directly would not have
caught the KPM bug either).

### Precision

Add `chi_prec="double"|"single"` alongside, following `kpm_prec`. Not a
Tier 1 deliverable -- only worth wiring once double precision is measured
on the device, since on a data-centre card FP64 is fast enough that the
single-precision option may not be worth its accuracy cost. Do not make it
the default under any measurement.

## 5. Tiers

Each needs explicit user sign-off before implementation, same process as
the rest of `gpu_porting_plan.md`.

### Tier 0 -- baseline and reference, no GPU code

- `benchmarks/cases/rpa_spin_response.py` following the existing
  `benchmarks/harness.py` contract (`time_cold_warm`, `save_records`),
  sweeping N (a chain and a 2d supercell) at fixed nk/nw, so that every
  later claim has a same-machine baseline to divide by.
  Use a **realistic nk for a 100-site cell**, not `chiAB`'s `nk=60`
  default: a 100-site supercell has an already-folded Brillouin zone, and
  60 would mean 3600 k-points in 2d, which would dominate the benchmark
  with something no real calculation does.
- On this workstation the baselines must be taken under `taskset` pinned
  to the performance cores: it is a 2P + 10E-core hybrid part, and the
  same kernel is ~2x slower on E silicon, so an unpinned baseline is not
  a number a later speedup can be divided by.
- A pinned numerical reference for a small system (chain, honeycomb Neel)
  so Tier 1 has something exact to diff against beyond "it looks right".

Acceptance: the table in section 2 reproduced by the committed benchmark.

### Tier 1 -- the exact GPU Lindhard kernel

`chijax.chiAB_matrix_gpu` as above: gather + pad the contributing pairs,
build `TA`/`TB`, `lax.map` (not `vmap`) over chunks of the frequency grid
to bound device memory, jitted with static chunk shapes.

Chunk over `w`, not over `p`: the `p` axis is the GEMM's contraction
dimension and splitting it forces an accumulation, while `w` is a free
output axis that splits for free. The KPM port chose chunking over
`vmap` for exactly the same memory reason (`vmap` materializes every
slice's state at once) and the same argument applies.

Acceptance:
- max abs difference against `chiAB_matrix` below 1e-10 relative, on
  chain and honeycomb, gapped and metallic, q=0 and q!=0, one operator
  (`ladder`) and three (`full`) -- run through jax's CPU fallback on the
  dev machine.
- no shape retraces across k-points or q-points in a scan (assert on
  `jit._cache_size()` or count compilations; a retrace per k-point would
  eat the entire speedup).

### Tier 2 -- keep the whole (k,q,omega) sweep on the device

The kernel being fast is not sufficient. Three things around it currently
force a round trip per k-point or per q-point:

1. `chiAB_q` builds `H(k)` and `H(k+q)` and calls `algebra.eigh` per k in
   a Python loop. Batch: build the whole mesh, one batched `jnp.linalg.eigh`.
   Note `es1/ws1` (at `k`) is **q-independent** -- compute it once and
   reuse it across the entire q-path instead of recomputing it at every q,
   which halves the eigen work in a dispersion scan.
2. The RPA dressing `chi@inv(1-Vq@chi)` (`chi_AB_RPA`, `chi_ops_RPA`) is a
   Python list comprehension over frequencies of a 3N x 3N inverse. Batch
   it as `jnp.linalg.solve` over the frequency axis on device. Cheap in
   flops (300^3 * nw) but it is a per-frequency host round trip today.
3. `magnon_bands` and `get_qdos_iets` scan q with `parallel.pcall`, i.e.
   **multiple processes**. With one GPU that is contention, not
   parallelism -- and it is exactly the failure mode `gpu_porting_plan.md`
   item 4 suspects behind `classicalspin.py`'s unconditional
   `jax_platform_name='cpu'`. Under `chi_cpugpu="GPU"`, force `cores=1`
   and loop over q in-process on the device.

Acceptance: a full `get_magnon_bands(method="rpa")` q-path at N=100
completes with a single set of jit compilations and no per-q host
transfer of `TA`/`TB`; the resulting bands match the CPU path's poles to
the frequency-grid spacing.

**Documentation is part of this tier, not an afterthought.**
`chi_cpugpu` (and later `chi_prec`) are user-facing kwargs, so
`CLAUDE.md`'s rule applies: `documentation/user_guide.md` gets a section
in the existing style (physics/motivation, runnable snippet, an entry in
the "Main functions and methods" reference), and `README.md`'s
FUNCTIONALITIES list gets the mention. Land it with the tier that
introduces the switch.

### Tier 3 -- optional binned spectral acceleration (approximate, opt-in)

An algorithmic reduction available on top of the port, worth writing down
because it is large and because it must never be enabled silently.

`D[p,w]` is a Lorentzian in `e_a - e'_b` alone. Deposit the pairs into a
histogram of that energy difference with linear (cloud-in-cell) weights,
`S[i,j,bin] = sum_{p in bin} TA[i,p] TB[j,p] * weight`, then broaden once
with an `(nbins x nw)` Lorentzian matrix. Cost drops from
`ni*nj*n^2*nw` to `ni*nj*n^2 + ni*nj*nbins*nw` -- at nw=100 and
nbins~400 that is roughly **50x** on top of whatever the device gives.

It is an approximation: the error is set by (bin width / delta) and is
second order with linear deposition. It ships only as `chi_binned=True`
with an explicit `nbins`, only after a convergence study against the exact
kernel is committed as a test, and never as a default. Do not fold this
into Tier 1 -- a plan that changes the numbers and the hardware in the
same step cannot attribute either.

### Tier 4 -- single precision

Only if Tier 2's measurements justify it. See section 4.

## 6. Memory

Per k-point at N=100, complex128: `TA`, `TB` are `ni * n^2 * 16 B` =
**192 MB each**; `D` (all frequencies) 64 MB; the output `(nw,3N,3N)`
144 MB. Under 1 GB, comfortable on any data-centre card.

The scaling is what matters for the chunking design: `TA` is
`3N * (2N)^2 * 16 B = 192 N^3` bytes, i.e. **1.5 GB at N=200**. The
frequency chunking of Tier 1 does not help with that (it is the `p` axis
that grows), so above ~N=150 the k-loop must stream `TA`/`TB` per k rather
than hold several. Build it that way from the start.

## 7. Testing on the dev machine (no GPU)

`tests/chi/test_chi_gpu.py`, mirroring `tests/kpm/test_kpm_gpu_batch.py`:

- kernel vs numba reference (the Tier 1 acceptance list above);
- an explicit multi-chunk case: a frequency grid larger than the default
  chunk, plus a small custom `w_batch_size` forcing several uneven chunks;
- a padding case: a filling where the number of contributing pairs is not
  a round number, so the padded and gathered paths must still agree;
- end-to-end reachability through `get_spinchi_full`,
  `get_spinchi_ladder`, `get_magnon_bands(method="rpa")` and
  `get_iets_ldos` with `chi_cpugpu="GPU"` -- asserting the GPU code
  actually ran (not just that the answer is right, which it would also be
  if the kwarg were silently dropped);
- the physics invariants that already exist, re-run through the GPU path:
  the Goldstone residual (`tests/magnon/test_goldstone.py`) and the
  RPA-vs-TDHF crosscheck (`tests/magnon/test_rpa_crosscheck.py`, which
  agrees to 0.4917/0.9037 today).

All of this runs through jax's transparent CPU fallback, which validates
correctness but says nothing about speed -- per `gpu_porting_plan.md`,
**do not quote a speedup that was not measured on a GPU.**

Remember `CLAUDE.md`'s rule: do not pipe pytest's output. Redirect to a
file and read that.

## 8. Measuring on Triton

Cluster rules live in `CLAUDE.local.md` / `docs/hpc_cluster.local.md`;
this is only the part specific to this work.

**Environment.** jax must be a CUDA build there; the workstation's is
CPU-only. Check `module spider cuda` and whether a jax/CUDA module exists
before installing anything, and **ask before any install** -- if a pip
install into a venv is needed it is `jax[cuda12]` matching the driver, and
that is a user decision, not an agent one.

**Where.** `$WRKDIR/calculations/pyqula_gpu_rpa/`, never `$HOME`.

**Job script.** Every `#SBATCH` value must be checked against
<https://scicomp.aalto.fi/triton/tut/gpu/> before proposing it -- this
plan deliberately does not invent time/memory/partition/`--gres` values.
Two things that are settled regardless:

- `export MKL_NUM_THREADS=1 OMP_NUM_THREADS=1` and `-c 1` for the GPU
  runs. The CPU baseline run is the one case that wants more cores, and it
  must then be timed alone on the node or its numbers are meaningless.
- Price the queue with `sbatch --test-only` before committing to a
  partition. In the dmrgpy GPU work this is what surfaced a ~10-hour A100
  wait against an immediately-free H200 partition; it creates no job and
  is not polling.

**Do not poll the queue.** Wait on the job's output file; use
`seff <jobid>` afterwards.

**What to report.** Cold and warm timings separately, per the benchmark
harness's own contract -- the dmrgpy port found cold-start (jit tracing
plus compilation) dominating everything until shapes were held still, and
found the effect roughly twice as large on the device as the host
measurement predicted. A GPU number here that does not separate the two is
not interpretable.

## 9. What could make this not pay off

Written down in advance so the answer is a measurement, not a
rationalization:

- **Small N.** Below roughly N=30 the kernel is milliseconds and dispatch
  overhead dominates; the GPU will lose. That is fine and expected -- it is
  why the switch is explicit. What must be produced is the **crossover in
  N**, the same deliverable `gpu_porting_plan.md` demands of its Tier 2.
- **FP64 throughput.** Everything here is complex128. On a card with
  1/32-rate FP64 (any consumer GPU) none of these estimates hold. State
  the card with every number.
- **Retracing.** A shape that varies per k-point or per q-point (the pair
  gather is the obvious candidate, the frequency chunk the second) turns a
  one-off compile into a per-iteration compile and can make the GPU path
  slower than numba outright. This is the single largest implementation
  risk and is why padding is in Tier 1 rather than deferred.
- **`parallel.pcall` contention.** If Tier 2's `cores=1` forcing is
  forgotten, several processes will each try to hold ~400 MB of device
  state and the q-scan can serialize or fail outright.

## 10. Follow-ups this plan deliberately leaves out

- The pair-basis (`chitk/pairchi.py`) and TDHF (`bsetk/spinflip.py`)
  routes -- see section 1. If a 100-site *non-collinear* or non-onsite-V
  case is the real target, that is a different plan, since the site-basis
  vertex refuses those interactions anyway
  (`_require_onsite_only_V`).
- `htk/eigenvectors.py`'s batched `eigh` (Tier 2 of
  `gpu_porting_plan.md`). Tier 2 here batches the eigendecomposition
  *locally* inside the chi path; folding that into the shared helper so
  `dos.py`/`bandstructure.py` benefit too is the more general job and
  should not be smuggled in here.

## 11. Implementation status

What is on disk (uncommitted at the time of writing), against the tiers
above:

**Tier 0 -- done.** `benchmarks/cases/rpa_spin_response.py`, registered in
`benchmarks/cases/__init__.py`, on the existing `harness.py` contract
(`time_cold_warm` + `save_records`, cold and warm reported separately). It
sweeps N for a ferromagnetic chain supercell at `nk=4`, `nw=40`, and
compares the two backends' `Im Tr chi` summed over the frequency grid. The
exchange field is explicit rather than self-consistent on purpose -- the
kernel's cost does not depend on the mean field being converged, and an SCF
per size would time the solver. `--quick` (N=4,8,12) reproduces the shape
of section 2's table; the docstring carries the `taskset` warning for
hybrid CPUs.

**Tier 1 -- done.** `src/pyqula/chitk/chijax.py`: `pair_plan` (gather +
quantized padding), `_gathered_operator_tensors`, `_chi_from_tensors`
(`lax.map` over frequencies), `chiAB_matrix_gpu` (drop-in for
`chiAB_matrix`) and `chi_matrix_kmesh_gpu` (batched `eigh` over the mesh,
accumulation without host round trips). Verified against the numba kernel
at ~1e-15 relative on chain and honeycomb, gapped and metallic, q=0 and
q!=0, one operator and three, and at low temperature -- see
`tests/chi/test_chi_gpu.py`. `_occupations` uses the tanh form rather than
`1/(1+exp(beta*e))`: identical mathematically, but the exp form overflows
noisily at low temperature (`chiAB_full_matrix_jit` already used tanh, so
there was precedent in the file).

**Tier 2 -- done, minus one deferred optimization.**
`chiAB_q` gained `chi_cpugpu="CPU"|"GPU"`; the GPU branch builds `H(k)` and
`H(k+q)` for the whole mesh and calls `chi_matrix_kmesh_gpu`. It
**raises** for `mode="trace"`/`"diagonal"` and for `imode="adaptive"`
rather than silently computing on the CPU -- a silent fallback is the same
failure as a dropped kwarg. `chijax` is imported inside that branch only.
`spinchi._map_over_q` replaces the bare `parallel.pcall` in `magnon_bands`,
`get_qdos_iets` and `densitychi.plasmon_bands`, keeping the q-loop in one
process under `chi_cpugpu="GPU"`. Documented in `documentation/user_guide.md` (prose
section plus a reference entry) and `README.md`, per CLAUDE.md.

*Deferred:* the `ws1`-at-`k` reuse across a q-path (section 5, Tier 2
item 1). It halves the eigen work in a dispersion scan, but the eigen work
is milliseconds against the kernel's seconds, so it buys nothing until the
kernel is actually fast on a device. The RPA dressing (item 2) is likewise
still a host-side per-frequency solve; same reasoning. Both are now worth
re-profiling -- see "What is still open" below, since the kernel they were
measured against has since become ~800x faster.

**Tiers 3 and 4 -- not started, and deliberately so.** The binned spectral
acceleration and single precision both change the numbers; neither should
be touched before the device measurement attributes the plain port.

### Device measurements (Tesla V100-SXM2-32GB, 2026-08-27)

Triton jobs 19967215 (`--quick`, 1:35 wall, 1.77 GB host RAM) and 19967260
(`--full`), both on partition `gpu-debug` with 1 GPU + 1 core and
`MKL_NUM_THREADS=OMP_NUM_THREADS=1`, using jax 0.7.1 + `jax_cuda12_plugin`
from `scicomp-python-env/2025.2`. **No install was needed**: that
environment already carries a CUDA-capable jax, which removes the one user
decision section 8 was blocked on.

`benchmarks/cases/rpa_spin_response.py --full` (job 19967260; nk=4, nw=40,
complex128, the full Sx,Sy,Sz response of a ferromagnetic chain supercell,
one core with BLAS pinned on both sides):

| N | numba warm | jax/V100 warm | ratio | agreement |
|---|---|---|---|---|
| 4 | 0.006 s | 0.016 s | 0.36x | 1.3e-15 |
| 8 | 0.041 s | 0.018 s | 2.2x | 1.8e-16 |
| 12 | 0.196 s | 0.015 s | 12.8x | 1.4e-15 |
| 16 | 0.635 s | 0.016 s | 40x | 2.6e-15 |
| 24 | 3.031 s | 0.026 s | 116x | 3.4e-16 |
| 32 | 9.513 s | 0.046 s | 207x | 1.7e-16 |
| 48 | 48.097 s | 0.096 s | 499x | 2.3e-16 |
| 64 | 157.712 s | 0.189 s | 833x | 5.1e-16 |

Three things this settles.

**The crossover of section 9 is real and sits at N ~ 7** on a V100. Below it
the CPU wins, which is exactly why the switch is explicit and defaults to
CPU.

**Above it the ratio keeps growing, and it has not stopped.** The CPU side
follows `N^4` (157.7 s at N=64 against 9.5 s at N=32 is 16.6x for 16x the
work); the device side grows far more slowly (0.046 -> 0.189 s over the
same step), because at these sizes the card is still filling up. At N=12
`seff` reported 1% average GPU utilization; by N=64 the kernel sustains
~4.1 TFLOP/s of the nominal complex128 work, against the V100's 7.8
TFLOP/s FP64 peak -- so the card is finally busy, and the gather means the
real arithmetic is lower still. **Order 100x was the section-2 guess; the
measured number at N=64 is 833x, on the smallest of Triton's data-centre
cards.**

**The reformulation is exact on the device**, at every size: the two
backends agree to 1e-16..1e-15 relative, so `jax_enable_x64` is doing its
job and nothing silently dropped to complex64.

Cold start is flat at 0.5-0.8 s for jax at *every* N, against numba's 12 s
one-off compile -- i.e. the padding+jit work of Tier 1 did its job and
there is no per-k or per-q retrace. The 24 GB of reserved VRAM `seff`
reports is jax's default preallocation, not real use.

Projecting to the 100-site target: the CPU side is ~940 s per (nk=4, nw=40)
call, which is a safe `N^4` extrapolation of a clean `N^4` curve. The device
side is **not** safe to extrapolate the same way -- it is still bending
(48 -> 64 cost 1.97x for 3.16x the work, i.e. the card was still filling
up), and N=100 also crosses into the memory regime of section 6, where
`TA`/`TB` are ~192 MB each and the k-loop has to stream rather than hold
several. Nothing at N<=64 exercised that. So the honest statement is: the
CPU side of a 100-site dispersion is hours, the device side is very
probably minutes, and only a run at N=100 turns "probably" into a number.

The 54 passes of `tests/chi/test_chi_gpu.py` in that job are 27 non-slow
tests run twice: Triton's pytest 8.4.2 collects every file in this repo
twice (pre-existing files included, so it is a pytest-version quirk of that
environment, not something about this test file). Nothing failed, and
complex128 held on the device -- which is the thing that would silently
break if `jax_enable_x64` were not set.

### What is still open

- **A bigger card.** Everything above is a V100 (7.8 TFLOP/s FP64). An A100
  or H100 should widen the ratio further; queue prices on 2026-08-27 were
  ~2 h and ~6 h against gpu-debug's immediate start, so that is a
  deliberate, later run rather than something to wait on.
- **N=100 itself**, rather than an extrapolation from N=64. The CPU
  baseline there is ~940 s per call, so the honest way to run it is
  device-only against the N=64 CPU point, or an overnight CPU baseline in
  its own job.
- **The deferred Tier 2 items** (`ws1` reuse across a q-path, the RPA
  dressing on device). Now that the kernel is 800x faster, the host-side
  per-frequency `solve` and the duplicated eigendecomposition are a much
  larger share of the remaining time than they were when they were
  deferred; re-profile a full `get_magnon_bands` q-path before deciding.
- **Tiers 3 and 4** (binned spectral acceleration, single precision) remain
  untouched and unjustified: at 833x the plain port already clears the
  problem this plan was written for.

### Test coverage

`tests/chi/test_chi_gpu.py`, 28 tests: kernel-vs-numba across
system/q/operator-count and at low temperature; padding to lengths that are
not the survivor count; `pair_pad` too small raising instead of truncating;
end-to-end backend agreement through `get_spinchi_full`/`get_spinchi_ladder`
with and without the RPA dressing; a jit-cache assertion that a 3-q, 6-k
scan adds at most one compilation (the retracing risk of section 9); a spy
asserting `chi_cpugpu="GPU"` actually *reaches* the kernel from
`get_spinchi_full`, `get_spinchi_ladder`, `get_iets_ldos`,
`get_magnon_bands(method="rpa")`, `get_qdos_iets` and
`get_densitychi_RPA`, and that the default never does; refusals
for the unsupported combinations; and the site-basis Goldstone invariant of
`tests/chi/test_magnon_goldstone_doped_chain.py` re-run on the device path.

### The scope question, still unanswered

Section 1 assumes "the RPA spin response" means the site-basis route. The
three commits preceding this work are all pair-basis transverse-RPA
(`chitk/pairchi.py`, `bsetk/spinflip.py`). If those were the intended
target, the device-residency machinery here transfers but the kernel does
not -- their cost is linear in N, or a Casida eigenproblem -- and that
needs its own tier.
