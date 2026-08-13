# Benchmark section plan

Status: **Tier 1 landed** (harness + report pipeline, `green_renormalization` smoke test,
`dos_methods`, `diag_dense_sparse`; see `benchmarks/`). Tiers 2-3 (KPM CPU/GPU, SCF solver
comparison) are still just the plan below. Nothing beyond Tier 1 should be implemented
without explicit user sign-off per tier, same process as `gpu_porting_plan.md`.

## Findings from building Tier 1

Building and running the `dos_methods` case (mode="ED" vs "Green" vs "KPM") immediately
caught two real, pre-existing normalization bugs in `dos.py`, both fixed alongside this
work (with new regression tests in `tests/dos/`) rather than worked around in the
benchmark case:

- **`dos_kpm`'s k-mesh average divided by `nk` instead of `numk = len(ks)`.** Invisible on
  1D lattices (`numk == nk` there), but wrong by a growing factor of `nk` on 2D/3D lattices
  (`numk = nk**dimensionality`) -- and, more surprisingly, wrong by a factor of the
  (physically irrelevant) default `nk=100` for `dimensionality=0` systems, where there is
  always exactly one k-point. `tests/kpm/test_kpm_dos_chain.py`'s pinned reference value was
  itself based on the buggy 0D behavior and has been corrected (~100x its old value).
  New test: `tests/dos/test_dos_kpm_2d_normalization.py`.
- **`mode="Green"/"RG"` DOS values were missing the standard `1/pi` prefactor.** `dos.py`
  used `green.green_operator`'s raw `-Im[Tr G]` directly as "the DOS", while `mode="ED"`
  (`dos_kmesh`) explicitly applies `ys *= 1./np.pi`. The mismatch was a near-exact factor of
  pi. New test: `tests/dos/test_dos_green_mode_normalization.py`.

This is the reason the plan calls the agreement column "non-negotiable" below -- it isn't
hypothetical, it caught real bugs on the very first case run.

**Dev-machine facts that shape this plan:**
- `pdflatex` and `latexmk` are both present here, so real PDF compilation is possible on
  this machine. They are *not* a repo dependency (`pyproject.toml` has no LaTeX/PDF Python
  package — the only existing PDF pipeline is `documentation/convert.sh`'s
  `pandoc user_guide.md -o user_guide.pdf`), so a fresh clone on another machine may not
  have them. The report step must degrade gracefully (write `report.tex`, print a warning,
  skip compilation) rather than crash if `latexmk`/`pdflatex` are missing.
- `jax.devices()` returns only `CpuDevice` here — no GPU. Any KPM CPU-vs-GPU case is really
  "numba vs jax-on-CPU" on this machine, and must be labeled that way rather than implying a
  measured GPU speedup (same honesty rule as `gpu_porting_plan.md`).

## Relationship to the existing `examples/*_benchmark` scripts

`examples/{0d,1d,2d}/kpm_scf_benchmark(_SC)/main.py` and
`examples/transport/keldysh_jax_benchmark/main.py` already exist and already compare
methods with timing + a relative-difference number, matplotlib plots, and (for the Keldysh
one) a nicely formatted console table. **Proposal: leave them where they are as
self-contained, documentation-style usage examples** (no LaTeX dependency, runnable
individually, teach a user how to call the alternate method). The new `benchmarks/` folder
is a separate, systematic layer on top: many cases run through one harness, one JSON
results format, one aggregate LaTeX/PDF report. It does not replace or absorb the examples,
though a couple of new cases (see below) will cover similar ground with the harness's
richer output (size sweep, cold/warm split, agreement column). Flag if you'd rather
consolidate them instead.

## Proposed structure

```
benchmarks/
  harness.py       # timing loop (cold + warm), machine-info capture, JSON writer/reader
  cases/
    __init__.py     # registry of available cases, by name
    dos_methods.py       # tier 1
    diag_dense_sparse.py # tier 1
    green_numba.py       # tier 1 (smoke test)
    kpm_cpu_gpu.py        # tier 2
    scf_solvers.py         # tier 3
  report.py        # JSON records -> report.tex (+ matplotlib PDFs) -> latexmk -pdf
  run_all.py       # CLI entry point: --quick (default, small sizes) / --full, --case NAME
  results/          # gitignored: raw JSON + compiled report.pdf land here
```

Each case module exposes one function, `run(sizes) -> list[record]`, where a record is a
plain dict: `{case, method, size, t_cold, t_warm, value, ref_value, reldiff, meta}`. Reusing
one record shape lets `report.py` stay generic across cases instead of special-casing each
one.

Build the LaTeX by plain f-string templating (no `jinja2` — not a current dependency, and
the templates here are simple tables + `\includegraphics`), and render size-sweep plots
with matplotlib (already a hard dependency) saved as PDF and pulled in via
`\includegraphics`, rather than pgfplots (would need a second LaTeX package assumption).
Persist every run's raw records to `results/<case>.json` so `report.py` can re-render the
document without re-running the (potentially slow) sweep.

## Non-negotiable harness behaviors

Three things that would otherwise silently produce misleading numbers:

1. **JIT warm-up.** Every candidate case below touches numba (`kpmnumba`,
   `green_renormalization_jit`) or jax (`kpmjax`, SCF jax solvers). The harness must do one
   untimed call before timing, and report cold (first-call, includes compile) and warm
   (steady-state) separately — compile time is a real cost for a one-shot script but
   shouldn't be blamed on the algorithm.
2. **Agreement column.** Comparing wall time between two methods that don't agree
   numerically is meaningless. Every record carries `reldiff` against a designated
   reference method for that case (mirrors what
   `examples/transport/keldysh_jax_benchmark/main.py` already does).
3. **Size sweep, not one number.** The physically interesting result is the crossover size
   where one method overtakes another, not a single timing at one arbitrary size. Every case
   defines a size axis (matrix dimension, k-mesh density, number of sites) and the report
   shows a table *and* a log-log plot per case.

`kpm_prec="single"|"double"` (in `kpmtk/kpmnumba.py`) is an accuracy/speed tradeoff within
one method, not two methods for the same quantity — it does not belong in this benchmark
suite as an "either/or" case.

## Candidate cases, ranked

### Tier 1 — infrastructure + first real cases

1. **`greentk/rg.py::green_renormalization(intra, inter, numba=True|False)`** — trivial,
   clean A/B (pure-Python Sancho-Rubio vs the numba-jitted version), same function, one
   kwarg flips the backend. Lowest risk case to validate the harness/report pipeline
   end-to-end (JSON -> tex -> compiled PDF) before building anything more complex on top of
   it. Do this one first, even before the "highest value" case below, purely to prove the
   plumbing.
2. **`dos.get_dos_general(h, mode="ED"|"Green"|"KPM")`** — one public entry point, three
   genuinely different algorithms (`dos_kmesh` exact diagonalization,
   `green.green_operator` via `parallel.pcall`, Chebyshev/KPM `dos_kpm`), textbook
   size-crossover story. Best first "real" case for demonstrating what this tool is for.
3. **Dense vs sparse diagonalization around `limits.densedimension` (currently 10000)** —
   `bandstructure.get_bands_nd` switches from `scipy.linalg.eigh` to
   `h.turn_sparse()` + `scipy.sparse.linalg.eigsh`. This case is *actionable*: its output
   tells the maintainer whether 10000 is the right cutoff on their own hardware, not just a
   speed comparison for its own sake.

### Tier 2 — needs the honest GPU caveat

4. **KPM moments, CPU vs GPU — `kpmtk/kpmnumba.py::kpm_moments_batch(kpm_cpugpu="CPU"|"GPU")`**,
   GPU branch dispatching to `kpmtk/kpmjax.py`. High potential value since
   `gpu_porting_plan.md` already flags the GPU batch path as an unbatched Python loop, but
   on any GPU-less dev machine this case must be labeled "numba vs jax-CPU-fallback", not a
   GPU speedup — same rule as the GPU porting plan.

### Tier 3 — richer but noisier, needs a second metric

5. **SCF solver comparison — `scftk/densitydensity.py::generic_densitydensity(solver="plain"|"krylov"|"anderson"|"broyden1"|"broyden_mixing")`**
   plus the jax solvers in `densitydensity_jax.py`
   (`solver="newton"|"fsolve"|"newton_krylov"|"lbfgs"|"fixed_point"|"broyden_mixing"`). The
   richest set of alternatives in the codebase, but wall time alone is misleading here since
   different solvers can take a different number of iterations to reach the same
   convergence tolerance. This case's record needs an extra field, `n_iterations`, and the
   report should show time-per-iteration alongside total time, not just total time.

### Deferred, not in scope for the first pass

- `topology.py::chern(integration="grid"|"qtci")` — plausible future case, not ranked yet.
- Keldysh jax vs numba (`keldyshtk/current_jax.py` vs `current.py::dc_current`) — a
  benchmark script already exists (`examples/transport/keldysh_jax_benchmark/main.py`);
  revisit whether to also formalize it into this harness once the harness exists, rather
  than building both at once.
- Wannierization — no second method to compare against yet.

## Proposed phased plan

- **Tier 1 — build the harness + report pipeline, validated by the `green_renormalization`
  smoke-test case, then add the DOS and dense/sparse-diagonalization cases.** This is the
  entire useful deliverable for a first pass: working `harness.py`/`report.py`/`run_all.py`,
  one case proving the pipeline, two cases with real physical/engineering payoff.
- **Tier 2 — KPM CPU/GPU case**, with the GPU-labeling caveat above.
- **Tier 3 — SCF solver comparison**, with the added `n_iterations` metric.
- Revisit the deferred list only after Tier 1–3 land and are useful in practice.

## Process notes

- `benchmarks/` must **not** be wired into `pytest`/`tests/` — the existing suite is already
  ~7.5 min, and benchmark timings are inherently non-deterministic across machines, the
  opposite of what a pass/fail test suite needs. `run_all.py --quick` (small sizes, default)
  keeps a manual smoke-run fast; `--full` is for an intentional, possibly long, sweep.
- Machine-info capture (for the report header, and so results from different machines are
  never silently compared as if equivalent): `platform.platform()`, CPU count
  (`os.cpu_count()`, and `numba`'s configured thread count via `parallel.py`), and installed
  `numpy`/`scipy`/`numba`/`jax` versions plus `jax.devices()`.
- `results/` (raw JSON + compiled PDFs) should be gitignored, same treatment as other
  machine-specific generated output — these numbers are only meaningful for the machine
  that produced them.
