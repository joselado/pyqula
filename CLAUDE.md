# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project context
When working with this codebase, prioritize readability over cleverness. Ask clarifying questions before making architectural changes.

When a physics/numerical algorithm issue (e.g. a mean-field, topology, or Green's-function derivation) is
proving hard to resolve from the code and general knowledge alone, ask the user whether to pull a specific
paper from arXiv to help sort it out, rather than guessing at the underlying formalism.

Whenever doing a new implementation (a formalism or algorithm not already present in this codebase, as
opposed to extending an existing one), check arXiv for a paper covering it before writing code, rather than
relying on general knowledge alone, and use that reference to guide the implementation. If a tested,
open-source implementation of the same method is available under a permissive/compatible license (e.g.
MIT, GPL), use it as a benchmark (compare results against it) or mirror its structure/approach where
fitting, rather than writing the algorithm from scratch against general knowledge alone.

## What this is

pyqula is a Python library for quantum tight-binding calculations on lattices: band structures, mean-field
(self-consistent) interacting Hamiltonians, topological invariants (Chern numbers, Z2, Berry curvature),
Green's function/spectral-function methods, Chebyshev polynomial (KPM) algorithms, and quantum transport
(NEGF, heterostructures/junctions).

## Install / build

```bash
pip install -e .                       # editable install from repo root (package lives in src/)
```

There is no separate lint config in this repo — no ruff/flake8/black config and no GitHub Actions workflow.
Don't assume tooling that isn't here.

## Tests

```bash
python -m pytest tests            # run the whole suite
python -m pytest tests/scf -v     # run one topic
```

**Do not pipe pytest's output** (`... | tail`, `... | grep`). The shell reports the *pipe's* exit
status, not pytest's, so a failed or even crashed run looks like success. This is not hypothetical:
it masked a fatal interpreter abort (numba's non-threadsafe `workqueue` layer entered from two
threads, fixed in `d759b32`) as exit code 0, and separately made an `unrecognized arguments` error —
which ran no tests at all — also report exit code 0. If you must post-process the output, use
`set -o pipefail` first, or redirect to a file and read that.

`tests/<topic>/test_*.py` holds pytest tests that assert a physical/numerical invariant (e.g. that a
self-consistent mean-field result doesn't depend on the random initial guess used to seed it, or that two
independent code paths computing the same quantity agree to numerical tolerance). `pyproject.toml`'s
`[tool.pytest.ini_options]` puts `src` on `pythonpath` so `pyqula` resolves to `src/pyqula` — it must use
`--import-mode=importlib`, because the repo root directory is itself named `pyqula` and contains a stray
empty `__init__.py`; with the default import mode pytest's package-root walk from `tests/` would otherwise
resolve `import pyqula` to the repo root instead of `src/pyqula`. Some of these tests do a handful of
repeated SCF/RPA calculations to check invariance and take several seconds each — the slowest individual
tests (SCF/RPA, jax Newton solvers, Keldysh transport) run 10-25s each, so the full suite takes many
minutes, not under a minute. It currently collects **687 tests** (`pytest tests --collect-only -q`);
the old "~7.5 min for 406 tests" figure predates the Keldysh, transport and AAA suites and is stale —
`tests/scf` alone is ~15 min and `tests/keldysh` ~12 min. A fresh whole-suite wall time still needs
measuring on an idle machine; treat any timing taken while other jobs are running as meaningless.

`examples/` (organized by dimensionality: `0d/ 1d/ 2d/ 3d/`, plus `transport/`, `embedding/`, `wannier/`,
`classicalspin/`, `latticegas/`) contains runnable `main.py` scripts that double as usage documentation —
grep there for an example of any feature before implementing something from scratch.

## Architecture

### Core objects and where behavior lives

- `geometry.Geometry` (`src/pyqula/geometry.py`) — atomic positions/lattice vectors. Built via factory
  functions like `geometry.chain()`, `geometry.honeycomb_lattice()`, `geometry.kagome_lattice()`, etc.
  `Geometry.get_hamiltonian()` builds a `Hamiltonian` from it (default first-neighbor hopping).
- `hamiltonians.Hamiltonian` (`src/pyqula/hamiltonians.py`) — the central object almost everything hangs
  off of (bands, DOS, topology, transport, mean field, KPM...). The class itself is intentionally thin:
  nearly every method is a one-line delegator to a function in another module or a `*tk` subpackage, e.g.
  `get_bands` → `bandstructure.get_bands(self, ...)`, `get_chern` → `topology` module, `get_kdos_bands` →
  `kdos.kdos_bands(self, ...)`. When changing behavior, find the real implementation in the delegated-to
  module, not in the `Hamiltonian` method itself.
- Real-space hoppings between unit cells are stored as a `multihopping.MultiHopping`, essentially a dict
  keyed by lattice vector `(n1,n2,n3) -> hopping matrix`. `Hamiltonian` operator overloads (`+`, `*`, scalar
  mul) are implemented in `algebratk/hamiltonianalgebra.py` by combining the two Hamiltonians'
  `get_multihopping()` dicts and calling `set_multihopping()`.
- `Geometry`/`Hamiltonian` methods are frequently modified in place *and* returned, and `.copy()` is used
  heavily before mutating (`h1 = h0.copy(); h1.add_exchange(...)`) — follow that convention rather than
  assuming immutability or that a method returns a new object without side effects.

### The `*tk` subpackage convention

Most non-trivial functionality lives in `<topic>tk/` subpackages (e.g. `topologytk/`, `sctk/` for
superconductivity, `scftk/` for self-consistency, `kpmtk/`, `greentk/`, `transporttk/`, `dostk/`,
`geometrytk/`, `htk/` for low-level Hamiltonian internals like Bloch construction and supercells,
`operatortk/`, `wanniertk/`, `symmetrytk/`, `paralleltk/`, `algebratk/`). A top-level module of the same
name (e.g. `topology.py`) is typically the public-facing entry point that composes the `*tk` internals and
is what `Hamiltonian` methods call into. When asked to "add a feature to X", check both `X.py` and `Xtk/`.

### Performance backends: numba, dense/sparse

Hot inner loops have two possible backends, chosen automatically at import time — don't assume only numpy.
There used to be a third, opt-in Fortran/f2py backend (`src/pyqula/fortran/`, `compilefortran.py`); it was
removed since it required a manual compile step (`f2py`) that most installs never ran, so those code paths
were effectively dead. Every call site now goes straight through the pure-Python/numba path — if you see a
lone `*tk` module or comment mentioning a `use_fortran`/`f90` backend, that's leftover phrasing, not a real
branch to preserve.

- **numba**: used for jitted numeric routines; `parallel.py` centralizes thread-count configuration
  (`numba.set_num_threads`).
- **Dense vs. sparse linear algebra**: `limits.densedimension` (`src/pyqula/limits.py`, currently 10000) is
  the matrix-size cutoff for switching between dense (`scipy.linalg`) and sparse (`scipy.sparse.linalg`)
  diagonalization.
- Multiprocessing/parallelism (`paralleltk/`) across parameter sweeps (k-points, energies) runs through
  `parallel.pcall`/`parallel.set_cores(n)`, backed by a real `multiprocess.Pool` (`paralleltk/multiprocess.py`);
  default `cores=1` keeps it serial. `parallel.enabled` (default `True`) is a master switch — set it to
  `False` via `parallel.set_enabled(False)` to force the whole package to run strictly serially (no process
  pool, numba/BLAS threads clamped to 1), e.g. for debugging, profiling, or reproducibility; `set_enabled(True)`
  only lifts the restriction, it does not auto-restore a previous `cores` count.
- **For new parallel code, prefer numba `@jit(parallel=True)`/`prange` over `parallel.pcall`'s multiprocess
  pool.** Reach for `pcall` only when there's no alternative (e.g. the work isn't a numba-jittable loop —
  calls out to non-jitted Python/SciPy per item). Benchmarked on the KPM moment loop (`kpm.full_trace`,
  `kpm.random_trace` batched via `kpmtk/kpmnumba.py`'s `kpm_moments_batch`) over a 10,000-site sparse matrix:
  batching the per-vector loop into one `prange`-parallel numba kernel gave ~4-5x over serial, while
  `pcall`'s process pool was net *slower* than plain serial for the same workload (`ntries=40`, ~12ms/task —
  process spawn/IPC overhead exceeded the actual work). Multiprocessing still has a place for coarser-grained
  or non-jittable work, but it isn't the default first move anymore.

### Typical call pattern

```python
from pyqula import geometry
g = geometry.honeycomb_lattice()      # 1. build geometry
h = g.get_hamiltonian()               # 2. build tight-binding Hamiltonian
h.add_exchange([0.,0.,0.3])           # 3. add terms (onsite, Zeeman, SOC, pairing...) — mutates + returns
h2 = h.get_mean_field_hamiltonian(U=2.0, filling=0.15, mf="swave")  # 4. optional SCF interacting step
(k, e) = h2.get_bands()               # 5. compute an observable (bands, DOS, Chern, transport, KPM DOS...)
```

Junctions/transport compose two `Hamiltonian` leads via `heterostructures.build(h1, h2)`; impurities/defects
in infinite systems use `embedding.Embedding(h, m=h_with_defect)`.

### Wannierization (`wanniertk/`)

`h.get_wannier_hamiltonian(bands=[a,b], nk=...)` (`src/pyqula/wanniertk/wannierize.py`) Wannierizes a fixed,
contiguous range of `h`'s bands (0-indexed, both ends inclusive, Wannierized jointly as one group; no
disentanglement yet — that's a planned follow-up) and returns a new, smaller multicell `Hamiltonian` whose real-space hoppings
exactly reproduce that band subspace on the wannierization k-mesh. It's built on
[wannierpy](https://github.com/joselado/wannierpy)'s pure-Python Wannier90 port, bundled directly in this
repo at `src/pyqula/wanniertk/wannierpy/` (no Fortran source, no compiled extension — the pure-Python
backend needs neither; its only dependency is numpy, already required by pyqula) and imported normally by
`wannierize.py`, not as an optional backend. See `examples/wannier/get_wannier_hamiltonian/main.py` for a
runnable demo and `tests/wannier/` for correctness tests (exact-reproduction checks against the original
spectrum).

## Notes

- `src/pyqula/__init__.py` deliberately leaves all submodule imports commented out — always import
  submodules explicitly (`from pyqula import geometry`), not `import pyqula` and expect attributes to exist.
- `update.py` and `pipupdate.sh` are the maintainer's personal git-push / PyPI-publish shortcuts — not part
  of the library and not something to invoke on the user's behalf.
- When a change adds or materially changes a user-facing feature, update `documentation/user_guide.md`
  (and `README.md`'s FUNCTIONALITIES list where relevant) to describe it, following the existing style: a
  short prose section with the physics/motivation, a runnable code snippet, and — for anything with a
  method on `Hamiltonian`/`Geometry` — an entry in the "Main functions and methods" reference at the end of
  the user guide.
- `documentation/gpu_porting_plan.md` is a maintainer-facing roadmap (not started) for moving compute-heavy
  paths onto GPU via `jax` (already a hard dependency), covering batched dense diagonalization
  (`htk/eigenvectors.py`), the partially-started KPM GPU path (`kpmtk/kpmjax.py`/`kpmtk/kpmnumba.py`), and
  why sparse/ARPACK-based Green's-function work is a harder/lower-priority case. Check it before starting
  any GPU-related work in this repo.
