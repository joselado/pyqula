# CLAUDE.md

Guidance for Claude Code in this repository. This file loads in full into every session,
so it holds only what has to be in context before anyone thinks to ask for it: the
triggers for the project skills, the rules whose failure is expensive, and enough of the
architecture to orient a task. Reference material lives in `.claude/skills/` and loads on
demand; `future_development/` holds the roadmaps and the audit records.

## What this is

pyqula is a Python library for quantum tight-binding calculations on lattices: band
structures, mean-field (self-consistent) interacting Hamiltonians, topological invariants
(Chern numbers, Z2, Berry curvature), Green's function and spectral-function methods,
Chebyshev polynomial (KPM) algorithms, and quantum transport (NEGF, heterostructures and
junctions).

## Skills: load each one before the work it names, not after

The skill descriptions carry their own triggers, but an instruction in a file that is
always loaded does not depend on the description matching, so the triggers are restated
here.

- `new-feature`, before adding a method, formalism, observable or Hamiltonian term, and
  when asked whether a feature is finished. It has the rule that a formalism not already
  in the codebase starts from an arXiv paper and an open-source benchmark, where the code
  and the test go, and the five documentation surfaces that move with a user-facing
  feature.
- `error-conventions`, before writing any `raise` in `src/pyqula`, before adding or
  changing an option selected by a string, and before writing a guard on whether a
  Hamiltonian is spinful, has Nambu or has a sublattice. The package went through a
  deliberate sweep to reach one shape for its errors, and a hand-written guard undoes
  part of it. The short version: a real exception with a message naming what the
  routine requires, never a bare `raise`; a string-selected option lists its accepted
  values, derived from a registry rather than a hand-kept second list; Hilbert-space
  requirements go through the shared guards in `check.py`.
- `user-guide-voice`, before writing a single sentence of `documentation/user_guide.md`
  and before touching the FUNCTIONALITIES list in `README.md`. Prose written without it
  reads like a language model wrote it, which is the failure it exists to prevent.
- `refresh-docs`, when asked to refresh or rebuild the docs or the PDF guide, after
  landing a feature or a batch of fixes, and before a release. It recounts the suite,
  propagates every number that moved, re-runs `tests/documentation` and rebuilds the PDF.
- `gpu-backend`, before any GPU, jax or device work. Each tier of the porting plan wants
  the maintainer's explicit sign-off before it starts: propose, do not begin.
- `wannierization`, before working under `src/pyqula/wanniertk/` or calling
  `h.get_wannier_hamiltonian()`.

## How to work here

- Prioritize readability over cleverness, and ask before an architectural change.
- When a physics or numerical derivation (a mean-field, topology or Green's-function
  issue) is proving hard to resolve from the code and general knowledge alone, ask
  whether to pull a specific paper from arXiv rather than guess at the formalism.
- Always import submodules explicitly (`from pyqula import geometry`).
  `src/pyqula/__init__.py` deliberately leaves every submodule import commented out, so
  `import pyqula` alone does not expose the submodules.
- `update.py` and `pipupdate.sh` are the maintainer's personal git-push and PyPI-publish
  shortcuts, not part of the library and not something to run on the maintainer's behalf.
- There is no lint config (no ruff, flake8 or black) and no GitHub Actions workflow. Do
  not assume tooling that is not here.
- Check `future_development/README.md` before starting work in an area it covers: it
  indexes every roadmap and audit record with what each one settles and what it leaves
  open. When a piece of work leaves something deliberately unbuilt, write it up there
  with what was measured and add it to the index. Nothing from that index is copied
  here, since a second list only goes stale.

## Install

```bash
pip install -e .                       # editable install from repo root (package lives in src/)
```

## Tests

```bash
python -m pytest tests            # the whole suite
python -m pytest tests/scf -v     # one topic
```

**Do not pipe pytest's output** (`... | tail`, `... | grep`). The shell reports the
pipe's exit status, not pytest's, so a failed or even crashed run looks like success.
This is not hypothetical: it masked a fatal interpreter abort (numba's non-threadsafe
`workqueue` layer entered from two threads, fixed in `d759b32`) as exit code 0, and
separately made an `unrecognized arguments` error, which ran no tests at all, also report
exit code 0. To post-process the output, `set -o pipefail` first, or redirect to a file
in the session scratchpad and read that.

`tests/<topic>/test_*.py` asserts a physical or numerical invariant (a self-consistent
result that does not depend on the random initial guess, two independent code paths
agreeing to tolerance), not a recorded number. `pyproject.toml` puts `src` on
`pythonpath` and must use `--import-mode=importlib`: the repo root is itself named
`pyqula` and holds a stray empty `__init__.py`, so the default import mode would resolve
`import pyqula` to the repo root instead of `src/pyqula`.

The suite collects **2644 tests** (`pytest tests --collect-only -q`). The slowest
individual tests (SCF and RPA, jax Newton solvers, Keldysh transport) run 10 to 25 s
each. The last whole-suite measurement on an idle machine was 37:34 at 1966 tests, so
budget more than that now, with `tests/scf` alone about 15 min and `tests/keldysh` about
12 min. A timing taken while other jobs run is meaningless. When tailing a running suite,
pytest writes its progress without a newline until a 72-character line fills, so
`tail -c` on a redirected log shows the same stale chunk for minutes and a healthy run
looks stalled.

`examples/` holds runnable `main.py` scripts that double as usage documentation,
organized by dimensionality (`0d/ 1d/ 2d/ 3d/`) and by topic (`transport/`,
`embedding/`, `wannier/`, `classicalspin/`, `latticegas/`, `kondolattice/` and others).
Grep there for an example of a feature before implementing anything from scratch.

## Architecture

### Core objects and where behavior lives

- `geometry.Geometry` (`src/pyqula/geometry.py`) holds atomic positions and lattice
  vectors, built by factory functions such as `geometry.chain()`,
  `geometry.honeycomb_lattice()` or `geometry.kagome_lattice()`.
  `Geometry.get_hamiltonian()` builds a `Hamiltonian` from it, with first-neighbor
  hopping by default.
- `hamiltonians.Hamiltonian` (`src/pyqula/hamiltonians.py`) is the central object that
  almost everything hangs off (bands, DOS, topology, transport, mean field, KPM). The
  class is deliberately thin: nearly every method is a one-line delegator to a function
  in another module or a `*tk` subpackage, e.g. `get_bands` to `bandstructure.get_bands`,
  `get_chern` to `topology`, `get_kdos_bands` to `kdos.kdos_bands`. When changing
  behavior, find the real implementation in the delegated-to module, not in the method.
- Real-space hoppings between unit cells are stored as a `multihopping.MultiHopping`,
  essentially a dict keyed by lattice vector `(n1,n2,n3) -> hopping matrix`. The
  `Hamiltonian` operator overloads (`+`, `*`, scalar multiplication) live in
  `algebratk/hamiltonianalgebra.py` and combine the two `get_multihopping()` dicts
  before calling `set_multihopping()`.
- `Geometry` and `Hamiltonian` methods are frequently modified in place *and* returned,
  and `.copy()` is used heavily before mutating (`h1 = h0.copy(); h1.add_exchange(...)`).
  Follow that convention rather than assuming immutability or a side-effect-free return.

### The `*tk` subpackage convention

Most non-trivial functionality lives in `<topic>tk/` subpackages (`topologytk/`, `sctk/`
for superconductivity, `scftk/` for self-consistency, `kpmtk/`, `greentk/`,
`transporttk/`, `dostk/`, `geometrytk/`, `htk/` for low-level Hamiltonian internals such
as Bloch construction and supercells, `operatortk/`, `wanniertk/`, `symmetrytk/`,
`paralleltk/`, `algebratk/`). A top-level module of the same name (`topology.py`) is the
public entry point that composes the `*tk` internals and is what the `Hamiltonian`
methods call. When asked to add a feature to X, check both `X.py` and `Xtk/`.

### Performance backends

- numba jits the hot inner loops; `parallel.py` centralizes the thread count
  (`numba.set_num_threads`).
- `limits.densedimension` (`src/pyqula/limits.py`, currently 10000) is the matrix-size
  cutoff between dense (`scipy.linalg`) and sparse (`scipy.sparse.linalg`)
  diagonalization.
- Parameter sweeps (k-points, energies) parallelize through `parallel.pcall` and
  `parallel.set_cores(n)`, backed by a `multiprocess.Pool` in
  `paralleltk/multiprocess.py`; the default `cores=1` is serial.
  `parallel.set_enabled(False)` is the master switch that forces the whole package
  serial (no pool, numba and BLAS threads clamped to 1) for debugging, profiling or
  reproducibility; `set_enabled(True)` only lifts the restriction and does not restore a
  previous `cores` count.
- For new parallel code prefer numba `@jit(parallel=True)` with `prange` over the
  `pcall` pool, and reach for `pcall` only when the work is not numba-jittable (it calls
  non-jitted Python or SciPy per item). On the KPM moment loop over a 10,000-site sparse
  matrix the batched `prange` kernel gave 4 to 5x over serial while the pool was net
  slower than serial; the measurement is recorded in `future_development/bug_audit_2.md`.
- There is no Fortran backend any more; a comment mentioning a routine ported from
  Fortran is history, not a branch to preserve.

### Typical call pattern

```python
from pyqula import geometry
g = geometry.honeycomb_lattice()      # 1. build geometry
h = g.get_hamiltonian()               # 2. build tight-binding Hamiltonian
h.add_exchange([0.,0.,0.3])           # 3. add terms (onsite, Zeeman, SOC, pairing...): mutates and returns
h2 = h.get_mean_field_hamiltonian(U=2.0, filling=0.15, mf="ferro")  # 4. optional SCF interacting step
# (a superconducting guess needs the Nambu degree of freedom first:
#  h.setup_nambu_spinor(); h.get_mean_field_hamiltonian(U=-2.0, ..., mf="swave"))
(k, e) = h2.get_bands()               # 5. compute an observable (bands, DOS, Chern, transport, KPM DOS...)
```

Junctions and transport compose two `Hamiltonian` leads via
`heterostructures.build(h1, h2)`; impurities and defects in infinite systems use
`embedding.Embedding(h, m=h_with_defect)`.

## Package-wide switches

- **The CPU/GPU backend is one switch, `src/pyqula/gpu.py`**, defaulting to the CPU on
  every machine. A new GPU path routes on `gpu.get_gpu()` rather than growing a switch
  of its own; the `gpu-backend` skill has the precision arguments and the tiered plan.
- **Files written to the working directory go through one switch,
  `src/pyqula/filewrite.py`.** A routine that writes an output file takes `write=None`
  and resolves it first thing with `write = filewrite.resolve(write, <its own default>)`,
  so that a `write=` in the call wins, then `filewrite.set_write()`, then the routine's
  default. A literal `write=True` default would pass `True` down explicitly and override
  the switch. Internal calls that pass `write=False` on purpose (intermediate
  computations) keep it. When a routine writes a file and reads it back, build the
  result in memory instead, so `write=False` still works.

## HPC-cluster material never goes into git

pyqula is a public repository; the maintainer's cluster details (login hosts, scratch
paths, partition names, queue measurements, account-specific job scripts, run logs) are
none of the public's business and must not reach GitHub. They live in `docs/` and in
`CLAUDE.local.md`, both gitignored for exactly this reason. Keep them there, and never
`git add -f` them, move their contents into a tracked file, or quote cluster specifics
into a commit message, a docstring, `documentation/` or `future_development/`.
Performance *conclusions* are welcome in the tracked roadmaps ("the batched dense solve
is the GPU-favourable shape"); the hostnames, paths and job IDs that produced them are
not.
