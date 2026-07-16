# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project context
When working with this codebase, prioritize readability over cleverness. Ask clarifying questions before making architectural changes.

## What this is

pyqula is a Python library for quantum tight-binding calculations on lattices: band structures, mean-field
(self-consistent) interacting Hamiltonians, topological invariants (Chern numbers, Z2, Berry curvature),
Green's function/spectral-function methods, Chebyshev polynomial (KPM) algorithms, and quantum transport
(NEGF, heterostructures/junctions).

## Install / build

```bash
pip install -e .                       # editable install from repo root (package lives in src/)
python src/pyqula/compilefortran.py    # optional: compile the f2py Fortran acceleration modules (needs f2py)
```

There is no separate lint config in this repo — no ruff/flake8/black config and no GitHub Actions workflow.
Don't assume tooling that isn't here.

## Tests

```bash
python -m pytest tests            # run the whole suite
python -m pytest tests/scf -v     # run one topic
```

`tests/<topic>/test_*.py` holds pytest tests that assert a physical/numerical invariant (e.g. that a
self-consistent mean-field result doesn't depend on the random initial guess used to seed it, or that two
independent code paths computing the same quantity agree to numerical tolerance). `pyproject.toml`'s
`[tool.pytest.ini_options]` puts `src` on `pythonpath` so `pyqula` resolves to `src/pyqula` — it must use
`--import-mode=importlib`, because the repo root directory is itself named `pyqula` and contains a stray
empty `__init__.py`; with the default import mode pytest's package-root walk from `tests/` would otherwise
resolve `import pyqula` to the repo root instead of `src/pyqula`. Some of these tests do a handful of
repeated SCF/RPA calculations to check invariance and take several seconds each — the full suite runs in well
under a minute.

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

### Performance backends: Fortran, numba, dense/sparse

Hot inner loops have three possible backends, chosen automatically at import time — don't assume only numpy:

- **Fortran via f2py**: modules under `src/pyqula/fortran/<name>/` compile to `.so` files (see
  `compilefortran.py` for the full folder→module list, e.g. `kpm`, `dos`, `berry`, `green`, `chi`,
  `density_matrix`, `algebra`). Call sites use a `try: from . import <x>f90; use_fortran = True except:
  use_fortran = False` pattern (see top of `kpm.py`) and branch on `use_fortran`. If the `.so` files aren't
  built, the pure-Python/numba path is used instead — both paths must stay correct.
- **numba**: used for jitted numeric routines; `parallel.py` centralizes thread-count configuration
  (`numba.set_num_threads`).
- **Dense vs. sparse linear algebra**: `limits.densedimension` (`src/pyqula/limits.py`, currently 10000) is
  the matrix-size cutoff for switching between dense (`scipy.linalg`) and sparse (`scipy.sparse.linalg`)
  diagonalization.
- Multiprocessing/parallelism (`paralleltk/`) across parameter sweeps (k-points, energies) is currently
  stubbed to run serially — see the comment "parallelization not working yet, workaround" in `parallel.py`
  before assuming `cores`/`pcall` actually parallelize.

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

## Notes

- `src/pyqula/__init__.py` deliberately leaves all submodule imports commented out — always import
  submodules explicitly (`from pyqula import geometry`), not `import pyqula` and expect attributes to exist.
- `update.py` and `pipupdate.sh` are the maintainer's personal git-push / PyPI-publish shortcuts — not part
  of the library and not something to invoke on the user's behalf.
