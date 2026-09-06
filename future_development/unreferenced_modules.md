# Unreferenced modules

Status: the removals are **done**; what is left is a short list of judgement calls
recorded so they do not have to be re-derived.

## How "unreferenced" was established

`grep` is not good enough here: several module names in this package are also ordinary
English words (`density`, `states`, `waves`, `effective`, `reduce`, `merge`), so a
word-boundary grep over `src/`, `tests/` and `examples/` reports dozens of "references"
that are prose in comments and docstrings. It also *misses* real uses, because pyqula
examples import several modules on one line (`from pyqula import geometry,spintexture`),
which a naive `from pyqula import <name>` pattern does not match.

The reliable method is to parse every `.py` file under `src/`, `tests/`, `examples/`,
`benchmarks/` and `jupyter-notebooks/` with `ast`, collect every name that appears in an
`Import` or `ImportFrom` node (both the module and the imported symbols), and ask which
top-level modules of `src/pyqula/` never appear. There are no dynamic imports
(`importlib`, `__import__`, `exec`) anywhere in the package, so an unimported module is
genuinely unreachable.

That check found `waves.py` (used by `ldostk/ldoswaves.py`) and `spintexture.py` (used by
`examples/2d/spiral_texture_reciprocal_space_determinant/main.py`) to be *live*, both of
which a grep-based sweep had wrongly flagged.

## What was removed

15 modules, in the commit that added this note's sibling changes:

- the ten the audit sweep had annotated as "candidate for removal in a future cleanup"
  (`015b776` and its follow-up): `alloy`, `effective`, `estimators`, `fitting`, `mullen`,
  `numbaneighbor`, `reciprocalmap`, `slabs`, `junctions`, `surface_TI`
- five more that are both unreferenced *and* broken at runtime, so nothing outside the
  repo could have been calling them either: `gsenergy` (Python-2 `import klist` inside
  its one function), `hall` (undefined `hin`, `lg`, `nbands`, `csc_matrix`), `hexagonal`
  (reaches for an `h.ty`/`h.txy` Hamiltonian API that no longer exists, plus an undefined
  `check`), `reduce` (uses an unimported `np`), `symmetrize` (undefined `dagger` in five
  places, and it opened a debug file at import time)

`effective.py` could not even be imported (`import multicell`, Python-2 style).

## What was deliberately kept

These are unreferenced anywhere in the repo but import cleanly and are not obviously
broken, so removing them is a policy decision about the public API rather than a repair.
pyqula is a public package, and someone's script may well be doing
`from pyqula import density`:

| module | what it is | why it is unused here |
| --- | --- | --- |
| `data.py` | a trivial `Band` data holder, imports matplotlib at module scope | superseded by returning plain arrays |
| `density.py` | electron density in an energy window, arpack/dense | superseded by `ldos.py` and `densitymatrix.py` |
| `states.py` | writes eigenstates to files for 0d/2d | superseded by `ldos.py`'s writers |
| `hybrid.py` | glue two ribbons into one hybrid Hamiltonian | superseded by `heterostructures.py` |
| `merge.py` | merge two spinless Hamiltonians into one spinful one | still referenced by a comment in `htk/modify.py` |
| `massive_green.py` | filesystem bookkeeping for batches of Green's-function runs | a workflow helper, not physics |
| `spinwaves.py` | Holstein-Primakoff spin waves for a classical spin model | the magnon machinery in `chi.py`/`bse.py` covers the electronic case; this is the spin-model one, and `chitk/`'s route is what the tests exercise. One function also has an undefined `genij`. |

If the decision is ever taken to drop them, the cheapest path is a deprecation warning at
import for one release, then removal — not a silent delete, since none of them would fail
loudly for a user who has them in a script today.
