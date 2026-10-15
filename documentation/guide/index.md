# pyqula

pyqula is a Python library for tight-binding models of electrons on
lattices. It computes band structures, densities of states and spectral
functions, self-consistent mean-field Hamiltonians for interacting
electrons, superconductivity, topological invariants, response functions and
quantum transport, and also classical spin and lattice-gas models.

This guide explains how to compute physical quantities. For each quantity it
gives the Hamiltonian or the formula, its physical meaning, the code, and the
keywords that matter. The code is on
[GitHub](https://github.com/joselado/pyqula), and the whole guide is also
available as one [PDF](https://joselado.github.io/pyqula/user_guide.pdf).

## Installation

```bash
pip install pyqula
```

pyqula is written in Python, with numba for the heavy loops, so it needs no
compiler. To use the code of a clone of the repository, run
`pip install -e .` from its root.

Note that `import pyqula` alone gives access to nothing. Always import the
module you need, for example `from pyqula import geometry`.

## A first calculation

Almost every calculation has four steps: we build a lattice, we generate its
tight-binding Hamiltonian, we add terms to it, and we compute an observable.

```python
from pyqula import geometry
g = geometry.honeycomb_lattice()   # 1. a lattice
h = g.get_hamiltonian()            # 2. its tight-binding Hamiltonian
h.add_zeeman([0.,0.,0.3])          # 3. add terms (in place)
(k,e) = h.get_bands()              # 4. compute an observable
```

Here the lattice is the honeycomb lattice of graphene, the Hamiltonian has
first-neighbor hopping $t=1$ and is spinful, and the Zeeman field
$B_z=0.3$ splits the two Dirac cones of the two spins by $0.6$.
`h.get_bands()` returns the k-points along a path in the Brillouin zone and
the energies at each of them. Energies are in units of the hopping $t$, and
lengths in units of the first-neighbor distance.

Each code block in this guide has its own imports and runs alone, except
when it continues the block just before it on the same page. The blocks stop
at the computed arrays, and the plotting is left to you.

## Where to find more

- `examples/` in the repository contains several hundred runnable scripts,
  ordered by dimension (`0d/ 1d/ 2d/ 3d/`) and by topic (`transport/`,
  `embedding/`, `wannier/`, `classicalspin/`, `latticegas/` and others).
  Most of them end with a figure.
- `jupyter-notebooks/functionalities/` contains one executed notebook for
  each functionality listed in the README, with the physics and the plots.
  Look there to see what a result should look like.
- The [Reference](reference-hamiltonian.md) tab lists the methods of
  `Geometry` and `Hamiltonian` and their arguments.
