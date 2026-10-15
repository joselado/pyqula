# Ising models

Let us now take the lattice gas one step further, to Ising spins
$s_i\in\{-1,+1\}$ on a lattice, with `latticeising.LatticeIsing`. The spins
interact through a real-space coupling $-J_{ij}s_is_j$ and a site-dependent
field $-b_is_i$, the usual Ising Hamiltonian, with $J>0$ ferromagnetic
(favoring alignment). It mirrors `LatticeGas` closely, and shares its
`add_tensor()`, `regroup()` and `get_specific_heat()`/`get_susceptibility()`
helpers, but note that the sign convention of the energy is the **opposite**:
in `LatticeGas` the coupling $\sum J_{ij}n_in_j$ carries no minus sign, so a
positive $J$ there is a *repulsion*, while here a positive $J_{ij}$ in
`add_interaction()` means ferromagnetic alignment, the usual convention for
Ising spins.

The simplest case is a ferromagnet on a square lattice, which is bipartite, so
nothing frustrates the alignment:

```python
from pyqula import geometry
from pyqula import latticeising

g = geometry.square_lattice() # bipartite, so ferromagnetic order is not frustrated
g = g.get_supercell(12) # periodic, so no site sits at an edge

li = latticeising.LatticeIsing(g,m=0.0) # random +-1 spins, zero net magnetization
li.add_interaction(Jij=[1.]) # first-neighbor ferromagnetic coupling
es,ms = li.anneal(temps=[3.,1.,0.3,0.1,0.03],ntries=1e4) # simulated annealing
```

As in the lattice gas, the supercell is periodic, so that no site sits at an
edge, and `g.dimensionality = 0` would turn it into an open flake. `li.s`
holds the current $\pm1$ spin array, `es` the energy along the anneal
and `ms` the total magnetization $\sum_i s_i$ along it. What you see in `ms` is
the phase transition: at the high temperatures at the start of the schedule
the magnetization per site, `ms/li.nsites`, fluctuates around zero, and as the
temperature drops below the transition it grows to $+1$ or $-1$, all the spins
aligned, with the sign chosen by chance. `li.optimize_energy()` runs
single-spin-flip Metropolis dynamics, the usual Monte Carlo move for Ising
spins, in which the total magnetization is *not* conserved (it fluctuates under
`li.b`), so, mirroring `LatticeGas.optimize_grand_canonical()`, it returns
`(es, ms)`, the energy and total magnetization trajectories, the latter usable
directly with `latticegas.get_susceptibility()`. `li.optimize_conserved()`
instead uses Kawasaki spin-exchange (swap) dynamics, which *does* conserve the
total magnetization, the analog of the fixed-filling swaps of
`LatticeGas.optimize_energy()`. `li.anneal()` wraps `optimize_energy()` in a
decreasing-temperature schedule, and `optimize_energy_multistart()` keeps the
best of several independent restarts. `get_local_energy()` and
`get_local_field()` give the per-site energy and effective field of the current
snapshot, and `get_correlator()` and `get_structure_factor()` are the same
real- and reciprocal-space diagnostics `LatticeGas` has, locating ordered
ground states (ferromagnetic, checkerboard antiferromagnetic) and their
ordering wavevector. Note that `li.pairs` lists both directions of every bond
(the same convention as `LatticeGas`), so `get_energy()` is twice the usual sum
over unordered bonds: the critical temperature of the ferromagnet on the square
lattice sits near $2\times2.269$ in these units, not $2.269$.

See `examples/latticeising/` for runnable versions of the anneal, a
temperature scan (magnetization and specific heat), and maps of the local
energy and the local field of a frustrated antiferromagnet on the triangular
lattice, and
`jupyter-notebooks/functionalities/classical_spin_models_monte_carlo/04_ising_monte_carlo.ipynb` and
`05_correlators_thermodynamics.ipynb` in the same folder for executed notebooks.
