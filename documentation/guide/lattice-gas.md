# Lattice gas models

We will now see how to model classical particles that either occupy a site or
do not, with `latticegas.LatticeGas`. The degrees of freedom are occupations
$n_i\in\{0,1\}$, adsorbates on a surface, vacancies in a crystal, or any
classical binary order parameter, and they interact through a real-space
coupling $J_{ij} n_i n_j$ and a site-dependent chemical potential $\mu_i n_i$,
so that a positive $J_{ij}$ is a repulsion between two occupied sites. As the
spin model of the previous chapter, it takes its lattice and neighbor shells
from a `Geometry` but is otherwise independent of the quantum `Hamiltonian`:
the energy is evaluated directly from the occupation array, and the ground
state is searched by Monte Carlo, swapping the occupations of pairs of sites
with the Metropolis rule at a given temperature, rather than by
diagonalization.

Let us take a triangular lattice with a third of the sites occupied and a
repulsion between first, second and third neighbors, so that the particles
have to arrange themselves as far from each other as they can:

```python
from pyqula import geometry
from pyqula import supercell
from pyqula import latticegas

g = geometry.triangular_lattice()
g = supercell.turn_orthorhombic(g)
g = g.get_supercell(10) # periodic, so no site sits at an edge

lg = latticegas.LatticeGas(g,filling=1./3.) # 1/3 of the sites randomly occupied
lg.add_interaction(Jij=[1.,1.,1.]) # first, second and third neighbor repulsion
es = lg.optimize_energy(temp=0.5,ntries=1e4) # simulated annealing
```

The filling fixes the number of particles, which the swap moves conserve.
The supercell keeps its periodicity, so the couplings wrap around its edges
and every site has the same neighbors, as in the bulk. Setting
`g.dimensionality = 0` before building the model cuts the bonds across the
edges instead, which is what several of the scripts in `examples/latticegas/`
do, and gives an open flake whose edge sites have fewer neighbors, so that
the order and the filling near them differ from those of the bulk.
`lg.den` holds the current 0/1 occupation array and `es` the energy along the
anneal, which decreases and then flattens once the particles have found their
arrangement; `temp` sets how often an uphill swap is accepted, and `ntries` the
number of attempted swaps. What that arrangement is can be read from
`get_local_energy()` and `get_local_mu()`, the per-site energy and
chemical-potential contribution of the current snapshot, and from
`get_correlator()` and `get_structure_factor()`, the density-density
correlator in real and in reciprocal space, whose peak locates the ordering
wavevector of an ordered ground state (a striped one, a honeycomb arrangement
of vacancies). A single run at one temperature can get stuck in a local
minimum: `anneal()` wraps `optimize_energy()` in a decreasing-temperature
schedule and keeps the best configuration seen along it, and
`optimize_energy_multistart()` keeps the best of several independent restarts.
`optimize_grand_canonical()` switches from fixed-filling swap moves to
single-site flips under `lg.mu`, letting the filling itself fluctuate, which is
the way to scan a phase diagram as a function of the chemical potential, or to
estimate thermodynamic quantities such as the specific heat
(`get_specific_heat()` and `get_susceptibility()`) from an equilibrium
trajectory at a fixed temperature. `add_tensor()` adds couplings beyond fixed
neighbor shells, and `write()` and `read()` checkpoint a snapshot to and from
disk.

See `examples/latticegas/` for runnable versions of the anneal, the
local-energy map, the correlators, and the grand-canonical scan of the filling
against the chemical potential, and
`jupyter-notebooks/functionalities/classical_spin_models_monte_carlo/03_lattice_gas_monte_carlo.ipynb` and
`05_correlators_thermodynamics.ipynb` in the same folder for executed notebooks.
