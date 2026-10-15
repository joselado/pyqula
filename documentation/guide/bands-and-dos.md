# Band structures and density of states

# Observables

With a Hamiltonian in hand, let us now ask it for the quantities that describe its
electronic structure: the band structure, the density of states and its local and
momentum-resolved versions, the Fermi surface, the quasiparticle interference pattern of a
defect, and the spin splitting of an altermagnet. Each of them is one method on the
Hamiltonian, and each returns the arrays a plot is made of.

## Electronic band structures

We will start with the band structure, with `h.get_bands()`. For any system that is periodic
in space, we can compute the electronic band structure as given by

$$
H = \sum_{k,\alpha} \epsilon_{k,\alpha} \Psi^\dagger_{k,\alpha} \Psi_{k,\alpha}
$$

where $\alpha$ is the band index, meaning that at each momentum $k$ the Hamiltonian is
diagonal with one energy per band. The previous calculation can be performed as

```python
from pyqula import geometry
g = geometry.honeycomb_lattice() # geometry of the 2D model
h = g.get_hamiltonian() # generate the Hamiltonian
(k,e) = h.get_bands() # compute band structure
```

`h.get_bands()` returns two arrays of the same length, the position along the k-path and one
energy per band at each k-point, so the two together are the scatter plot of the band
structure. For the honeycomb lattice what you see is the two bands touching at zero energy
at the K point, the Dirac point, with a bandwidth of three times the hopping on each side.

Optional arguments
- `kpath`: k-path to use, either a list of high-symmetry labels (e.g. `["G","K","M"]`) or explicit k-vectors; defaults to the geometry's standard path
- `nk`: number of k-points along the path
- `operator`: color/weight each band by the expectation value of an operator (or a list of operators), returning `(k,e,c)` instead of `(k,e)`
- `num_bands`: for large sparse Hamiltonians, only compute this many bands around `central_energy`, instead of the full spectrum

Passing an operator computes its expectation value in every eigenstate and returns it as a
third array, which is what colors the bands in a plot: the group velocity below tells apart
flat bands from dispersive ones, and the spin operator `"sz"` tells apart the two spin
channels of a magnet, which we will use in the mean-field chapter

```python
from pyqula import geometry
g = geometry.honeycomb_lattice()
h = g.get_hamiltonian(has_spin=True)
(k,e,c) = h.get_bands(operator="velocity") # bands colored by group velocity
```

Passing a list of operators computes the expectation value of each of them for every eigenstate, returning one extra column per operator (`(k,e,c1,c2,...)` instead of `(k,e,c)`)

```python
(k,e,c_sz,c_site) = h.get_bands(operator=["sz","site"]) # two operators at once
```

For a very large system, a moire supercell say, diagonalizing the full Hamiltonian at every
k-point is wasteful if only a handful of bands around the Fermi level are of interest.
Passing `num_bands` switches to a sparse solver that computes only those bands

```python
(k,e) = h.get_bands(num_bands=20) # only the 20 bands closest to central_energy
```

See `examples/2d/velocity_bands/main.py` and `examples/2d/strain_TBG/main.py` for runnable
versions, and `jupyter-notebooks/functionalities/single_particle_hamiltonians/04_state_resolved_bands.ipynb`
for an executed notebook with operator-colored bands.



## Density of states

We will now count the states rather than follow them along a path, with `h.get_dos()`. The
density of states counts how many states are in a certain energy window. It is defined as

$$
D(\omega) = \int \delta(\omega-\epsilon_k) dk
$$

where $\epsilon_k$ are the eigenenergies of the Hamiltonian, meaning that every band
contributes a plateau where it disperses and a peak where it is flat, so that the density of
states is where the van Hove singularities of a band structure show up. It can be used as
shown below

```python
from pyqula import geometry
g = geometry.triangular_lattice() # get the geometry
h = g.get_hamiltonian()  # get the Hamiltonian
(es,ds) = h.get_dos()
```

`h.get_dos()` returns the energies and the density of states at each of them, by default on
a window from $-4$ to $4$ in units of the hopping, sampled on a k-mesh and broadened with a
Lorentzian of width `delta`. For the triangular lattice what you see is a density of states
that is not symmetric in energy, with a van Hove singularity at $E=-2$, the saddle point of
the dispersion at the M point; the top of its band, at $E=6$, lies outside the default window,
so pass `energies=` when the whole band is needed.

Optional arguments
- energies: array with the energies for which the DOS is computed
- delta: smearing of the DOS
- operator: operator to which the DOS is projected
- mode: how the DOS is computed, `"ED"` (default), `"Green"`/`"RG"`, `"KPM"` or `"adaptive"`
- nk: number of k-points per direction of the mesh, for `"ED"` and `"KPM"`

The four modes are four ways of doing the same integral. `"ED"` diagonalizes the Hamiltonian
on a k-mesh and broadens the eigenvalues; `"Green"` and `"RG"` sum a Green's function at each
energy, which pays off when only a handful of energies are needed; `"KPM"` expands the
density of states in Chebyshev polynomials and is the choice for large sparse systems (see
the section on Chebyshev kernel polynomial methods); `"adaptive"` does not sample a mesh at
all: it integrates over the Brillouin zone with an error-controlled quadrature, tuned by
`error=1e-1`, and uses `nk` only as a limit on how far it subdivides. `"Green"` and `"RG"`
pass `nk` to the Brillouin-zone sum behind the self-energy, where it matters for
`gmode="full"` and, in two dimensions, for `gmode="renormalization"`, but not for the default
`gmode="adaptive"`.

Whatever the mode, the result is normalized as a density of states, meaning that integrating
it over the energies gives the number of states per unit cell: two for a spinful chain, and
all four modes reproduce that number to better than 1% on the same window, so they can be
compared with each other directly, and so can the density of states that `h.get_multildos()`
writes next to an LDOS map. On the spinful chain below, with longer-range hopping and Rashba
coupling, the two curves fall on top of each other

```python
from pyqula import geometry
import numpy as np
g = geometry.chain()
h = g.get_hamiltonian(tij=[0.5,0.,0.,0.5],has_spin=True)
h.add_rashba(0.7)
energies = np.linspace(-4.,4.0,60)
(e1,d1) = h.get_dos(energies=energies,delta=1e-2,mode="ED",nk=1000)
(e2,d2) = h.get_dos(energies=energies,delta=1e-2,mode="Green")
```

An operator can be passed to project the DOS onto a subspace, e.g. the sublattice-resolved
DOS of a gapped honeycomb lattice, where the sublattice imbalance puts the states below the
gap on one sublattice and the states above it on the other, so that the projected density of
states comes out with opposite signs on the two sides of the gap

```python
from pyqula import geometry
g = geometry.honeycomb_lattice()
h = g.get_hamiltonian(has_spin=True)
h.add_sublattice_imbalance(.4)
(es,ds) = h.get_dos(operator="sublattice",nk=40,delta=5e-2)
```

See `examples/1d/dos_GF/main.py` and `examples/2d/operator_dos/main.py` for runnable
versions, and `jupyter-notebooks/functionalities/single_particle_hamiltonians/06_operator_resolved_dos.ipynb`
for an executed notebook.
