# Quantum geometry and Berry curvature

The Berry curvature is only one part of the geometry of the Bloch states. In
this page we compute the full quantum geometric tensor, whose other part is
the quantum metric, and then we resolve the Berry curvature in energy and in
real space.

## Quantum geometric tensor

The quantum geometric tensor is computed with
`h.get_quantum_geometric_tensor()`. Its antisymmetric part is the Berry
curvature. Its symmetric part is the quantum metric, which measures the
distance between neighboring Bloch states. In a flat band the metric is the
important quantity: it says how spread the states are, and it controls, for
example, the superfluid weight of a flat-band superconductor (see
[Superconductivity](superconductivity.md)).

For a chosen subspace $S$ of bands, for example the occupied bands, the
tensor is

$$
Q_{ij}^{mn}(\mathbf k) = \sum_{l \notin S}
\frac{\langle u_m|\partial_{k_i} H|u_l\rangle \langle u_l|\partial_{k_j} H|u_n\rangle}
{(E_m-E_l)(E_n-E_l)}, \qquad m,n \in S ,
$$

where $m$ and $n$ are bands of the subspace, $l$ runs over the bands outside
it, $|u\rangle$ are the Bloch states and $E$ their energies. The quantum
metric and the Berry curvature are

$$
g_{ij} = \mathrm{Re}\,\sum_{m\in S} Q_{ij}^{mm}, \qquad
\Omega_{ij} = -2\,\mathrm{Im}\,\sum_{m\in S} Q_{ij}^{mm} .
$$

Only states outside $S$ appear in the energy denominators. So the formula is
well defined when $S$ contains degenerate bands: a spin-degenerate pair, the
Kramers pairs of a model with inversion and time-reversal symmetry, or
several orbitals that meet at a high-symmetry point. The only case that
fails is a degeneracy between $S$ and a band outside it, and then the call
raises a `ValueError`. The derivative $\partial_{k_i} H$ is computed
analytically from the hoppings, without finite differences.

### The tensor of each pair of bands

The individual entries $Q_{ij}^{mn}$ are not physical. A rotation of the
states of $S$ among themselves, which is a free choice inside a degenerate
multiplet, also rotates the tensor. Only its trace and its invariants are
physical. For this reason `non_abelian=True` returns the tensor in the
orbital basis,

$$
Q_{ij}(\mathbf k) = \sum_{m,n\in S} |u_m\rangle\, Q_{ij}^{mn}\, \langle u_n|
= P\,\partial_{k_i}P\,\partial_{k_j}P\,P ,
$$

where $P$ is the projector on $S$. This depends only on the subspace, not on
the basis chosen inside it. Its trace is the tensor above, and in any basis
$|v_m\rangle$ of $S$ the tensor of each pair of bands is
$\langle v_m|Q_{ij}|v_n\rangle$.

### Coordinates

The momenta are in reduced coordinates, $\mathbf k = \sum_i k_i \mathbf b_i$,
with $\mathbf b_i$ the reciprocal lattice vectors and $k_i$ of period 1, as
everywhere in pyqula. The tensor in Cartesian momenta, the one to compare with
a continuum calculation, is $J Q J^T$ with
$J_{\alpha i} = (\mathbf a_i)_\alpha/2\pi$. `coordinates="cartesian"` returns
it, as a $3\times3$ tensor in $x$, $y$ and $z$. Only this tensor has the
symmetry of the crystal, because the reciprocal lattice vectors are in
general not orthogonal. For example, on the honeycomb lattice the trace of
the reduced metric is not $g_{xx}+g_{yy}$, and it is different at the three
M points.

### Positions of the orbitals

The band structure does not depend on where the orbitals are inside the unit
cell, but the Bloch states do, and so does their geometry. We can write the
Bloch Hamiltonian with the phase $e^{i\mathbf k\cdot\mathbf R}$ of the lattice
vector only, as if every orbital were at the origin of its cell, or with the
full bond vector, $e^{i\mathbf k\cdot(\mathbf R+\mathbf r_j-\mathbf r_i)}$.
The two choices give the same bands and the same Chern number, but a
different Berry curvature and quantum metric at each $\mathbf k$, and a
different Brillouin-zone integral of the metric. Only the second one is the
geometry of the crystal. It respects the point group: on the honeycomb lattice
with a Haldane flux and a sublattice imbalance, the Berry curvature is the
same at three $\mathbf k$-points related by a $C_3$ rotation, while with the
first choice it even changes sign between them. And it does not depend on the
choice of unit cell: the Brillouin-zone average of $\mathrm{Tr}\,g$, which is
the gauge-invariant spread of the Wannier functions of the subspace, doubles
exactly when the cell is doubled.

- `gauge="atomic"` (default): every orbital at its position in the geometry.
- `gauge="lattice"`: every orbital at the origin of its cell. This is the
  convention of part of the work on flat-band superconductivity, and the one
  to use to reproduce numbers from it. The dependence on the orbital
  positions is discussed by Simon and Rudner, Phys. Rev. B **102**, 165148.

### Code

```python
from pyqula import geometry
from pyqula import topology
g = geometry.honeycomb_lattice() # create honeycomb lattice
h = g.get_hamiltonian() # create hamiltonian of the system (spinful by default)
h.add_haldane(0.2) # Add Haldane coupling
h.shift_fermi(0.3) # put the Fermi level safely mid-gap (gap is [-0.9,0.9])

Q = h.get_quantum_geometric_tensor(k=[0.1,0.2,0.],occ_idxs=[0,1]) # at a k-point
g_metric = h.get_quantum_metric(k=[0.1,0.2,0.],occ_idxs=[0,1]) # quantum metric only

# non-Abelian tensor of the two occupied bands, in the orbital basis
Qna = topology.quantum_geometric_tensor(h,k=[0.1,0.2,0.],occ_idxs=[0,1],
        non_abelian=True)

# along a k-path, and integrated over the BZ: the integral of the Berry
# curvature part reproduces h.get_chern()
inds,gpath,omegapath = topology.quantum_geometric_tensor_path(h,occ_idxs=[0,1])
C = topology.chern_from_qgt(h,nk=20,occ_idxs=[0,1])
```

In two dimensions `Q` is a complex $2\times2$ array, one entry per pair of
directions, and the metric is its real part. With `non_abelian=True` it is a
$2\times2\times4\times4$ array: one $4\times4$ matrix in the spin-orbital basis
for each pair of directions. There is no spin-orbit coupling here, so this
matrix is block diagonal in spin, and the trace over the spin-up orbitals is
the contribution of the spin-up electrons. Along the k-path the function
returns the position along the path, and the metric and the curvature at each
point, one $2\times2$ tensor per k-point. The integral of the curvature over
the Brillouin zone gives the Chern number, here $C=2$ for the two
spin-degenerate occupied bands, the same as `h.get_chern()`.

- `occ_idxs`: the indices of the bands in $S$. By default, the bands with
  $E<0$, as in `h.get_chern()`, so it follows `h.shift_fermi()`.
- `k`: the k-point, in reduced coordinates.
- `non_abelian`, `coordinates`, `gauge`: as explained above.

The same functions work in one and three dimensions, where the tensor has one
and nine components, and `topology.quantum_geometric_tensor_mesh(h,nk=...)`
returns it on a uniform k-mesh.

See `examples/2d/quantum_geometric_tensor/main.py` for the spin-resolved
tensor above, `examples/2d/quantum_geometry/main.py` for the Berry curvature
and the metric of the occupied bands along a k-path, and
`jupyter-notebooks/functionalities/topological_characterization/06_quantum_geometric_tensor.ipynb`.

## Berry curvature density in energy

The Chern number is one integer for all the occupied bands. To see from which
energies it comes, we use `topology.chern_density()`. The Berry curvature of
the occupied states is written as an integral over energy,

$$
\Omega (\mathbf k) = \int_{-\infty}^{\epsilon_F} \Xi (\mathbf k,\omega)\, d\omega ,
$$

where $\Xi (\mathbf k,\omega)$ is the energy-resolved Berry curvature: how much
curvature the states at energy $\omega$ carry. It is built from the Green's
function with a broadening `delta`. `topology.chern_density()` integrates
$\Xi(\mathbf k,\omega)$ over the Brillouin zone at a set of energies, and also
returns its cumulative integral in energy.

```python
from pyqula import geometry
from pyqula import topology
import numpy as np
g = geometry.honeycomb_lattice() # create honeycomb lattice
h = g.get_hamiltonian() # create hamiltonian of the system
h.add_haldane(0.05) # Add Haldane coupling
(es,cs,csi) = topology.chern_density(h,nk=10,es=np.linspace(-1.0,1.0,40))
```

`es` are the energies, `cs` the Berry curvature density at each energy, and
`csi` its cumulative integral. Inside the gap `csi` has a plateau, and when
the integral goes over all the bands it comes back to zero, because the Berry
curvatures of all the bands of a Hamiltonian add up to zero. Read this
quantity as a qualitative profile in energy, not as a number. Its
normalization is not the one of `h.get_chern()`: for the snippet above the
plateau at zero energy is near $4$, twice the $C=2$ of
`h.get_chern()`. It also needs a fine enough `nk` and a large enough `delta`,
otherwise it shows spurious peaks near k-points where bands come close.

- `nk`: number of k-points per direction of the mesh.
- `es`: energies at which the density is computed.
- `delta`: broadening of the Green's function.
- `operator`: weight the curvature with an operator, as in `h.get_chern()`.
- `write`: if `True`, also write `CHERN_DENSITY.OUT` and
  `CHERN_DENSITY_INTEGRATED.OUT`.

The map of $\Xi(\mathbf k,\omega)$ over the Brillouin zone at one energy is
given by `topology.dOmega_dE_kmap(h,nk=40)`, which writes it to
`BERRY_DENSITY_KMAP.OUT`. See `examples/2d/berry_density/main.py`,
`examples/2d/chern_density/main.py` and `examples/2d/berry_density_kmap/main.py`,
and
`jupyter-notebooks/functionalities/topological_characterization/03_frequency_resolved_topological_density.ipynb`.

## Berry curvature density in real space

A Chern number needs a Brillouin zone. The Berry curvature can also be
resolved in real space, to see where in the sample the topological flux is.
This also works for an island, a disordered sample, or a supercell too large
for a k-mesh. It is computed with `topology.real_space_chern()`. The idea is

$$
\Omega (\mathbf k) = \int \Gamma (\mathbf k,\mathbf r)\, d^2 \mathbf r ,
$$

where $\Gamma (\mathbf k,\mathbf r)$ is the Berry curvature resolved in
position. In practice, for a finite system, pyqula computes a local Chern
marker of the Bianco-Resta type,

$$
C(\mathbf r) = \frac{2\pi}{A_{\mathbf r}}\,
\mathrm{Im}\,\langle \mathbf r|\, [PxP,\, PyP] \,|\mathbf r\rangle ,
$$

with $P$ the projector on the occupied states, $x$ and $y$ the position
operators, the diagonal element summed over the orbitals of the site
$\mathbf r$, and $A_{\mathbf r}$ the area of the site. The occupied states
are those below zero energy, so a Hamiltonian whose gap is not at zero must
be shifted first, for example with `h.set_filling(0.5)`. Deep in the bulk
this local quantity is the Chern number.

```python
from pyqula import islands
from pyqula import topology
g = islands.get_geometry(name="honeycomb",n=8,nedges=3) # a honeycomb island
h = g.get_hamiltonian() # create hamiltonian of the system
h.add_haldane(0.05) # Add Haldane coupling
(r,c) = topology.real_space_chern(h) # spatially-resolved Berry curvature
```

`r` are the positions of the sites and `c` the local marker at each site. The
call also writes `REAL_SPACE_CHERN.OUT`, one value per site, to plot as a map
over the island. The Hamiltonian must be finite; a periodic one raises a
`ValueError`. For a periodic system, build a large supercell and make it
finite first. How to read the marker, and how to get the Chern number from
it, is explained in [Topology in real space](real-space-topology.md).

See `examples/0d/real_space_chern/main.py` and
`jupyter-notebooks/functionalities/topological_characterization/04_spatially_resolved_topological_flux.ipynb`.
