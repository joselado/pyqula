# Quantum geometry and Berry curvature

## Quantum geometric tensor (multiorbital/multiband)

The Berry curvature is only half of the geometry of the Bloch states, and we will now see how
to compute the whole of it, the quantum geometric tensor, with
`h.get_quantum_geometric_tensor()`. Its antisymmetric part is the Berry curvature, and its
symmetric part is the quantum metric, a measure of the distance between neighboring Bloch
states, which is the quantity that matters in a flat band, where the curvature alone says
nothing about how spread out the states are. For a chosen band subspace $S$ (e.g. the occupied
bands) it is

$$
Q_{ij}^{mn}(\mathbf k) = \sum_{l \notin S}
\frac{\langle u_m|\partial_{k_i} H|u_l\rangle \langle u_l|\partial_{k_j} H|u_n\rangle}
{(E_m-E_l)(E_n-E_l)}, \qquad m,n \in S
$$

where $m$ and $n$ run over the bands of the subspace and $l$ over the bands outside it, with
the quantum metric $g_{ij}^{mn} = \mathrm{Re}\,Q_{ij}^{mn}$ (symmetric part) and Berry
curvature $\Omega_{ij}^{mn} = -2\,\mathrm{Im}\,Q_{ij}^{mn}$ (antisymmetric part) recovered in
the band-trace ("Abelian") case. What about the tensor resolved per pair of bands, the
"non-Abelian" one? Note that $Q_{ij}^{mn}$ is not gauge independent: rotating the states of
$S$ among themselves, which is exactly the freedom a diagonalization has inside a degenerate
multiplet, rotates the tensor too, so its individual entries are not physical, only its trace
and its invariants are. For that reason `non_abelian=True` returns it in the orbital basis,

$$
Q_{ij}(\mathbf k) = \sum_{m,n\in S} |u_m\rangle\, Q_{ij}^{mn}\, \langle u_n|
= P\,\partial_{k_i}P\,\partial_{k_j}P\,P ,
$$

with $P$ the projector on $S$, meaning that it depends on the subspace alone and not on the
basis chosen inside it; its trace is the Abelian tensor, and the band-resolved tensor in any
basis $|v_m\rangle$ of $S$ you choose is $\langle v_m|Q_{ij}|v_n\rangle$.

What makes this form suitable for a genuinely multiorbital model is that only states *outside* $S$ enter the energy denominators, so it
stays well defined when $S$ contains an exactly or nearly degenerate multiplet of bands, an
exactly spin-degenerate pair, say, the Kramers pairs of a model with inversion and time-reversal
symmetry, or several orbitals meeting at a high-symmetry point, which an ordinary single-band
formula cannot handle. The derivative $\partial_{k_i} H$ is evaluated analytically from the
hoppings, with no finite-difference error, with $k$ in the same reduced (dimensionless,
period-1) coordinates as the rest of pyqula's k-space code, $\mathbf k = \sum_i k_i \mathbf b_i$
with $\mathbf b_i$ the reciprocal lattice vectors, so the tensor in Cartesian momenta, the one
to compare with a continuum calculation, is $J Q J^T$ with $J_{\alpha i} = (\mathbf a_i)_\alpha/2\pi$,
and this is what `coordinates="cartesian"` returns, a $3\times3$ tensor in $x$, $y$ and $z$
whatever the dimensionality. Only this one has the symmetry of the crystal, since the reciprocal
lattice vectors are not orthogonal in general: on the honeycomb lattice the trace of the reduced
metric is not $g_{xx}+g_{yy}$, and it differs between the three M points.
`occ_idxs` defaults to the bands with $E<0$, the same convention `h.get_chern()` uses, so it
tracks `h.shift_fermi(...)`.

Where do the orbitals sit inside the unit cell? The band structure does not care, but the
Bloch states do, and so does their geometry. Writing the Bloch Hamiltonian with the phase
$e^{i\mathbf k\cdot\mathbf R}$ of the lattice vector alone, as if every orbital sat at the
origin of its cell, or with the full bond vector, $e^{i\mathbf k\cdot(\mathbf R+\mathbf r_j-\mathbf r_i)}$,
gives the same bands and the same Chern number, since the two differ by a $\mathbf k$-dependent
phase on each orbital, but a different Berry curvature and quantum metric at each $\mathbf k$ and
a different Brillouin-zone integral of the metric. Only the second is the geometry of the
crystal, meaning that it respects its point group (on the honeycomb lattice with a Haldane flux
and a sublattice imbalance the Berry curvature is the same at three $\mathbf k$-points related
by a $C_3$ rotation, while the first convention even changes its sign between them) and that it
does not depend on how the crystal is divided into unit cells: the Brillouin-zone average of
$\mathrm{Tr}\,g$, which is the gauge-invariant spread of the Wannier functions of the subspace,
doubles exactly when the cell is doubled. `gauge="atomic"`, the default, places every orbital at
its position in the geometry, and `gauge="lattice"` drops the positions, which is the
convention of part of the work on flat-band superconductivity and the one to use to reproduce
numbers derived in it; "Superfluid weight and BKT temperature" takes the same keyword, and this
dependence on the orbital positions is discussed by Simon and Rudner, Phys. Rev. B **102**,
165148.

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

`Q` is a complex $2\times2$ array in two dimensions, one entry per pair of directions, and the
metric is its real part; with `non_abelian=True` it is a $2\times2\times4\times4$ array, one
$4\times4$ matrix in the spin-orbital basis per pair of directions, and since there is no
spin-orbit coupling here it is block diagonal in spin, the trace over the spin-up orbitals
being the contribution of the spin-up electrons. Along the k-path the function
returns the position along the path and the metric and the curvature at each point, arrays
with one $2\times2$ tensor per k-point, and integrating the curvature part over the Brillouin
zone reproduces `h.get_chern()`, here $C=2$ for the spin-degenerate pair of occupied bands. A
degeneracy between the chosen subspace and the bands outside it is the one thing the formula
cannot handle, and it raises a `ValueError` rather than dividing by a vanishing energy
difference. The same functions work in one and three dimensions, where the tensor has one and
nine components, and `topology.quantum_geometric_tensor_mesh(h,nk=...)` returns it on a uniform
k-mesh in any of them.

See `examples/2d/quantum_geometric_tensor/main.py` for a runnable version of the spin-resolved
tensor above, `examples/2d/quantum_geometry/main.py` for the Berry curvature and the metric of
the occupied bands along a k-path, and
`jupyter-notebooks/functionalities/topological_characterization/06_quantum_geometric_tensor.ipynb`
for an executed notebook.

## Berry curvature density in frequency space

The Chern number is a single integer for the whole set of occupied bands; we will now see how
to resolve where in energy it comes from, with `topology.chern_density()`. The Berry curvature
in frequency space is defined as 

$$
\Omega (\mathbf k) = \int_{-\infty}^{\epsilon_F} \Xi (\mathbf k,\omega) d\omega
$$

where $\Omega$ is the Berry curvature of the occupied bands and $\Xi (\mathbf k,\omega)$
is the energy-resolved Berry curvature, meaning that $\Xi$ says how much curvature the states
at each energy carry, and the Chern number is its integral up to the Fermi energy.
`topology.chern_density` integrates $\Xi(\mathbf k,\omega)$ over the whole Brillouin zone at a
set of energies, giving the frequency-resolved Berry-curvature density and its cumulative
(energy-integrated) sum

```python
from pyqula import geometry
from pyqula import topology
import numpy as np
g = geometry.honeycomb_lattice() # create honeycomb lattice
h = g.get_hamiltonian() # create hamiltonian of the system
h.add_haldane(0.05) # Add Haldane coupling
(es,cs,csi) = topology.chern_density(h,nk=10,es=np.linspace(-1.0,1.0,40))
```

`es` are the energies, `cs` the Berry-curvature density at each energy, and `csi` its
cumulative integral. In the gap `csi` should plateau at a value related to the total Chern
number of the occupied bands, and integrated over all the bands it returns to zero, since the
Berry curvatures of all the bands of a Hamiltonian cancel. This estimator is numerically
delicate: it needs a fine enough `nk` and a large enough `delta` to avoid spurious peaks near
nearly degenerate k-points, and its overall sign and normalization are not guaranteed to
match `h.get_chern()` (for the snippet above the plateau at zero energy sits near $4$, twice
the $C=2$ of the Chern number section). Read it as a qualitative profile in frequency, and
check any number it gives against `h.get_chern()`.

Optional arguments

- nk: number of k-points per direction of the mesh
- es: array of energies at which the density is evaluated
- delta: broadening of the Green's function the density is built from
- operator: restrict the curvature to a subspace, as for `h.get_chern()`
- write: if `True`, also write `CHERN_DENSITY.OUT` and `CHERN_DENSITY_INTEGRATED.OUT`

The k-resolved counterpart at a single energy, $\Xi(\mathbf k,\omega)$ over a full k-mesh, can
be obtained with `topology.dOmega_dE_kmap(h,nk=40)`, which writes the map to
`BERRY_DENSITY_KMAP.OUT`. See `examples/2d/berry_density/main.py`,
`examples/2d/chern_density/main.py` and `examples/2d/berry_density_kmap/main.py` for runnable
versions, and
`jupyter-notebooks/functionalities/topological_characterization/03_frequency_resolved_topological_density.ipynb`
for an executed notebook.

## Berry curvature density in real-space

A Chern number needs a Brillouin zone, but the Berry curvature can also be resolved in real
space, so that one sees where in the sample the topological flux is carried; this is what
makes it usable on an island, a disordered sample or a supercell too large for a k-mesh. We
will now see how to compute it with `topology.real_space_chern()`. The Berry curvature in
real-space is defined as 

$$
\Omega (\mathbf k) = \int \Gamma (\mathbf k,\mathbf r) d^2 \mathbf r
$$

where $\Omega$ is the Berry curvature of the occupied bands and $\Gamma (\mathbf k,\mathbf r)$
is the spatially-resolved Berry curvature. Note that this object is meaningful for periodic
systems with very large unit cells. In pyqula, the real-space Berry curvature is obtained from
a Bianco-Resta-type commutator of the position operators and the projector on the occupied
states, evaluated on a large real-space (0-dimensional) supercell or island

```python
from pyqula import islands
from pyqula import topology
g = islands.get_geometry(name="honeycomb",n=8,nedges=3) # a honeycomb island
h = g.get_hamiltonian() # create hamiltonian of the system
h.add_haldane(0.05) # Add Haldane coupling
(r,c) = topology.real_space_chern(h) # spatially-resolved Berry curvature
```

`r` are the site positions and `c` the local Berry-curvature marker at each site (the call
also writes `REAL_SPACE_CHERN.OUT`, one value per site, ready to be plotted as a map over the
island). The routine takes a finite Hamiltonian, and a periodic one raises a `ValueError`
naming the requirement: to use it on a periodic system, build a large supercell and open it
first. See `examples/0d/real_space_chern/main.py` for a runnable version and
`jupyter-notebooks/functionalities/topological_characterization/04_spatially_resolved_topological_flux.ipynb`
for an executed notebook.
