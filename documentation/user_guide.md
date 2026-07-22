
# Setting up a Hamiltonian
In this basic tutorial we will address how to compute the band structure of a one dimensional tight binding model.

The Hamiltonian of a one dimensional tight binding chain takes the form

$$H = \sum_n c^\dagger_n c_{n+1} + h.c.$$

This model can be diagonalized analytically, giving rise to a diagonal Hamiltonian of the form

$$
H = \sum_k \epsilon_k \Psi^\dagger_k \Psi_k
$$

where energy momentum dispersion takes the form

$$
\epsilon_k = 2\cos{k}
$$

With the pyqula library, the previous band structure can be computed as

```python
from pyqula import geometry
g = geometry.chain() # geometry of the 1D chain
h = g.get_hamiltonian() # generate the Hamiltonian
(k,e) = h.get_bands() # compute band structure
```

## Including second and third neighbor hopping

By default, the Hamiltonian generated includes only first neighbor hopping
$t_1=1$. However, we may want to consider a generalized Hamiltonian of the form

$$
H = 
\sum_n c^\dagger_n c_{n+1} +
t_2\sum_n c^\dagger_n c_{n+2} +
t_3\sum_n c^\dagger_n c_{n+3} +
h.c.
$$

To compute the eigenvalues in this generalized model
taking $t_2 =0.2$ and $t_3=0.3$, we write

```python
from pyqula import geometry
g = geometry.chain() # geometry of the 1D chain
h = g.get_hamiltonian() # generate the Hamiltonian
(k,e) = h.get_bands(tij=[1.0,0.2,0.3]) # compute band structure
```

## Including an onsite energy

The Hamiltonian can have an onsite energy term, that is equivalent to a chemical potential
that takes the form

$$
H =
\mu \sum_n c^\dagger_n c_{n}
$$

This can be added to the Hamiltonian as

```python
from pyqula import geometry
g = geometry.chain() # geometry of the 1D chain
h = g.get_hamiltonian() # generate the Hamiltonian
mu = 0.3 # value of the onsite
h.add_onsite(mu) # add onsite energy
```

Possible inputs

- Float: the same onsite energy is added to all the sites

- Iterable (list or array): adds a different onsite energy to each site in teh geometry

- Callable (function): adds a different onsite energy to each site according to its location $\mathbf r$


## Including an external Zeeman field

In the following we will consider that we want to add an external Zeeman field to the electronic system. We now include the existence of a spin degree of freedom, considering the Hamiltonian

$$
H = H_0 +H_Z
$$

where $H_0$ is the original tight binding Hamiltonian

$$
H_0 = \sum_{n,s} c^\dagger_{n,s} c_{n+1,s} + h.c.
$$

and

$$
H_Z = \sum_{n,s,s'} \vec B \cdot \vec \sigma^{s,s'} c^\dagger_{n,s} c_{n,s'}
$$

with $n$ running over the sites and $s,s'$ running over the spin degree of freedom. The magnetic field takes the form $\vec B = (B_x,B_y,B_z)$, and
$\sigma_\alpha$ are the spin Pauli matrices. To add a magnetic field
of the form $\vec B = (0.1,0.2,0.3)$ to our chain we write

```python
from pyqula import geometry
g = geometry.chain() # geometry of the 1D chain
h = g.get_hamiltonian() # generate the Hamiltonian
h = g.add_zeeman([0.1,0.2,0.3]) # add the Zeeman field
(k,e) = h.get_bands() # compute band structure
```

## Including an external orbital field

An external magnetic field can be included using the Peierls substitution

$$
t_{\alpha \beta} \rightarrow t_{\alpha \beta} e ^{i\int_{r_\alpha}^{r_\beta} \vec A \cdot d \vec l}
$$

where $\vec A$ is the magnetic potential so that $\vec B = \nabla \times \vec A$. It can be used as shown in the example below

```python
from pyqula import geometry
N = 20 # number of unit cells as the width
g = geometry.square_ribbon(N) # ribbon
B = 0.02 # magnetic field in quantum flux unit
h.add_orbital_magnetic_field(B) # add an out-of plane magnetic field
```

## Setting a fiilling

If you want to enforce a certaing filling $\nu$ in a Hamiltonian, so that
$$
\langle c^\dagger_n c_n \rangle = \nu
$$

use 
```python
from pyqula import geometry
g = geometry.chain() # chain
h = g.get_hamiltonian()
h.set_filling(0.7) # enforce a filling
```

Possible inputs

- float: enforce the filling on average

- array: enforce that each site has a specific filling


# Observables

## Electronic band structures

For any system that is periodic in space, 
can compute the electronic band structure as given by

$$
H = \sum_{k,\alpha} \epsilon_{k,\alpha} \Psi^\dagger_{k,\alpha} \Psi_{k,\alpha}
$$

where $\alpha$ is the band index.

The previous calculation can be performed as

```python
from pyqula import geometry
g = geometry.honeycomb_lattice() # geometry of the 2D model
h = g.get_hamiltonian() # generate the Hamiltonian
(k,e) = h.get_bands() # compute band structure
```

Optional arguments
- `kpath`: k-path to use, either a list of high-symmetry labels (e.g. `["G","K","M"]`) or explicit k-vectors; defaults to the geometry's standard path
- `nk`: number of k-points along the path
- `operator`: color/weight each band by the expectation value of an operator (or a list of operators), returning `(k,e,c)` instead of `(k,e)`
- `num_bands`: for large sparse Hamiltonians, only compute this many bands around `central_energy` with ARPACK, instead of the full spectrum

```python
from pyqula import geometry
g = geometry.honeycomb_lattice()
h = g.get_hamiltonian(has_spin=True)
(k,e,c) = h.get_bands(operator="velocity") # bands colored by group velocity
```

For a very large system (e.g. a moire supercell), diagonalizing the full Hamiltonian at every k-point is wasteful if only a handful of bands around the Fermi level are of interest. Passing `num_bands` switches to a sparse ARPACK solver that targets only those bands

```python
(k,e) = h.get_bands(num_bands=20) # only the 20 bands closest to central_energy
```

See `examples/2d/velocity_bands/main.py` and `examples/2d/strain_TBG/main.py` for runnable versions.



## Density of states

The density of states counts how many states are in a certian energy window. It is defined as

$$
D(\omega) = \int \delta(\omega-\epsilon_k) dk
$$

where $\epsilon_k$ are the eigenenergies of the Hamiltonian. It can be used as shown below

```python
from pyqula import geometry
g = geometry.triangular_lattice() # get the geometry
h = g.get_hamiltonian()  # get the Hamiltonian
(es,ds) = h.get_dos()
```

Optional arguments
- energies: array with the energies for which the DOS is computed
- delta: smearing of the DOS
- operator: operator to which the DOS is projected
- mode: how the DOS is computed -- `"ED"` (default, broadens a k-mesh band structure), `"Green"` (sums a Green's function per energy, useful when only a handful of energies are needed), `"KPM"` (Chebyshev kernel-polynomial expansion, for large sparse systems -- see the "Chebyshev kernel polynomial (KPM) methods" section), or `"adaptive"`
- nk: number of k-points in the mesh (`"ED"`/`"KPM"` modes)

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

An operator can be passed to project the DOS onto a subspace, e.g. the sublattice-resolved DOS of a gapped honeycomb lattice

```python
from pyqula import geometry
g = geometry.honeycomb_lattice()
h = g.get_hamiltonian(has_spin=True)
h.add_sublattice_imbalance(.4)
(es,ds) = h.get_dos(operator="sublattice",nk=40,delta=5e-2)
```

See `examples/1d/dos_GF/main.py` and `examples/2d/operator_dos/main.py` for runnable versions.

## Local density of states

The density of states counts how many states are in a certian energy window. It is defined as

$$
D(\omega,n) = \int \delta(\omega-\epsilon_k) | \langle \Psi_k | n \rangle |^2 dk
$$

where $\epsilon_k$ are the eigenenergies of the Hamiltonian. It can be used as shown below

```python
from pyqula import geometry
g = geometry.hoenycomb_zigzag_ribbon() # get the geometry
h = g.get_hamiltonian()  # get the Hamiltonian
(x,y,d) = h.get_ldos()
```


Optional arguments
- e: energy at which the LDOS is evaluated
- delta: smearing of the LDOS
- operator: operator to which the LDOS is projected
- projection: `"TB"` (default, one value per lattice site), `"TBRS"` (same, but interpolated onto a continuous real-space map for smoother plotting), or `"atomic"` (projected onto atomic orbitals rather than tight-binding sites)
- num_bands: for large sparse Hamiltonians, use ARPACK to only compute this many states around the target energy

```python
from pyqula import geometry
g = geometry.honeycomb_zigzag_ribbon() # get the geometry
h = g.get_hamiltonian()  # get the Hamiltonian
(x,y,d) = h.get_ldos(e=0.0,projection="TBRS") # interpolated real-space map
```

`h.get_multildos()` computes the LDOS at many energies at once, writing one file per energy to a `MULTILDOS/` folder (useful for building an LDOS(x,y,E) movie/stack)

```python
import numpy as np
h.get_multildos(energies=np.linspace(-2.0,2.0,100),projection="atomic")
```

See `examples/0d/island/main.py` (single-energy, `projection="TBRS"`, superconducting island) and `examples/readme_examples/ldos_island/main.py` (`get_multildos`, `projection="atomic"`) for runnable versions.

## Momentum resolved spectral functions

Apart from the band structure, in certain cases it is interesting to compute the momentum resolved spectral function, that takes the form

$$
A(k,\omega) = \delta(\omega-\epsilon_k) | \langle \Psi_k | A | \Psi_k \rangle|^2
$$

where $A$ is a certain operator. The previous quantity allows define a heatmap of the momentum-resolved spectral function. For the example, in a superconducting state, if operator is chosen to be projection onto the electron-sector, the previous quantity shows the electronic spectral fucntion

```python
from pyqula import geometry
g = geometry.triangular_lattice() # get the geometry
h = g.get_hamiltonian()  # get the Hamiltonian
h.get_kdos_bands()
```

Optional arguments
- energies: array with the energies for which the DOS is computed
- delta: smearing of the DOS
- operator: operator to which the DOS is projected


## Fermi surfaces

For a 2D periodic Hamiltonian, `h.get_fermi_surface()` computes the spectral weight on a $(k_x,k_y)$ mesh at a single energy (by default the Fermi level, `e=0.0`), i.e. a single constant-energy cut

```python
from pyqula import geometry
g = geometry.triangular_lattice() # get the geometry
h = g.get_hamiltonian() # get the Hamiltonian
(kx,ky,fs) = h.get_fermi_surface(e=0.0,nk=50,delta=1e-1)
```

Optional arguments
- e: energy of the cut
- nk: number of k-points per direction
- delta: broadening
- operator: project/weight the Fermi surface by an operator, giving e.g. a spin- or valley-textured Fermi surface

```python
from pyqula.specialhamiltonian import NbSe2
h = NbSe2(soc=0.9) # multi-orbital spin-orbit-coupled Hamiltonian
(kx,ky,fs) = h.get_fermi_surface(e=0.,nk=100,delta=3e-1,operator="sz")
```

`h.get_multi_fermi_surface()` computes the same kind of map at many energies at once, writing one file per energy to a `MULTIFERMISURFACE/` folder -- convenient for scanning how the Fermi surface evolves away from the Fermi level

```python
import numpy as np
h.get_multi_fermi_surface(energies=np.linspace(-4,4,100),delta=1e-1)
```

Passing `operator="unfold"` together with `nsuper` unfolds the Fermi surface of a defective/disordered supercell back onto the primitive Brillouin zone (see the "Electronic structure folding and unfolding" section); as with QPI unfolding, the supercell must be built with `store_primal=True`

```python
from pyqula import geometry
g0 = geometry.triangular_lattice()
n = 3 # size of the supercell
g = g0.get_supercell(n,store_primal=True)
h = g.get_hamiltonian()
h.add_onsite(lambda r: 100.0 if np.linalg.norm(r-g.r[0])<1e-1 else 0.0) # a point defect

out = h.get_multi_fermi_surface(nk=50,energies=np.linspace(-4,4,100),delta=0.1,
        nsuper=n,operator="unfold")
```

See `examples/readme_examples/fermi_surface/main.py`, `examples/2d/operator_fermi_surface/main.py` and `examples/readme_examples/unfolding_FS/main.py` for runnable versions.

## Quasiparticle interference

Quasiparticle interference (QPI) maps the momentum-space scattering pattern that a defect or impurity produces, and is what an STM quasiparticle-interference measurement probes. `h.get_qpi()` is only available for 2D Hamiltonians; unlike the other observables here it does not return arrays -- it writes its output to disk, one file per energy in an output folder (default `MULTIQPI/`) plus a combined `DOS.OUT`

```python
import numpy as np
from pyqula import geometry
g = geometry.triangular_lattice()
h = g.get_hamiltonian(has_spin=False)
h.get_qpi(mode="pm",nk=50,delta=1e-1,energies=np.linspace(-6.,6.,100))
```

Optional arguments
- energies: array of energies to compute
- nk: number of k-points per direction
- delta: broadening
- mode: `"pm"` ("poor man's") autoconvolves the actual k-resolved spectral weight of the (possibly defective) system in q-space -- the physically meaningful QPI signal for a real scatterer; `"response"` (default) instead computes a cheaper Lindhard-like joint-DOS convolution from the clean band structure only, ignoring wavefunction form factors
- nunfold: for a defect embedded in an `nunfold`x`nunfold` supercell, unfold the QPI signal back onto the primitive Brillouin zone

A single point defect embedded in a supercell, with the resulting QPI unfolded back onto the primitive cell, is a realistic use case. The supercell must be built with `store_primal=True` so pyqula remembers the primitive-cell reference needed to unfold; `operator="unfold"` then resolves to the corresponding unfolding operator

```python
from pyqula import geometry
g0 = geometry.honeycomb_lattice()
ns = 2
g = g0.get_supercell(ns,store_primal=True)
h = g.get_hamiltonian(has_spin=False)
h.add_onsite(lambda r: 100.0 if np.linalg.norm(r-g.r[0])<1e-1 else 0.0) # a strong point defect

h.get_qpi(mode="pm",delta=1e-2,operator="unfold",nsuper=2,nk=140,nunfold=ns)
```

See `examples/2d/multiqpi/main.py` (clean system, `mode="pm"`) and `examples/2d/multiqpi_unfold/main.py` (defect in a supercell, unfolded) for runnable versions.


# Operators

Both when computing band structures, density of states and expectation values we could define operators to filter the results. In this section we elaborate on some important operators that are available, and we comment on their physical meaning.

Operators in pyqula have some important properties. First, for periodic Hamiltonian they can have an intrinsic momentum dependence. Second, pyqula allows for native algebra between them, namely they can be summed or multiplied, automatically accounting for intrinsic momentum depences. Third, they can be non-linear, providing a generalization of matrix operators.

## Spin operators

The simplest operators are the spin operators
$$
S_\alpha = \sum_n \sigma_\alpha^{\mu\nu} c^\dagger_{n,\mu} c_{n,\nu}
$$

with $\sigma_\alpha$ the Pauli matrices, that can be obtained as

```python
from pyqula import geometry
g = geometry.triangular_lattice() # get the geometry
h = g.get_hamiltonian()  # get the Hamiltonian
sx = h.get_operator("sx") # Spin x component
sy = h.get_operator("sy") # Spin y component
sz = h.get_operator("sz") # Spin z component
```

## Location operator

To understand the spatial location of the states we can use the spatial operators, that
denote where wavefucntions are located in real space
$$
R_\alpha = \sum_{r,s} r_\alpha c^\dagger_{r,s} c_{r,s}
$$

with $r_\alpha$ is the component of the position of site r

```python
from pyqula import geometry
g = geometry.triangular_lattice() # get the geometry
h = g.get_hamiltonian()  # get the Hamiltonian
x = h.get_operator("xposition") # x component
y = h.get_operator("yposition") # y component
z = h.get_operator("zposition") # z component
```


## Bulk-edge operator

In order to know if a state is located at the edge or in the bulk of the system
you can use the bulk-edge location operators. The edge operator takes value 1 for
sites on the edge, and 0 for sites in the bulk. 


$$
hat E  = \sum_{r\in \text{Edge},s} c^\dagger_{r,s} c_{r,s}
$$

The bulk
operator takes value 1 for sites on the bulk, and 0 for sites on the edge.

$$
hat B  = \sum_{r\in \text{Bulk},s} c^\dagger_{r,s} c_{r,s}
$$


```python
from pyqula import geometry
g = geometry.honeycomb_zigzag_ribbon() # get the geometry
h = g.get_hamiltonian()  # get the Hamiltonian
b = h.get_operator("bulk") # bulk operator
e = h.get_operator("edge") # edge operator
```

## Valley operator

For hoenycomb like systems, including aligned and twisted multilayers,
an operator that allows to extract the valley degree of freedom can be extracted.
This operator takes the form

$$
V = i \sum_{\langle \langle ij \rangle\rangle,s} \nu_{ij} \sigma_{ij}  c^\dagger_{r_i,s} c_{r_j,s}
$$

where $\nu = \pm 1$ and $\sigma = \pm 1$ for clockwise/anticlockwise, sublattice A/B. This
the so-called anti-Haldane hopping, and takes opposite values in opposite valleys.
It can can be obtained for honeycomb systems as

```python
from pyqula import geometry
g = geometry.honeycomb_zigzag_ribbon() # get the geometry
h = g.get_hamiltonian()  # get the Hamiltonian
vall = h.get_operator("valley") # valley operator
```

## In-plane valley operators

The operator above is the out-of-plane valley pseudospin $\tau_z$. The two remaining components of the valley pseudospin, $\tau_x$ and $\tau_y$, can also be obtained, giving access to the full valley vector $(\tau_x,\tau_y,\tau_z)$ -- the valley-space analogue of $(S_x,S_y,S_z)$ for real spin. They are built from a chiral Kekule coupling (symmetrized over the 3 inequivalent Kekule registries so the result is exactly $C_3$-covariant about every atom, not only about special high-symmetry points) rather than from the second-neighbor coupling behind $\tau_z$

```python
from pyqula import geometry
g = geometry.honeycomb_lattice().supercell(3) # Kekule-commensurate cell
h = g.get_hamiltonian(has_spin=False)  # get the Hamiltonian
taux = h.get_operator("valley_x") # tau_x
tauy = h.get_operator("valley_y") # tau_y
```

Both require a honeycomb-like geometry with a sublattice index; for a periodic (non-0d) Hamiltonian they additionally require a Kekule-commensurate cell (already a 3x3, or other multiple-of-3, supercell of the primitive honeycomb cell) to be well-defined -- a finite (0d) flake needs no such commensurability.

A single vacancy in an otherwise pristine honeycomb flake is an atomically-sharp, intervalley-scattering defect, and induces a vortex in the in-plane valley pseudospin around it -- a nice way to see $\tau_x,\tau_y$ in action

```python
from pyqula import islands
from pyqula import spectrum
g = islands.get_geometry(name="honeycomb",n=8,nedges=6)
gv = g.remove(g.get_central()[0]) # flake with a single vacancy
hv = gv.get_hamiltonian(has_spin=False)
dvx = spectrum.real_space_vev(hv,operator=hv.get_operator("valley_x"))
dvy = spectrum.real_space_vev(hv,operator=hv.get_operator("valley_y"))
```

`h.add_valley_exchange(v)`, with `v=(vx,vy,vz)`, adds a valley-space exchange term $\vec{v}\cdot(\tau_x,\tau_y,\tau_z)$ to the Hamiltonian -- the valley-pseudospin analogue of `add_exchange` for real spin

```python
from pyqula import geometry
g = geometry.honeycomb_lattice().supercell(3) # Kekule-commensurate cell
h = g.get_hamiltonian(has_spin=False)
h.add_valley_exchange([0.1,0.05,0.2]) # (vx,vy,vz)
```

See `examples/0d/valley_vortex_vacancy/main.py` and `examples/2d/valley_vortex/main.py` for runnable versions of the vacancy-vortex example.

## Nambu operators

In the presence of superconductivity, you can project onto the electron or
hole component of the Nambu spinor using the electron-hole operators

```python
from pyqula import geometry
g = geometry.triangular_lattice() # get the geometry
h = g.get_hamiltonian()  # get the Hamiltonian
e = h.get_operator("electron") # electron component
h = h.get_operator("hole") # hole component
```

## Berry curvature operator

The Berry curvature operator is a first example of an operator that is intrinsically momentum dependent. The Berry curvature operator is defined as

$$
 O |\Psi_k\rangle = \Omega(k,\epsilon_k) |\Psi_k \rangle
$$

where $\Omega(k,\omega)$ is the Berry curvature evaluated at the momentum $k$ and energy $\omega$ of the eigenstate $|\Psi\rangle$. In particular, this operator allows to directly see the contribution to the Berry curvature of different states in the band structure.

## Inverse participation ratio operator

So far we have considered operators that are linear, namely that fufill the condition

$$
A (|\Psi_1 \rangle + |\Psi_2 \rangle) = 
A |\Psi_1 \rangle + A|\Psi_2 \rangle 
$$

There is however one operator that it is interesting to consider that does not fufill such condition. The operator is the so-called inverse participation ration, which we define as

$$
 O |\Psi\rangle = \sum_i | \langle i | \Psi \rangle |^4 |\Psi \rangle
$$

In particular, the previous operator allows to identify states that are highly localized in a few lattice sites, becoming useful to highlight impurity states and localized modes.

```python
from pyqula import geometry
g = geometry.honeycomb_zigzag_ribbon() # get the geometry
h = g.get_hamiltonian()  # get the Hamiltonian
h.add_onsite(0.3) # add a sublattice imbalance
ipr = h.get_operator("IPR") # IPR operator
```



# Superconductivity
Up to now we have focused on Hamiltonians that contain only normal terms,
namely that the full Hamiltonian can be written as

$$
H_0 = \sum_{ijss'} t_{ijss'}c^\dagger_{i,s} c_{j,s'}
$$

where $ij$ runs onver sites and $ss'$ over spins.

In the presence of superconductivity, an anomalous term appears in the
Hamiltonian taking the form


$$
H_{SC} = \sum_{ijss'} \Delta_{ij}^{ss'}c_{i,s} c_{j,s'} + h.c.
$$

To solve the Hamiltonian

$$
H = H_0 + H_{SC}
$$

we define a Nambu spinor that takes the form

$$
\Psi_n =
\begin{pmatrix}
c_{n,\uparrow} \\
c_{n,\downarrow} \\
c^\dagger_{n,\downarrow} \\
-c^\dagger_{n,\uparrow} \\
\end{pmatrix}
$$


and rewrite the Hamiltonian as
$$
H = \Psi^\dagger \mathcal H \Psi
$$

where $\mathcal H$ is the nambu Hamiltonian. In this new basis, the
Hamiltonian can be written in a diagonal form as

$$
H = \sum_\alpha \epsilon_\alpha \Psi^\dagger_\alpha \Psi_\alpha
$$

where $\epsilon_\alpha$ are the Nambu eigenvalues.


## s-wave superconductivity

The simplest form of superconductivity is spin-singlet
s-wave superconductivity. A minimal superconducting term of this form
can be written as
$$
H_{SC} = 
\Delta_0
\sum_n 
c_{n,\uparrow} c_{n,\downarrow} + h.c.
$$

In the following, we address the electronic structure of a triangular
lattice with s-wave superconductivity, whose Hamiltonian takes the form

$$H = H_0 + H_{SC} $$

with 

$$H_0 = \sum_{\langle ij\rangle} c^\dagger_i c_{j} + h.c.$$

The previous Hamiltonian can be computed for $\Delta_0=0.2$ as

```python
from pyqula import geometry
g = geometry.triangular_lattice() # geometry of the 2D model
h = g.get_hamiltonian() # generate the Hamiltonian
h.add_swave(0.2) # add s-wave superconductivity
(k,e) = h.get_bands() # compute band structure
```
Note that due to the BdG nature of the Hamiltonian, the bandstructure shows both the electron and hole states



# Interactions at the mean-field level

In this section we address how interactions can be treated at the mean-field level.

## The collinear Hubbard model

We will start with the simplest interaction term, a local repulsive interaction in a spinful system. Our full Hamiltonian takes the form

$$
H = \sum_{\langle ij\rangle} c^\dagger_i c_{j} + h.c.
+
U\sum_{i} 
c^\dagger_{i,\uparrow} c_{i,\uparrow} 
c^\dagger_{i,\downarrow} c_{i,\downarrow} 
$$

The interaction term $U\sum_{i}c^\dagger_{i,\uparrow}c_{i,\uparrow}c^\dagger_{i,\downarrow}c_{i,\downarrow}$ is solved at the mean-field level. The mean-field approximation consists on replacing the previous four fermion operator, by all the terms that arise by taking expectation value in two of the fermions. In particular, in its simplest collinear form, the mean-field term takes the form

$$
H_U^{MF} = 
U\sum_{i} 
\langle c^\dagger_{i,\uparrow} c_{i,\uparrow} \rangle
c^\dagger_{i,\downarrow} c_{i,\downarrow} 
+
c^\dagger_{i,\uparrow} c_{i,\uparrow}
\langle c^\dagger_{i,\downarrow} c_{i,\downarrow} \rangle
$$

where $\langle\rangle$ denotes the ground state expectation value of those operators. The full Hamiltonian thus takes the form


$$
H^{MF} = 
\sum_{\langle ij\rangle} c^\dagger_i c_{j} + h.c.
U\sum_{i} 
\langle c^\dagger_{i,\uparrow} c_{i,\uparrow} \rangle
c^\dagger_{i,\downarrow} c_{i,\downarrow} 
+
c^\dagger_{i,\uparrow} c_{i,\uparrow}
\langle c^\dagger_{i,\downarrow} c_{i,\downarrow} \rangle
$$

As a result, the mean-field Hamiltonian depends on the specific ground state of the system, and the ground state depends of course on the specific mean-field Hamiltonian. The previous circular dependence between the ground state and the mean-field Hamiltonian gives rise to a selfconsistent problem. 

This selfconsistent condition is solved as follows. We start with an initial guess for the full many-body ground state, that we call $|GS_0\rangle$. With this initial state, we compute the mean-field Hamiltonian $H^{MF}_0$. This mean-field Hamiltonian allows to compute a new many-body ground state $|GS_1\rangle$, which in turn allows to compute a new mean-field Hamiltonian $H^{MF}_1$. The previous algorithm is represented as

$$
|GS_0\rangle
\rightarrow
H^{MF}_1
\rightarrow
|GS_1\rangle
\rightarrow
H^{MF}_1
\rightarrow
|GS_2\rangle
\rightarrow
H^{MF}_2
\rightarrow
...
$$

This iterative calculation is performed until $H^{MF}_n = H^{MF}_{n+1}$, at which point the algorithm has converged.

Two important notes can we taken from the previous approach. First, the final solution may be sensitive to the initial guess for the ground state. This guess corresponds to the initialization of the Hamiltonian, and it can be important for system whose eenrgy landscape has several local minima. A second point is that the update procedure from one iteration to the next can be done adiabatically, or very suddenly. This corresponds to the mixing between solutions, and for systems close to the critical point can lead to tricky convergence.

Let now show an example of a mean-field calculation. We will take now a square lattice, make a 2x2 supercell and include local repulsive interactions at half filling. The obtained ground state is an antiferromagnetic Neel state that opens a gap at half filling

```python
from pyqula import geometry
g = geometry.square_lattice() # geometry of a square lattice
g = g.get_supercell([2,2]) # generate a 2x2 supercell
h = g.get_hamiltonian() # create hamiltonian of the system
h = h.get_mean_field_hamiltonian(U=2.0,filling=0.5,
                            mf="random") # perform SCF
(k,e) = h.get_bands() # calculate band structure
m = h.get_magnetization() # get the magnetization
```


## Non-collinear Hubbard model
In the mean-field ansatz considered above, only a single term in the Wick contraction was considered. This term is the collinear term in the z-direction, and allows accounting for solutions that have magnetization in the z-direction. However, in the presence of frustration, external magnetic field or spin-orbit coupling, the magnetization of a system may be non-collinear and pointing in an arbitrary direction. To account for that phenomenology, the mean-field Hamiltonian must include the non-collinear term that takes the form

$$
H_U^{ncMF} = -
U\sum_{i} 
\langle c^\dagger_{i,\downarrow} c_{i,\uparrow} \rangle
c^\dagger_{i,\uparrow} c_{i,\downarrow} 
+ h.c.
$$

When including this additional term, the full mean-field Hubbard Hamiltonian is rotationally invariant, meaning that it respects SO(3) spin rotational symmetry. This rotationally symmetric form is the default form implemented in the library.

With the previous point in mind, we now solve a system that develops a non-collinear magnetic state. We take the square lattice considered in the section above, and we add an external magnetic field. The competition between Zeeman energy and antiferromagnetic correlations gives rise to a canted magnetic state

```python
from pyqula import geometry
g = geometry.square_lattice() # geometry of a square lattice
g = g.get_supercell([2,2]) # generate a 2x2 supercell
h = g.get_hamiltonian() # create hamiltonian of the system
h.add_zeeman([0.,0.,0.1]) # add out-of-plane Zeeman field
h = h.get_mean_field_hamiltonian(U=2.0,filling=0.5,
                                  mf="random") # perform SCF
(k,e,c) = h.get_bands(operator="sz") # calculate band structure
m = h.get_magnetization() # get the magnetization
```


## Superconducting mean-field

In the cases above we focused on local repulsive interactions that promote collinear or non-collinear magnetism. However, local interactions can promote a different type of symmetry breaking, in particular gauge symmetry breaking associated to superconductivity. The emergence of superconductivity is associated to one of the Wick contractions of the mean-field, the anomalous term, that takes the form


$$
H_U^{aMF} = 
U\sum_{i} 
\langle c_{i,\uparrow} c_{i,\downarrow} \rangle
c^\dagger_{i,\downarrow} c^\dagger_{i,\uparrow} 
+ h.c.
$$

The previous term in the mean-field Hamiltonian can become non-zero for $U<0$, and yields an interaction induced superconducting state. This term in the mean-field Hamiltonian is automatically accounted for in Hamiltonian with Nambu degree of freedom, of course apart from the collinear and non-collinear terms in the mean-field. We show below how an interaction induced superconducting state can be computed with pyqula

```python
from pyqula import geometry
import numpy as np
g = geometry.triangular_lattice() # geometry of a triangular lattice
h = g.get_hamiltonian()  # get the Hamiltonian
h.setup_nambu_spinor() # setup the Nambu form of the Hamiltonian
h = h.get_mean_field_hamiltonian(U=-1.0,filling=
                   0.15,mf="swave") # perform SCF
# electron spectral-function
h.get_kdos_bands(operator="electron",nk=400,
                   energies=np.linspace(-1.0,1.0,100))
```


## Long range interactions

Up to now we have considered interacting Hamiltonians that only have local (attractive or repulsive) Hubbard interactions. In the following we are going to consider systems that have many-body interactions also to a certain number of neighbors. Long range interactions are crucial to stabilize specific symmetry broken states, and in particular charge density waves, Peierls instabilities and unconventional superconductivity. The full Hamiltonian we will consider takes the form

$$
H = \sum_{\langle ij\rangle} c^\dagger_i c_{j} + h.c.
+
U\sum_{i}
c^\dagger_{i,\uparrow} c_{i,\uparrow}
c^\dagger_{i,\downarrow} c_{i,\downarrow}
+
V_1\sum_{\langle ij\rangle,s,s'} 
c^\dagger_{i,s} c_{i,s}
c^\dagger_{j,s'} c_{j,s'}
$$

where $U$ parametrizes onsite interactions and $V_1$ interactions between first neighbors. The previous Hamiltonian gives rise to a variety of terms when performing a mean-field decoupling. By default, pyqula includes all the Wick contractions of the mean-field, and in the presence of Nambu spinors it includes all the anomalous contractions. Let us now briefly elaborate on some of the additional terms that arise due to the first neighbor interaction $V_1$.

The first term is the charge order term, that takes the form

$$
H^{MF} \sim
\langle c^\dagger_{i,s} c_{i,s} \rangle
c^\dagger_{j,s'} c_{j,s'}
$$

this term can give rise to a different charge imbalance between different sites, and it leads to charge density wave states.

The second term we consider is the bond order, that takes the form

$$
H^{MF} \sim
\langle c^\dagger_{i,s} c_{j,s} \rangle
c^\dagger_{j,s} c_{i,s} + h.c.
$$

which leads to an interaction-enhanced hopping. If this happens in a non-uniform way in the system, the resulting state has a Peierls distortion.

Among the anomalous terms, the mean-field Hamiltonian can generate

$$
H^{MF} \sim
\langle c^\dagger_{i,\uparrow} c^\dagger_{j,\uparrow} \rangle
c_{j,\downarrow} c_{i,\downarrow} + h.c.
$$


$$
H^{MF} \sim
\langle c^\dagger_{i,\downarrow} c^\dagger_{j,\downarrow} \rangle
c_{j,\uparrow} c_{i,\uparrow} + h.c.
$$


$$
H^{MF} \sim
\langle c^\dagger_{i,\uparrow} c^\dagger_{j,\downarrow} \rangle
c_{j,\downarrow} c_{i,\uparrow} + h.c.
$$

where the first two-terms corresponds to the odd superconducting order, and the third term account both for even and odd orders.

Below, we show an example in which an interaction-induced spin-triplet term is generated. By considering a electronic structure with a large Zeeman splitting and attractive first neighbor interactions, a state with non-zero $\Delta_{\uparrow\uparrow}$ and $\Delta_{\downarrow\downarrow}$ emerges. 


```python
import numpy as np
from pyqula import geometry
g = geometry.triangular_lattice() # generate the geometry
h = g.get_hamiltonian() # create Hamiltonian of the system
h.add_exchange([0.,0.,1.]) # add exchange field
h.setup_nambu_spinor() # initialize the Nambu basis
# perform a superconducting non-collinear mean-field calculation
h = h.get_mean_field_hamiltonian(V1=-1.0,
                     filling=0.3,mf="random")
# electron spectral-function
h.get_kdos_bands(operator="electron",nk=400,
                   energies=np.linspace(-2.0,2.0,400))
```


# Spatially resolved density of states

```python
from pyqula import islands
g = islands.get_geometry(name="honeycomb",n=3,nedges=3) # get an island
h = g.get_hamiltonian() # get the Hamiltonian
h.get_multildos(projection="atomic") # get the LDOS
```



# Electronic structure folding and unfolding

Building a supercell folds the bands of the primitive cell into the smaller supercell Brillouin zone. The inverse operation, unfolding, recovers the primitive-cell-like spectral weight of a supercell calculation (e.g. a defect, a moire pattern, or a non-primitive choice of cell), and is essential to compare supercell calculations directly against ARPES-like band structures. Unfolding is implemented as a special operator, `"unfold"`, that projects onto the Bloch states of the primitive cell; it can be passed to any of the k-resolved observables (`h.get_bands()`, `h.get_kdos_bands()`, `h.get_multi_fermi_surface()`...). It requires the supercell to have been built keeping track of the primitive geometry

```python
from pyqula import geometry
import numpy as np
g = geometry.honeycomb_lattice() # primitive geometry
n = 3
gs = g.get_supercell(n,store_primal=True) # supercell, keeping the primitive cell info
h = gs.get_hamiltonian() # Hamiltonian of the supercell
(k,e,d) = h.get_kdos_bands(operator="unfold",delta=1e-1) # unfolded spectral function
```

`d` holds the unfolded spectral weight at each `(k,e)`; plotting a scatter of `k,e` colored/sized by `d` recovers the primitive-cell band structure out of the supercell calculation. The same `operator="unfold"` can be passed to `h.get_multi_fermi_surface()` to unfold constant-energy cuts. See `examples/2d/unfolding/main.py`, `examples/1d/unfolding/main.py` and `examples/readme_examples/unfolding_FS/main.py` for runnable versions.

# Surface spectral functions

In this section we address how we can compute the surface spectral function of a semi-infinite system, i.e. a system that is bulk-like far from a boundary but is cleaved along one direction. This is obtained from the surface Green's function, computed with a renormalization (decimation) technique for the semi-infinite bulk

```python
from pyqula import geometry
g = geometry.honeycomb_lattice() # create honeycomb lattice
h = g.get_hamiltonian() # create hamiltonian of the system
h.add_soc(0.05) # Kane-Mele spin-orbit coupling opens a topological gap
h.add_rashba(0.1) # break z-mirror symmetry to expose the edge states
(k,e,ds,db) = h.get_surface_kdos(delta=1e-2) # surface and bulk spectral functions
```

`ds` and `db` are, respectively, the surface and bulk spectral weight at each `(k,e)`; plotting `k,e` colored by `ds` shows the topologically-protected edge states living at the boundary, absent from the bulk spectrum `db`. See `examples/readme_examples/surface_2dTI/main.py` for a runnable version.


# Topological insulators

Here we provide a discussion of observables related with topological insulators

## Topological invariants

### Chern number

The Chern number characterizes two-dimensional topological insulators with broken time reversal symmetry. It is defined as

$$
C = \frac{1}{2\pi} \int \Omega (\mathbf k) d^2 \mathbf k
$$

where $\Omega$ is the Berry curvature. The Chern number can be computed with the following code

```python
from pyqula import geometry
from pyqula import kdos
g = geometry.honeycomb_lattice() # create honeycomb lattice
h = g.get_hamiltonian() # create hamiltonian of the system
h.add_haldane(0.05) # Add Haldane coupling
C = h.get_chern() # Chern number
```


### Z2 invariant

The Chern number characterizes two-dimensional topological insulators with time reversal symmetry. It can be computed with the following code

```python
from pyqula import geometry
from pyqula import kdos
g = geometry.honeycomb_lattice() # create honeycomb lattice
h = g.get_hamiltonian() # create hamiltonian of the system
h.add_soc(0.05) # Add spin-orbit coupling
from pyqula import topology
z2 = topology.z2_invariant(h) # Z2 invariant
```

## Berry curvature density in frequency space

The berry curvature in frequency space is defined as 

$$
\Omega (\mathbf k) = \int_{-\infty}^{\epsilon_F} \Xi (\mathbf k,\omega) d\omega
$$

where $\Omega$ is the Berry curvature of the occupiad bands and $\Xi (\mathbf k,\omega)$
is the energy-resolved Berry curvature. `topology.chern_density` integrates $\Xi(\mathbf k,\omega)$ over the whole Brillouin zone at a set of energies, giving the frequency-resolved Berry-curvature density and its cumulative (energy-integrated) sum

```python
from pyqula import geometry
from pyqula import topology
import numpy as np
g = geometry.honeycomb_lattice() # create honeycomb lattice
h = g.get_hamiltonian() # create hamiltonian of the system
h.add_haldane(0.05) # Add Haldane coupling
(es,cs,csi) = topology.chern_density(h,nk=10,es=np.linspace(-1.0,1.0,40))
```

`es` are the energies, `cs` the Berry-curvature density at each energy, and `csi` its cumulative integral. In the gap, `csi` should plateau at a value related to the total Chern number of the occupied bands, but this Green's-function-based estimator is numerically delicate (it involves a finite-difference k-derivative, so it needs a fine enough `nk`/`dk` and a large enough `delta` to avoid spurious peaks near quasi-degenerate k-points) and its overall sign/normalization is not guaranteed to match `h.get_chern()` -- treat it as a qualitative frequency-resolved profile and cross-check any quantitative reading against `h.get_chern()`. The k-resolved counterpart at a single energy, $\Xi(\mathbf k,\omega)$ over a full k-mesh, can be obtained with `topology.dOmega_dE_kmap(h,nk=40)`, which writes the map to `BERRY_DENSITY_KMAP.OUT`. See `examples/2d/berry_density_kmap/main.py` for a runnable version.

## Berry curvature density in real-space

The berry curvature in real-space is defined as 

$$
\Omega (\mathbf k) = \int \Gamma (\mathbf k,\mathbf r) d^2 \mathbf r
$$

where $\Omega$ is the Berry curvature of the occupiad bands and $\Gamma (\mathbf k,\mathbf r)$
is the spatially-resolved Berry curvature. Note that this object is meaningful for periodic
systems with very large unit cells. In pyqula, the real-space Berry curvature is obtained from a Bianco-Resta-type commutator of position and projector operators, evaluated on a large real-space (0-dimensional) supercell or island with `topology.real_space_chern`

```python
from pyqula import islands
from pyqula import topology
g = islands.get_geometry(name="honeycomb",n=8,nedges=3) # a honeycomb island
h = g.get_hamiltonian() # create hamiltonian of the system
h.add_haldane(0.05) # Add Haldane coupling
(r,c) = topology.real_space_chern(h) # spatially-resolved Berry curvature
```

`r` are the site positions and `c` the local Berry-curvature marker at each site (the call also writes `REAL_SPACE_CHERN.OUT`). See `examples/0d/real_space_chern/main.py` for a runnable version.

## Chern number in real-space

The berry curvature in real-space is defined as 

$$
C = \int F (\mathbf r) d^2 \mathbf r
$$

where $C$ is the total Chern number of the occupiad bands and $F (\mathbf r)$
is the spatially-resolved Chern number. Note that this object is meaningful for periodic
systems with very large unit cells. $F(\mathbf r)$ is the local marker computed by `topology.real_space_chern`; because it is built from a commutator $C = P X P Y P - P Y P X P$, its trace over the *entire* finite sample is exactly zero by construction, so summing it over every site is not how the invariant is recovered. Instead, deep in the interior of a large enough island -- away from the boundary, where the local environment looks like the infinite periodic bulk -- the marker plateaus at (approximately) the quantized bulk Chern number, while the edge sites carry compensating opposite-sign weight that cancels the bulk contribution exactly. This exact real-space cancellation is itself a manifestation of the bulk-boundary correspondence

```python
from pyqula import islands
from pyqula import topology
import numpy as np
g = islands.get_geometry(name="honeycomb",n=8,nedges=3) # a honeycomb island
h = g.get_hamiltonian() # create hamiltonian of the system
h.add_haldane(0.05) # Add Haldane coupling
(r,c) = topology.real_space_chern(h) # local marker, per site
r = np.array(r)
bulk = np.argsort(np.linalg.norm(r-r.mean(axis=0),axis=1))[:len(r)//4] # innermost sites
C = np.mean(c[bulk]) # bulk plateau value approximates the total Chern number
```

## Topological surface states

```python
from pyqula import geometry
from pyqula import kdos
g = geometry.honeycomb_lattice() # create honeycomb lattice
h = g.get_hamiltonian() # create hamiltonian of the system
h.add_haldane(0.05) # Add Haldane coupling
kdos.surface(h) # surface spectral function
```

## Topological markers

The real-space Berry curvature/Chern marker of the two sections above is an example of a topological marker: a local, position-resolved quantity, computable from ground-state projectors alone, that reveals a bulk topological invariant without relying on translational symmetry or a clean Brillouin zone. This makes topological markers well suited to disordered systems, finite flakes and islands, or systems with spatially varying parameters (e.g. a Haldane mass that changes sign across a boundary, or a topological insulator with dilute vacancies), where the marker density directly visualizes where the invariant is carried. See `topology.real_space_chern` above for the code that computes it.


# Response functions

Here we discuss how response functions can be computed

## Charge-charge response function

The charge-charge response function for a spinless system is computed as 
$$
\chi(\omega,i,j) = 
\sum_{n,m}
f(\epsilon_n) (1-f(\epsilon_m))
\frac{
\Psi_n(i)\Psi_m(j)
\Psi^*_m(i)\Psi^*_m(j)
}
{
\epsilon_n - \epsilon_m - \omega + i\delta
}
$$

where $f(\epsilon)$ is the Fermi-Dirac distribution

```python
from pyqula import geometry
g = geometry.chain() # create honeycomb lattice
h = g.get_hamiltonian() # create hamiltonian of the system
(es,chis) = h.get_chi(q=[0.,0.,0.]) # get response function
```

Optional arguments
- q: momentum transfer of the response function
- energies: array of frequencies
- delta: broadening
- nk: number of k-points used in the Brillouin-zone integration

The charge-charge response can also be scanned over momentum transfer to build a $\chi(q,\omega)$ map

```python
import numpy as np
g = geometry.chain()
h = g.get_hamiltonian()
qs = np.linspace(-1.,1.,40)
chis = [h.get_chi(q=[q,0.,0.],energies=np.linspace(-3,3,100),nk=40,delta=0.1)[1] for q in qs]
```

See `examples/1d/charge_response/main.py` for a runnable version.

## Generic operator-operator response function

`h.get_chi` computes the charge-charge response by default, but any pair of operators can be used instead, giving the generalized response

$$
\chi_{AB}(\omega,q) = \sum_{k,n,m}
f(\epsilon_{k,n}) (1-f(\epsilon_{k+q,m}))
\frac{
\langle \Psi_{k,n}|A|\Psi_{k+q,m}\rangle
\langle \Psi_{k+q,m}|B|\Psi_{k,n}\rangle
}
{
\epsilon_{k,n} - \epsilon_{k+q,m} - \omega + i\delta
}
$$

```python
from pyqula import geometry
import numpy as np
g = geometry.chain()
h = g.get_hamiltonian(has_spin=True)
sz = h.get_operator("sz") # any of the operators from the "Operators" section
(es,chi) = h.get_chi(A=sz,B=sz,energies=np.linspace(-2.,2.,100),nk=40,delta=0.1)
```

By default `A=B=`identity, which recovers the charge-charge response above. Any operator from the "Operators" section (spin, valley, sublattice, location, Nambu...) can be plugged in to build the corresponding susceptibility.

## RKKY response function

The RKKY (Ruderman-Kittel-Kasuya-Yosida) interaction between two magnetic impurities is the effective exchange coupling mediated by the itinerant electrons, and can be obtained from the same non-interacting response-function machinery. `rkky.rkky_map` computes it between a reference site and every other site in the system, as a function of distance

```python
from pyqula import geometry
from pyqula import rkky
g = geometry.chain()
h = g.get_hamiltonian()
h.add_onsite(1.)
m = rkky.rkky_map(h,n=10,mode="LR",nk=200) # linear-response RKKY vs distance
```

Optional arguments
- mode: `"LR"` (linear-response theory, using the same machinery as `get_chi`) or `"pm"` ("poor man's", computed by explicitly adding a small magnetic perturbation at each site and evaluating the total energy change)
- n: how many neighboring cells/distances to compute
- nk: number of k-points used in the Brillouin-zone integration

`m` is an array whose columns are `(distance, ..., ..., RKKY energy)`; the RKKY energy is in the last column

```python
x,e = m[:,0],m[:,3]
```

See `examples/1d/RKKY/main.py` and `examples/1d/rkky_minimal/main.py` (which compares `"pm"` and `"LR"` on the same system) for runnable versions.

## Spin susceptibility and RPA

The methods above compute the bare (non-interacting) response. For an interacting system, `h.get_spinchi_ladder` and `h.get_spinchi_full` dress the same response with the random phase approximation (RPA), using the Hubbard `U` stored on a mean-field Hamiltonian (`h.V`, see "Interactions at the mean-field level"), giving the physically relevant spin-excitation spectrum of e.g. a magnetically ordered state

```python
from pyqula import geometry
import numpy as np
g = geometry.chain()
h = g.get_hamiltonian(has_spin=True)
hmf = h.get_mean_field_hamiltonian(U=2.0,filling=0.5,mf="antiferro") # converge a magnetic state
(es,chis) = hmf.get_spinchi_ladder(energies=np.linspace(0.,2.,100),q=[0.1,0.,0.],nk=40,delta=2e-2)
```

- `get_spinchi_ladder` computes the transverse ($S^+/S^-$) response, i.e. spin-wave-like excitations
- `get_spinchi_full` computes the full $(S_x,S_y,S_z)$ tensor response
- `RPA=True` (default) dresses the response with the interaction; `RPA=False` returns the bare response
- `h.get_qdos_iets` scans `get_spinchi_full` over a q-path instead of a single q, directly giving a spin-excitation dispersion map along high-symmetry directions (needs a 2D lattice for the `"G","K","M"`-style path labels)

```python
g2 = geometry.honeycomb_lattice()
h2 = g2.get_hamiltonian(has_spin=True)
hmf2 = h2.get_mean_field_hamiltonian(U=2.0,filling=0.5,mf="antiferro")
qdisp = hmf2.get_qdos_iets(energies=np.linspace(0.,2.,100),qpath=["G","K","M"],nq=80,nk=40,delta=1e-2)
```

- `h.get_iets_ldos` instead computes the spatially resolved response at a single energy, i.e. a real-space map of the spin-flip ("inelastic tunneling spectroscopy", IETS) signal, which combined with `h.get_ldos` gives elastic + inelastic STM-like maps

See `examples/1d/rpa/main.py` (RPA spin response vs q for an antiferromagnetic chain), `examples/2d/rpa_triangular/main.py`/`examples/2d/rpa_honeycomb/main.py` (`get_qdos_iets` dispersion along a q-path) and `examples/0d/rpa_island/main.py`/`examples/0d/rpa_finite_chain/main.py` (`get_iets_ldos` real-space IETS maps) for runnable versions.

# Quantum transport

In this section we discuss how we can perform quantum transport calculations with pyqula.

## Magnetoresistence in metal-metal transport

As specific example, here we will address how we can compute magnetoresistence in transport between two magnetic metals. We build two copies of the same lead, give each one an exchange field pointing in a different direction, and compare the conductance of the parallel and antiparallel configurations

```python
from pyqula import geometry
from pyqula import heterostructures
import numpy as np
g = geometry.chain() # create the geometry
h = g.get_hamiltonian() # create the Hamiltonian
es = np.linspace(-.5,.5,50) # set of energies for dIdV
Gs = dict()
for name,m2 in [("parallel",[0.,0.,0.5]),("antiparallel",[0.,0.,-0.5])]:
    h1 = h.copy() ; h1.add_exchange([0.,0.,0.5]) # first lead, fixed magnetization
    h2 = h.copy() ; h2.add_exchange(m2) # second lead, parallel or antiparallel
    HT = heterostructures.create_leads_and_central(h1,h2,h1) # create the junction
    Gs[name] = [HT.didv(energy=e) for e in es] # calculate conductance
```

The magnetoresistance follows from the two conductance curves, e.g. $\mathrm{MR} = (G_P - G_{AP})/G_{AP}$ evaluated at the Fermi energy.

## Superconductor-metal transport

Here we address how transport between a superconducting lead and a metallic lead can be computed. As paradigmatic example, we will focus on the Andreev reflection regime and the tunneling regime

```python
from pyqula import geometry
from pyqula import heterostructures
import numpy as np
g = geometry.chain() # create the geometry
h = g.get_hamiltonian() # create the Hamiltonian
h1 = h.copy() # first lead
h2 = h.copy() # second lead
h2.add_swave(.01) # the second lead is superconducting
es = np.linspace(-.03,.03,100) # set of energies for dIdV
for T in np.linspace(1e-3,1.0,6): # loop over transparencies
    HT = heterostructures.build(h1,h2) # create the junction
    HT.set_coupling(T) # set the coupling between the leads
    Gs = [HT.didv(energy=e) for e in es] # calculate conductance
```


# Single defects in infinite systems

A single point defect or impurity embedded in an otherwise infinite, periodic system cannot be handled by a plain supercell calculation without artificially periodizing the defect. `embedding.Embedding` solves this properly with a Green's function embedding technique: it takes the pristine, periodic Hamiltonian `h` and a modified intracell matrix `m` describing the defect (or another `Hamiltonian` from which the modified matrix is taken), and gives access to the observables of the infinite system as perturbed by that single, non-periodic defect

```python
from pyqula import geometry
from pyqula import embedding
g = geometry.chain() # create the geometry
h = g.get_hamiltonian() # pristine, infinite Hamiltonian
hv = h.copy()
hv.add_onsite(lambda r: 1.0 if r[0]<0.01 else 0.0) # a single-site onsite defect
eb = embedding.Embedding(h,m=hv) # embed the defect in the infinite system
(x,y,d) = eb.get_ldos(energy=0.0,delta=1e-2,nsuper=200,nk=400) # LDOS around the defect
```

`get_ldos` returns the real-space positions and the LDOS profile in a window of `nsuper` unit cells around the defect, showing e.g. Friedel oscillations or bound/in-gap states induced by the impurity (it also writes `LDOS.OUT`; pass `write=False` to suppress that). `eb.get_dos()` gives the total DOS, `eb.multildos()` scans the LDOS over many energies (written to a `MULTILDOS/` folder), and `eb.get_didv()` computes transport through the embedded defect. See `examples/embedding/single_impurity_1D/main.py` and `examples/embedding/honeycomb_vacancy/main.py` for runnable versions, and the other scripts under `examples/embedding/` for further defect scenarios (vacancies, boundaries, Yu-Shiba-Rusinov states, self-consistent defects...).

# Wannierization

`h.get_wannier_hamiltonian()` Wannierizes a fixed, contiguous range of a
periodic Hamiltonian's bands and returns a new, smaller multicell
Hamiltonian whose real-space hoppings exactly reproduce that band subspace
on the wannierization k-mesh (there is no band disentanglement yet -- the
selected range is Wannierized jointly as one group).

As an example, consider a staggered honeycomb lattice, where a sublattice
potential opens a gap and gives a genuinely dispersive valence band to
Wannierize

```python
from pyqula import geometry
g = geometry.honeycomb_lattice()
h = g.get_hamiltonian(has_spin=False)
h.add_onsite([0.8,-0.8]) # sublattice potential opens a gap

# Wannierize just the lowest (valence) band: bands=[0,0] selects band
# index 0 at every k-point of a 12x12 Monkhorst-Pack wannierization mesh
hwan = h.get_wannier_hamiltonian(bands=[0,0],nk=12)

print("Number of Wannier functions:",hwan.intra.shape[0])
print("Wannier centres (Cartesian):\n",hwan.wannier_centres)
print("Wannier spreads:",hwan.wannier_spreads)
print("Total spread Omega:",hwan.wannier_spread_total)
```

The Wannierized Hamiltonian `hwan` behaves like any other pyqula
Hamiltonian, so its bands can be compared directly against the original
model's

```python
(k,e) = h.get_bands(write=False)
(kw,ew) = hwan.get_bands(write=False)
```

## Symmetry-enforced Wannierization

Passing `symmetries="auto"` makes `get_wannier_hamiltonian` check, before
Wannierizing, that the selected band range is a genuine union of
point-group-related multiplets everywhere on the mesh (point-group
operations are auto-detected from the geometry+Hamiltonian via
`symmetrytk.pointgroup.find_point_group`). A band selection that instead
slices through a symmetry-related degeneracy is rejected with a
`ValueError` rather than silently returning a mis-symmetrized model. A
list of explicit `symmetrytk.pointgroup.SymmetryOperation` can be passed
instead of `"auto"` to enforce a specific subgroup.

A good illustration is kagome's flat band: it is exactly degenerate with
the dispersive middle band at the K point, so no selection containing only
the flat band is a union of whole multiplets -- this is the well-known
topological obstruction behind kagome's flat band having no symmetric
exponentially-localized Wannier function, and the check catches it instead
of returning a broken model

```python
from pyqula import geometry
from pyqula.symmetrytk import pointgroup

g = geometry.kagome_lattice()
h = g.get_hamiltonian(has_spin=False)

try:
    h.get_wannier_hamiltonian(bands=[2,2],nk=12,symmetries="auto")
except ValueError as e:
    print("Flat band alone correctly rejected:",str(e).splitlines()[0])

# the full 3-band manifold has no such obstruction
hwan_sym = h.get_wannier_hamiltonian(bands=[0,2],nk=12,symmetries="auto")
print("Symmetries enforced:",[c.op.name for c in hwan_sym.wannier_symmetries])
```

See `examples/wannier/get_wannier_hamiltonian/main.py` and
`examples/wannier/symmetric_wannierization/main.py` for runnable versions
of these two examples.


# Chebyshev kernel polynomial (KPM) methods

For very large systems, exact diagonalization becomes impractical. Passing
`mode="KPM"` to observable methods switches to a stochastic Chebyshev
kernel polynomial expansion, which never builds the full spectrum and
scales to systems with millions of sites on a single core. It requires a
sparse Hamiltonian (`is_sparse=True`)

```python
from pyqula import geometry
import numpy as np
g = geometry.chain()
g = g.get_supercell(3000) # a big supercell
g.dimensionality = 0
h = g.get_hamiltonian(is_sparse=True,has_spin=False)
(x,y) = h.get_dos(mode="KPM",
            energies=np.linspace(-3.0,3.0,200), # energies
            delta=1e-4, # effective smearing (~1/npol)
            ntries=10 # number of random vectors for the stochastic trace
            )
```

The same expansion also gives non-local correlators and Green's functions
without inverting a matrix, through the lower-level `kpm` module

```python
from pyqula import kpm
(x,y) = kpm.dm_ij_energy(h.intra,npol=200,i=0,j=9,ne=1000)
```

See `examples/0d/kpm_dos/main.py` and `examples/0d/kpm_correlator/main.py`
for runnable versions, including a comparison of the KPM correlator against
the exact Green's function calculation.


# Main functions and methods

## Geometry functions and methods

### g.get_hamiltonian()
Generate the Hamiltonian from a geometry.

Optional arguments

- tij = [1.0,.0,0.]: List with 1st, 2nd, 3rd nearest neighbor hopping

Returns the Hamiltonian

### g.get_supercell()
Generate a supercell

Arguments

- N: size of the supercell to create, number or tuple

Optional arguments

- store_primal=False: keep a reference to the primitive-cell geometry on the supercell, needed by `operator="unfold"` (see "Electronic structure folding and unfolding")

Returns a new geometry

## Hamiltonian functions and methods

### h.get_bands()
Compute band structure

Optional arguments:

- nk = 20: number of k-points

Returns kpoint index and energies

### h.get_kdos_bands()
Compute a k-resolved spectral function (band structure dressed with a projection operator, or an unfolded spectral function) along a k-path.

Optional arguments:

- kpath: k-point path (auto-generated if not given)

- operator=None: operator used to weight the spectral function, e.g. `"unfold"` (see "Electronic structure folding and unfolding")

- energies, delta, nk: frequency range, broadening, k-point density

Returns k-path fraction, energy and spectral weight



### h.get_dos()
Compute the density of states.

Optional arguments:

- energies: array with frequencies of the DOS

- delta=0.01: broadening of the DOS

Return energies and DOS

### h.add_soc()
Add Kane-Mele intrinsic spin-orbit coupling

Arguments:

- value: value of the SOC


### h.add_zeeman()
Add a Zeeman field to the Hamiltonian

Arguments:

- value: value of the Zeeman, as a number (assumes [0,0,Bz]), array or callable function


### h.add_rashba()
Add Rashba spin-orbit coupling

Arguments:

- value: value of the Rashba SOC


### h.add_onsite()

Add a local onsite energy

Arguments:

- value: value of the onsite energy

### h.get_ldos()
Compute the local density of states.

Optional arguments:

- e: energy of the LDOS

- delta=0.01: broadening of the LDOS

- projection="TB": `"TB"`, `"TBRS"` (real-space interpolated) or `"atomic"`

Return x, position, y position and LDOS

### h.get_multildos()
Compute the LDOS at many energies, writing one file per energy to a `MULTILDOS/` folder.

Optional arguments:

- energies=linspace(-1,1,100): energies to compute

- projection="TB": `"TB"` or `"atomic"`

### h.get_chi()
Compute a non-interacting operator-operator response function (charge-charge by default).

Optional arguments:

- q=[0,0,0]: momentum transfer

- A=None, B=None: operators defining the response (default: identity, i.e. charge-charge)

- energies, delta, nk: frequency range, broadening, k-mesh density

Returns energies and the response function

### h.get_spinchi_ladder()
Compute the transverse ($S^+/S^-$) spin susceptibility, RPA-dressed by default using the Hubbard `U` of a mean-field Hamiltonian.

Optional arguments:

- q=[0,0,0], energies, delta, nk: as above

- RPA=True: dress with the random-phase approximation; `False` for the bare response

### h.get_fermi_surface()
Compute the spectral weight on a 2D k-mesh at a single energy.

Optional arguments:

- e=0.0: energy of the cut

- nk=50: k-points per direction

- delta: broadening

- operator=None: project/weight by an operator (e.g. `"sz"`, `"valley"`, `"unfold"`)

Returns kx, ky and the Fermi-surface weight

### h.get_multi_fermi_surface()
Compute the Fermi surface at many energies, writing one file per energy to a `MULTIFERMISURFACE/` folder.

Optional arguments:

- energies=[0.0]: energies to compute

- nk, delta, operator: as in `get_fermi_surface`

### h.get_surface_kdos()
Compute the surface and bulk spectral function of a semi-infinite system, from the surface Green's function (renormalization/decimation technique).

Optional arguments:

- kpath: k-point path (auto-generated if not given)

- energies, delta: frequency range, broadening

Returns k, energy, surface spectral weight and bulk spectral weight; also writes `KDOS.OUT`

### h.get_qpi()
Compute the quasiparticle-interference map (2D systems only). Writes output to disk (default `MULTIQPI/` folder plus `DOS.OUT`) rather than returning arrays.

Optional arguments:

- energies, nk, delta: as above

- mode="response": `"pm"` ("poor man's", autoconvolves the actual k-resolved spectral weight -- the physical QPI of a real scatterer) or `"response"` (cheaper Lindhard-like joint-DOS convolution of the clean bands)

- nunfold=1: unfold the QPI of a defect embedded in an `nunfold`x`nunfold` supercell back onto the primitive Brillouin zone

### h.get_chern()
Return Chern number of the Hamiltonian.

Optional arguments:
- nk=20: number of kpoints

### h.get_wannier_hamiltonian()
Wannierize a fixed range of bands and return the resulting real-space
Hamiltonian.

Arguments:

- bands = [a,b]: first and last band to Wannierize (0-indexed, both ends inclusive)

Optional arguments:

- nk=12: k-points per periodic direction for the wannierization mesh
- symmetries=None: `"auto"` to auto-detect and enforce the point group, or an explicit list of `symmetrytk.pointgroup.SymmetryOperation`

Returns a new, smaller Hamiltonian; `.wannier_centres`, `.wannier_spreads` and `.wannier_spread_total` hold the Wannier-function geometry

