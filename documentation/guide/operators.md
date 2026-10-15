# Operators

When computing band structures, densities of states or expectation values we can pass an
operator to filter the result, so that every state carries the expectation value of that
operator next to its energy. Let us now go through the operators that are available, and
comment on their physical meaning.

Operators in pyqula have some important properties. First, for periodic Hamiltonians they can
have an intrinsic momentum dependence. Second, pyqula allows for native algebra between them,
namely they can be summed or multiplied, automatically accounting for intrinsic momentum
dependences. Third, they can be non-linear, providing a generalization of matrix operators.

Every operator below is obtained with `h.get_operator(name)`, and goes into the `operator=`
argument of `h.get_bands()`, `h.get_dos()`, `h.get_ldos()`, `h.get_kdos_bands()` or
`h.get_vev()`: the band structure comes back with one extra column, the expectation value of
the operator in each eigenstate, the density of states becomes the density of states weighted
by that expectation value, and so on. The executed notebooks
`jupyter-notebooks/functionalities/single_particle_hamiltonians/04_state_resolved_bands.ipynb`
and `06_operator_resolved_dos.ipynb` show what an operator-resolved band structure and density
of states look like. Before the operators themselves, the first section takes up a quantity
that is not a matrix operator at all, the nonlinear spin current, which continues the
altermagnet section of the previous chapter.



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

Passing one of them to `h.get_bands(operator=sz)` colors each band by its spin projection,
which is how a spin-split band structure, the one of the Zeeman field in the first chapter,
is told apart from a spin-degenerate one; the three names also work directly as strings,
`operator="sz"`. The same operators resolve the density of states, the Fermi surface and the
momentum-resolved spectral function by spin, see `examples/2d/operator_fermi_surface/main.py`
for a spin-resolved Fermi surface and the executed notebook
`jupyter-notebooks/functionalities/spectral_functions/06_operator_momentum_resolved.ipynb`
for a spin-resolved spectral function.

## Listing the accepted names

Every routine that takes an `operator` argument (`h.get_bands()`, `h.get_dos()`,
`h.get_ldos()`, `h.get_kdos_bands()`, `h.get_vev()`, the topological invariants) accepts an
`Operator`, a `Hamiltonian`, a `Potential`, a callable of the position, a bare matrix, or the
name of one of the operators pyqula knows how to build. Named operators are not the only thing
selected by a string: so are the mean-field initialization (`mf=`), the superconducting
pairing symmetry (`mode=`), the high-symmetry k-point labels of a band path, and the
quantities `h.extract()` pulls out of a Hamiltonian. If you do not remember a name, each
accepted set can be listed

```python
from pyqula import operatorlist, meanfield, extract
from pyqula.sctk import pairing
from pyqula.kpointstk import labels

print(operatorlist.get_operator_names()) # every name h.get_operator accepts
print(meanfield.get_guess_names())    # every mf= mean-field initialization
print(pairing.get_pairing_modes())    # every h.add_pairing(mode=...) symmetry
print(labels.get_label_names())       # every high-symmetry kpoint label
print(extract.get_extractable_names())# every h.extract(...) quantity
```

A name outside those lists raises a `ValueError` quoting both the offending name and the
accepted ones, rather than failing somewhere downstream.

## Location operator

To understand the spatial location of the states we can use the spatial operators, that
denote where wavefunctions are located in real space
$$
R_\alpha = \sum_{r,s} r_\alpha c^\dagger_{r,s} c_{r,s}
$$

with $r_\alpha$ the component of the position of site $r$

```python
from pyqula import geometry
g = geometry.triangular_lattice() # get the geometry
h = g.get_hamiltonian()  # get the Hamiltonian
x = h.get_operator("xposition") # x component
y = h.get_operator("yposition") # y component
z = h.get_operator("zposition") # z component
```

The expectation value of `y` in each state of a ribbon says where across the width that
state lives, so `h.get_bands(operator=y)` separates the states at the two edges from the bulk
ones by their position, the continuous version of the bulk-edge operator that follows.


## Bulk-edge operator

In order to know if a state is located at the edge or in the bulk of the system
you can use the bulk-edge location operators. The edge operator takes value 1 for
sites on the edge, and 0 for sites in the bulk.


$$
\hat E  = \sum_{r\in \text{Edge},s} c^\dagger_{r,s} c_{r,s}
$$

The bulk
operator takes value 1 for sites in the bulk, and 0 for sites on the edge.

$$
\hat B  = \sum_{r\in \text{Bulk},s} c^\dagger_{r,s} c_{r,s}
$$


```python
from pyqula import geometry
g = geometry.honeycomb_zigzag_ribbon() # get the geometry
h = g.get_hamiltonian()  # get the Hamiltonian
b = h.get_operator("bulk") # bulk operator
e = h.get_operator("edge") # edge operator
```

The two are defined by position rather than by connectivity: for a ribbon the sites in the
outer tenth of the width on each side count as edge, for a finite island the sites at the
outer part of the radius, and for a two-dimensional multilayer the outermost layers along
$z$. In the zigzag ribbon above, `h.get_bands(operator=e)` gives an edge weight close to 1
for the flat band at zero energy, the zigzag edge states, and a small one for the dispersive
bulk bands, so the operator is a quick way to tell in-gap edge states from bulk states in a
band structure.

## Valley operator

For honeycomb-like systems, including aligned and twisted multilayers, we can extract the
valley degree of freedom of each state with the valley operator, `h.get_operator("valley")`.
This operator takes the form

$$
V = i \sum_{\langle \langle ij \rangle\rangle,s} \nu_{ij} \sigma_{ij}  c^\dagger_{r_i,s} c_{r_j,s}
$$

where $\nu = \pm 1$ and $\sigma = \pm 1$ for clockwise/anticlockwise, sublattice A/B. This is
the so-called anti-Haldane hopping, and takes opposite values in opposite valleys.
It can be obtained for honeycomb systems as

```python
from pyqula import geometry
g = geometry.honeycomb_zigzag_ribbon() # get the geometry
h = g.get_hamiltonian()  # get the Hamiltonian
vall = h.get_operator("valley") # valley operator
```

Its expectation value has opposite signs for the states around $K$ and around $K'$, so the
band structure computed with `h.get_bands(operator=vall)` is colored by valley, with the two
valleys at opposite ends of the color scale. This is most useful in a supercell, where the
folded bands of the two valleys land on top of each other and the valley operator is what
tells them apart. See `examples/1d/valley_operator/main.py` for the valley-resolved bands of a
zigzag ribbon, `examples/2d/valley_bands/main.py` for those of a honeycomb supercell, and the
executed notebook
`jupyter-notebooks/functionalities/single_particle_hamiltonians/04_state_resolved_bands.ipynb`.

## In-plane valley operators

The operator above is the out-of-plane valley pseudospin $\tau_z$. We will now see how to
obtain the two remaining components, $\tau_x$ and $\tau_y$, with `h.get_operator("valley_x")`
and `h.get_operator("valley_y")`, giving access to the full valley vector
$(\tau_x,\tau_y,\tau_z)$, the valley-space analogue of $(S_x,S_y,S_z)$ for real spin. They
are built from a chiral Kekule coupling, symmetrized over the three inequivalent Kekule
registries so that the result is exactly $C_3$-covariant about every atom and not only about
special high-symmetry points, rather than from the second-neighbor coupling behind $\tau_z$

```python
from pyqula import geometry
g = geometry.honeycomb_lattice().supercell(3) # Kekule-commensurate cell
h = g.get_hamiltonian(has_spin=False)  # get the Hamiltonian
taux = h.get_operator("valley_x") # tau_x
tauy = h.get_operator("valley_y") # tau_y
```

Both require a honeycomb-like geometry with a sublattice index. For a periodic Hamiltonian
they additionally require a Kekule-commensurate cell (a 3x3, or any other multiple-of-3,
supercell of the primitive honeycomb cell) to be well defined, which is the reason for the
`supercell(3)` above; a finite (0d) flake needs no such commensurability.

A single vacancy in an otherwise pristine honeycomb flake is an atomically sharp,
intervalley-scattering defect, and induces a vortex in the in-plane valley pseudospin around
it, a nice way to see $\tau_x$ and $\tau_y$ in action

```python
from pyqula import islands
from pyqula import spectrum
g = islands.get_geometry(name="honeycomb",n=8,nedges=6)
gv = g.remove(g.get_central()[0]) # flake with a single vacancy
hv = gv.get_hamiltonian(has_spin=False)
dvx = spectrum.real_space_vev(hv,operator=hv.get_operator("valley_x"))
dvy = spectrum.real_space_vev(hv,operator=hv.get_operator("valley_y"))
```

What comes back is the expectation value of each component on every site of the flake, and
plotting $(\tau_x,\tau_y)$ as an arrow on each site shows the in-plane valley pseudospin
winding around the vacancy.

The valley vector can also be coupled to, not only measured: `h.add_valley_exchange(v)`,
with `v=(vx,vy,vz)`, adds a valley-space exchange term $\vec{v}\cdot(\tau_x,\tau_y,\tau_z)$
to the Hamiltonian, the valley-pseudospin analogue of `h.add_exchange()` for real spin

```python
from pyqula import geometry
g = geometry.honeycomb_lattice().supercell(3) # Kekule-commensurate cell
h = g.get_hamiltonian(has_spin=False)
h.add_valley_exchange([0.1,0.05,0.2]) # (vx,vy,vz)
```

See `examples/0d/valley_vortex_vacancy/main.py` and `examples/2d/valley_vortex/main.py` for
runnable versions of the vacancy-vortex example.

## Nambu operators

In the presence of superconductivity, you can project onto the electron or hole component of
the Nambu spinor using the electron-hole operators. The Hamiltonian must already be in the
Nambu (BdG) basis, meaning that some pairing has been added or `h.setup_nambu_spinor()` has
been called; otherwise there is no hole sector to project onto, and the hole projector raises
an error saying so

```python
from pyqula import geometry
g = geometry.triangular_lattice() # get the geometry
h = g.get_hamiltonian()  # get the Hamiltonian
h.add_swave(0.2) # add pairing, doubling the basis into Nambu space
electron = h.get_operator("electron") # electron component
hole = h.get_operator("hole") # hole component
(k,e,c) = h.get_bands(operator=electron) # electron weight of each state
```

The electron weight is what separates the physical copy of the band structure from its
spurious one: far from the Fermi energy a state is either purely electron, weight 1, or purely
hole, weight 0, and only close to the superconducting gap do the two mix, with a weight of
one half exactly at the gap edge. This is the same weight that turns the full BdG density of
states into the electron spectral function of the superconductivity chapter.

## Berry curvature operator

The Berry curvature operator is a first example of an operator that is intrinsically momentum
dependent. The Berry curvature operator is defined as

$$
 O |\Psi_k\rangle = \Omega(k,\epsilon_k) |\Psi_k \rangle
$$

where $\Omega(k,\omega)$ is the Berry curvature evaluated at the momentum $k$ and energy
$\omega$ of the eigenstate $|\Psi\rangle$. In particular, this operator allows to directly
see the contribution to the Berry curvature of different states in the band structure. It is
obtained as `h.get_operator("berry")`, it is defined for two-dimensional Hamiltonians, and it
is evaluated through the Green's function at the energy of each state, with a broadening
`delta` that defaults to `0.1`; passing it to `h.get_bands(operator="berry")` colors the bands
of a Chern insulator by where their Berry curvature sits, and the Brillouin-zone integral of
that curvature is the Chern number of the topological insulators chapter.

## Inverse participation ratio operator

So far we have considered operators that are linear, namely that fulfill the condition

$$
A (|\Psi_1 \rangle + |\Psi_2 \rangle) = 
A |\Psi_1 \rangle + A|\Psi_2 \rangle 
$$

There is however one operator that it is interesting to consider that does not fulfill such
condition. The operator is the so-called inverse participation ratio, which we define as

$$
 O |\Psi\rangle = \sum_i | \langle i | \Psi \rangle |^4 |\Psi \rangle
$$

In particular, the previous operator allows to identify states that are highly localized in a
few lattice sites, becoming useful to highlight impurity states and localized modes.

```python
from pyqula import geometry
g = geometry.honeycomb_zigzag_ribbon() # get the geometry
h = g.get_hamiltonian()  # get the Hamiltonian
h.add_onsite(0.3) # add a sublattice imbalance
ipr = h.get_operator("IPR") # IPR operator
```

Its expectation value is $\sum_i |\langle i|\Psi\rangle|^4$, which is 1 for a state sitting
on a single site and $1/N$ for a state spread evenly over $N$ sites, so a large value flags a
localized state: in the zigzag ribbon above, `h.get_bands(operator=ipr)` gives the edge
states at zero energy an inverse participation ratio more than ten times larger than the one
of the bulk bands.
