# Superconductivity
Up to now we have focused on Hamiltonians that contain only normal terms,
namely that the full Hamiltonian can be written as

$$
H_0 = \sum_{ijss'} t_{ijss'}c^\dagger_{i,s} c_{j,s'}
$$

where $ij$ run over the sites and $ss'$ over the spins, so that every term
destroys one electron and creates one, and the number of electrons is
conserved.

In the presence of superconductivity, an anomalous term appears in the
Hamiltonian taking the form


$$
H_{SC} = \sum_{ijss'} \Delta_{ij}^{ss'}c_{i,s} c_{j,s'} + h.c.
$$

meaning that pairs of electrons are destroyed and created, so that the number
of electrons is no longer conserved; the pairing amplitude $\Delta_{ij}^{ss'}$
is what is called the superconducting order parameter, and it is what we will
put in by hand in this chapter and obtain self-consistently from an attractive
interaction in the mean-field chapter.

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


You can think of the Nambu spinor as the two electron operators of site $n$
stacked on top of a copy of their conjugates, so that a term creating two
electrons becomes a hopping between the upper and the lower half of the
spinor, and the problem is again a single-particle one. We then
rewrite the Hamiltonian as
$$
H = \Psi^\dagger \mathcal H \Psi
$$

where $\mathcal H$ is the Nambu Hamiltonian, also called the Bogoliubov-de
Gennes (BdG) Hamiltonian, a matrix twice as large as the normal one. In this
new basis, the Hamiltonian can be written in a diagonal form as

$$
H = \sum_\alpha \epsilon_\alpha \Psi^\dagger_\alpha \Psi_\alpha
$$

where $\epsilon_\alpha$ are the Nambu eigenvalues. Because the lower half of
the spinor is the conjugate of the upper one, the eigenvalues come in pairs
$\pm\epsilon_\alpha$, an electron copy and a hole copy of every state, and the
gap of the spectrum around zero energy is twice the smallest
$|\epsilon_\alpha|$. In pyqula the doubled basis is set up by
`h.setup_nambu_spinor()`, and the methods that add a pairing, `h.add_swave()`
and `h.add_pairing()`, set it up on their own when it is not there yet; every
observable of the guide, the band structure, the density of states, the LDOS,
is then computed in the Nambu basis. A superconducting mean-field calculation
needs the basis before it starts, which is why `h.setup_nambu_spinor()` is
called explicitly in the mean-field chapter.


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

meaning that an electron with spin up and an electron with spin down on the
same site are paired, with the same amplitude $\Delta_0$ on every site; this
is the pairing of a conventional superconductor.

In the following, we address the electronic structure of a triangular
lattice with s-wave superconductivity, whose Hamiltonian takes the form

$$H = H_0 + H_{SC} $$

with 

$$H_0 = \sum_{\langle ij\rangle} c^\dagger_i c_{j} + h.c.$$

The previous Hamiltonian can be computed for $\Delta_0=0.2$ with
`h.add_swave()` as

```python
from pyqula import geometry
g = geometry.triangular_lattice() # geometry of the 2D model
h = g.get_hamiltonian() # generate the Hamiltonian
h.add_swave(0.2) # add s-wave superconductivity
(k,e) = h.get_bands() # compute band structure
```
Note that due to the BdG nature of the Hamiltonian, the band structure shows
both the electron and the hole states. What you see is a gap of $2\Delta_0$
opening at zero energy along the whole Fermi surface of the triangular
lattice, regardless of where the Fermi surface is: an electron at the Fermi
energy pays $\Delta_0$ to enter the superconductor whatever its momentum,
which is the hallmark of s-wave pairing. Coloring the bands with
`operator="electron"` in `h.get_bands()` tells the electron copy from the hole
copy, which is how the spectral function of the electrons alone is read off a
BdG band structure.

See `examples/2d/comparison_swave/main.py` for a runnable version (the same
gap put in with `h.add_swave()` and with `h.add_pairing(mode="swave")`, bands
colored by their electron weight), and the notebooks
`jupyter-notebooks/functionalities/single_particle_hamiltonians/01_spinless_spinful_nambu_basis.ipynb`
and
`jupyter-notebooks/functionalities/single_particle_hamiltonians/03_magnetism_soc_superconductivity.ipynb`
for the Nambu basis and for superconductivity combined with magnetism and
spin-orbit coupling. A pairing that arises from an attractive interaction
rather than being put in by hand is the subject of the "Superconducting
mean-field" section.

## Spin-triplet d-vector and non-unitary superconductivity

In the s-wave case above the two paired electrons form a spin singlet. Let us
now address the other possibility, a pair in a spin triplet, and see how
pyqula characterizes it with `h.get_dvector_non_unitarity()`. For a
spin-triplet superconductor the pairing is a symmetric $2\times2$ matrix in
spin space, and the way it is parametrized is by a complex three-component
d-vector $\vec d$ as

$$
\Delta = i (\vec d \cdot \vec \sigma) \sigma_y =
\begin{pmatrix}
-d_x + i d_y & d_z \\
d_z & d_x + i d_y \\
\end{pmatrix}
$$

so that $d_z$ is the opposite-spin component $\Delta_{\uparrow\downarrow}$
and $d_x,d_y$ encode the equal-spin ones,
$\Delta_{\uparrow\uparrow}$ and $\Delta_{\downarrow\downarrow}$. Only
$\Delta_{\uparrow\downarrow}$ can also host a spin-singlet contribution, which
is antisymmetric under exchanging the two spins and even under exchanging the
two sites, the opposite of the triplet; pyqula projects it out before
reading off $d_z$, so that a state with both singlet and triplet order still
gives the triplet d-vector alone.

What separates two classes of triplet states is the product

$$
\Delta \Delta^\dagger = |\vec d|^2 \mathbb{1} + \vec q \cdot \vec \sigma,
\qquad
\vec q = i (\vec d \times \vec d^*)
$$

When $\vec q = 0$ the state is **unitary**: $\Delta \Delta^\dagger$ is
proportional to the identity, the two quasiparticle branches are degenerate,
and the Cooper pairs carry no net spin. When $\vec q \neq 0$ the state is
**non-unitary**: the gap is different for the two spin branches, and the pairs
carry a spin moment $\vec q$, which is real by construction (the cross product
of a vector with its own conjugate is purely imaginary). A $\vec d$ whose
components share a common phase, a real vector times a global phase, is
always unitary; non-unitarity requires a relative phase between components, as
in $\vec d \propto (1, i, 0)$, which is a pure $\Delta_{\uparrow\uparrow}$
pairing with $\vec q$ along $+z$. Non-unitary states therefore require broken
time-reversal symmetry, and they appear naturally in ferromagnetic
superconductors, where $\vec q$ is parallel to the magnetization because the
pairs form in the majority band.

`h.get_dvector_non_unitarity()` returns $\vec q$ per site, as an array of
shape (number of sites, 3), with $\vec d$ evaluated on a uniform k-mesh,
$\vec q$ averaged over that mesh and the pairing partners of each site summed
over. As an illustration, let us put in by hand a p-wave pairing with
$\vec d \propto (1,i,0)$, which is non-unitary with the pair spin along $+z$

```python
from pyqula import geometry
g = geometry.chain() # geometry of the 1D model
h = g.get_hamiltonian() # generate the Hamiltonian
h.setup_nambu_spinor() # initialize the Nambu basis
h.add_pairing(delta=0.3,mode="pwave",d=[1.,1j,0.]) # up-up pairing
q = h.get_dvector_non_unitarity(nk=20) # -> [[0., 0., 0.36]]
```

The array has one row, the single site of the chain, and only its $z$
component is nonzero, as the relative phase of $i$ between $d_x$ and $d_y$
dictates; the same p-wave pairing with a real d-vector, `d=[1.,0.,0.]`, is
unitary and returns zero. Non-unitary states also arise on their own from a
self-consistent calculation: adding attractive first-neighbor interactions on
top of a large Zeeman splitting, as in the example of the "Long range
interactions" section, gives rise to a spin-triplet order whose $\vec q$
follows the magnetization

```python
from pyqula import geometry
g = geometry.triangular_lattice() # generate the geometry
h = g.get_hamiltonian() # create Hamiltonian of the system
h.add_exchange([0.,0.,3.]) # add exchange field
h.setup_nambu_spinor() # initialize the Nambu basis
# perform a superconducting non-collinear mean-field calculation
h = h.get_mean_field_hamiltonian(V1=-1.5,filling=0.3,mf="random",nk=30)
q = h.get_dvector_non_unitarity(nk=30) # antiparallel to +z, as is h.get_magnetization()
```

For an inhomogeneous system, `h.write_non_unitarity()` writes the same
quantity as a real-space map in `NON_UNITARITY_MAP.OUT`, with columns
$(x,y,z,q_x,q_y,q_z)$. If all that is needed is whether the state is
non-unitary at all, and not in which direction,
`h.get_average_dvector(non_unitarity=True)` returns the k-averaged squared
components of $\vec q$, a magnitude that carries no sign; the same method
with its default `non_unitarity=False` returns
$(|d_x|^2,|d_y|^2,|d_z|^2)$, and is what
`superconductivity.identify_superconductivity` uses to label a state.

See `examples/1d/dvector_map/main.py` (a self-consistent triplet state along a
spin spiral, with its non-unitarity map) and `examples/2d/SC_symmetry/main.py`
(a mixed singlet and triplet pairing split into its two parts) for runnable
versions, and
`jupyter-notebooks/functionalities/interacting_mean_field_hamiltonians/07_order_parameter_identification.ipynb`
for how a self-consistent order parameter is identified. See
`jupyter-notebooks/functionalities/interacting_mean_field_hamiltonians/16_dvector_nonunitarity.ipynb`
for an executed notebook on the non-unitarity.

## Superfluid weight and BKT temperature

A finite pairing amplitude does not by itself make a superconductor: what
carries the supercurrent is the rigidity of the state against winding the
phase of the order parameter, and we will now see how to compute it with
`h.get_superfluid_weight()`. This rigidity is the *superfluid weight*, also
called the superfluid stiffness,

$$
D_s^{ab} = \frac{1}{V}\frac{\partial^2 \Omega}{\partial Q_a \partial Q_b}
$$

the second derivative of the grand potential with respect to the twist
wavevector $\mathbf Q$ of the phase, at fixed $|\Delta|$. What is computed is
the general multiband BdG expression of Liang and coworkers, Phys. Rev. B
**95**, 024515.

The superfluid weight can also be split into two parts. The *conventional*
part comes from the band velocities, and the *quantum-geometric* part from
interband matrix elements. For an isolated flat band the second reduces to
the integral of the quantum metric (Peotta and Törmä, Nat. Commun. **6**,
8944): a flat band has no velocity, so all of its stiffness is geometric,
which is why a flat band can superconduct at all. The split is offered on top
of the full result rather than used as the definition, and it is refused with
a `ValueError` when its assumptions (uniform onsite pairing and time-reversal
symmetry) do not hold. A band touching that lands on the k-mesh, such as the
Dirac point of the kagome and honeycomb lattices when `nk` is a multiple of 3,
does not stop it: the split is an integral over the Brillouin zone, and that
single k-point is sampled from the points around it. Let us compute both on
a square lattice away from half filling

```python
from pyqula import geometry
g = geometry.square_lattice() # geometry of the 2D model
h = g.get_hamiltonian() # generate the Hamiltonian
h.add_onsite(-0.6) # move away from half filling
h.add_swave(0.3) # BdG Hamiltonian with an s-wave gap

D = h.get_superfluid_weight(nk=20) # Cartesian (dim,dim) tensor
out = h.get_superfluid_weight(nk=20,decompose=True)
print(out["total"],out["conventional"],out["geometric"])
print(h.get_bkt_temperature(nk=20)) # Nelson-Kosterlitz criterion
```

`D` is the Cartesian tensor, isotropic on the square lattice, and the
decomposition returns a dictionary whose `"geometric"` entry vanishes here:
a single-band model has no other band to have matrix elements with, so all of
its stiffness is conventional, and the geometric part only appears in a
multiband model, a flat band above all.

The twist uses the full bond vector $\mathbf R + \mathbf r_j - \mathbf r_i$
rather than the lattice vector $\mathbf R$ alone, and this matters. With the
lattice vector alone $D_s$ comes out anisotropic on the honeycomb lattice,
which C3 symmetry forbids, and it changes when the same crystal is described
with a supercell. The full bond vector gives an isotropic, supercell-invariant
answer. The two agree only for a cell with one orbital. `gauge="lattice"`
selects the other convention, the one used by Peotta and Törmä, and
`h.get_quantum_metric()` takes the same keyword with the same default (see
"Quantum geometric tensor (multiorbital/multiband)"): at fixed $|\Delta|$ the superfluid weight genuinely
depends on where the orbitals sit, see Huhtinen, Herzog-Arbeitman, Chew,
Bernevig and Törmä, Phys. Rev. B **106**, 014518.

In two dimensions `h.get_bkt_temperature()` solves the Nelson-Kosterlitz
criterion $T_{\rm BKT} = (\pi/8) D_s(T_{\rm BKT})$ self-consistently by
bisection, at frozen $|\Delta|$ (there is no $\Delta(T)$ feedback, so it is
an upper estimate); for the square lattice above it comes out at $0.15$ in
units of the hopping. Setting `mode="finite_difference"` differentiates the
grand potential numerically instead: much slower, but assumption-free. One
caveat to know in advance: at $T=0$ with a *gapless normal state* and zero or
tiny pairing, the paramagnetic and diamagnetic contributions cancel through a
$-\partial f/\partial E$ that collapses to a delta function, which a finite
k-mesh cannot resolve, so $D_s$ comes out at the normal state's Drude weight
rather than zero; use a temperature the mesh resolves when checking that a
marginal state has no stiffness. See `examples/2d/superfluid_weight/main.py`
for a runnable version, and
`jupyter-notebooks/functionalities/interacting_mean_field_hamiltonians/15_superfluid_weight.ipynb`
for an executed notebook.
