# Topological insulators

Let us now look at the quantities that tell whether a gapped Hamiltonian is a topological
insulator, a system with a gap in the bulk and states at the edges that no perturbation can
remove as long as the gap stays open. In this chapter we first compute the invariants that
count those edge states, the Chern number for a system with broken time-reversal symmetry
and the $Z_2$ invariant for one that preserves it, then the quantum geometric tensor the Berry
curvature is part of, and afterwards we resolve the Berry curvature in energy and in real
space, where it becomes a local marker that needs no Brillouin zone at all. The chapter closes
with the surface spectral function, where the edge states the invariant predicts are seen
directly.

## Topological invariants

A topological invariant is an integer computed from the occupied bands over the whole
Brillouin zone, and an integer cannot change continuously: the only way to change it is to
close the gap, and this is the reason the edge states it counts are protected.

### Chern number

We will now see how to compute the Chern number, the invariant of a two-dimensional insulator
with broken time-reversal symmetry, with `h.get_chern()`. It is defined as

$$
C = \frac{1}{2\pi} \int \Omega (\mathbf k) d^2 \mathbf k
$$

where $\Omega$ is the Berry curvature, meaning that the Chern number is the total Berry flux
of the occupied bands through the Brillouin zone in units of $2\pi$. It is an integer, it is
the Hall conductivity of the insulator in units of $e^2/h$, and it counts the chiral states
that cross the gap at each edge, which is how one measures it. Let us take the Haldane model,
a honeycomb lattice with a complex second-neighbor hopping that breaks time-reversal symmetry
and opens a topological gap at the Dirac points. The Chern number can be computed with the
following code

```python
from pyqula import geometry
from pyqula import kdos
g = geometry.honeycomb_lattice() # create honeycomb lattice
h = g.get_hamiltonian() # create hamiltonian of the system
h.add_haldane(0.05) # Add Haldane coupling
C = h.get_chern() # Chern number
```

The result is $C=2$: the Hamiltonian is spinful by default and the Haldane term acts the same
way on both spins, so each spin contributes one, and the same model built with
`has_spin=False` gives $C=1$. The integral is a sum of the Berry curvature over a k-mesh of
`nk` points per direction, ten by default, which is enough for a gap of this size. The closer
the system is to a gap closing, the more sharply peaked the Berry curvature becomes and the
finer the mesh has to be before the number settles at an integer, so a Chern number that comes
out far from an integer is asking for a larger `nk`, or for the adaptive integration of the
next section. An `operator=` restricts the curvature to a subspace, a spin or a valley, which
is how a spin Chern number or a valley Chern number is obtained (see the Berry curvature
operator section), and `h.get_berry_curvature()` returns the curvature itself as a map over
the Brillouin zone, the same one that this integral sums.

See `examples/2d/chern/main.py` (a Chern insulator built from Rashba coupling and an exchange
field), `examples/2d/haldane_model/main.py` and `examples/2d/valley_chern/main.py` for
runnable versions, and
`jupyter-notebooks/functionalities/topological_characterization/01_chern_z2.ipynb` and
`02_operator_resolved_chern.ipynb` for executed notebooks with the curvature maps.

#### Tensor-cross-interpolation (qtci) integration

By default the Brillouin-zone integral above is a plain sum over a uniform `nk` x `nk` mesh,
so its cost grows as `nk^2`. There is a second way of evaluating it, `integration="qtci"`,
which treats the integrand as a function on a binary-refined grid and learns it adaptively by
quantics tensor cross interpolation, sampling where the function varies, and then integrates
the resulting representation with Gauss-Kronrod quadrature. The number of evaluations then
grows roughly logarithmically rather than quadratically in the effective resolution, which
`nk` sets

```python
from pyqula import geometry
g = geometry.honeycomb_lattice()
h = g.get_hamiltonian()
h.add_haldane(0.05)
C = h.get_chern(integration="qtci",nk=20) # tensor-cross-interpolated BZ integral
```

What matters is when to reach for it. The adaptive integration is accurate for a smooth Berry
curvature, a model with a large gap, where it reaches the integer to between $10^{-5}$ and
$10^{-9}$ from far fewer evaluations than a dense mesh. For a sharply peaked curvature, a
small gap, it is the wrong tool: the Haldane model above, whose gap is small, returns
$C=2.015$ with `nk=20`, and the error grows rather than shrinks as `nk` is raised, since
refining the quadrature grid leaves fewer of its nodes near the peak. The default mesh sum has
no such problem, because it counts the vortices of the Berry phase on the plaquettes of the
mesh rather than integrating a field, so it is exactly quantized at every `nk`; it is the
choice whenever the gap is small. The same backend can compute the density matrix in a
mean-field calculation, replacing the k-mesh sum there:

```python
from pyqula import geometry
g = geometry.honeycomb_lattice()
h = g.get_hamiltonian()
hscf,e = h.get_mean_field_hamiltonian(U=2.0,filling=0.5,mf="antiferro",
        nk=8,maxerror=1e-4,return_total_energy=True,integration="qtci")
```

In a metal the density matrix has a step at the Fermi surface, and a quadrature rule integrates a step slowly. The Fermi level is located on the same quadrature nodes the density matrix is integrated on, and the total energy is summed on them too, so that the charge and the energy belong to one another. On a lattice with high symmetry the nodes come in groups related by the symmetry, so at a temperature close to zero the level at the Fermi energy can be a group of several states holding a sizable fraction of an electron, and that level is then occupied only in part, so that the charge is exactly the requested filling. What remains is the quadrature error of a step, which the k-mesh has to converge away as usual: for an attractive $U=-2$ on the square lattice at `filling=0.3`, the pairing gap comes out 16% too small at `nk=8` and within 3% at `nk=16`. For a gapped system none of this applies, and qtci and the k-mesh sum agree. Note also that in two dimensions the density-matrix path ends up evaluating every node of the quadrature grid, so it is not cheaper than the k-mesh sum there.

The tensor cross interpolation itself is a pure-Python port of
`TensorCrossInterpolation.jl`, bundled with pyqula, so nothing extra needs installing. Note
that the density-matrix path supports 2D Hamiltonians only. Runnable versions are in
`examples/2d/chern_qtci/main.py` and `examples/2d/mean_field_qtci/main.py`.

#### Winding of the hybrid Wannier centers

We will now see a third way of obtaining the Chern number, `integration="wannier"`, which
needs no Berry curvature at all and is the construction the $Z_2$ invariant of the next section
is built on. Let us freeze the momentum $k_2$ along the second reciprocal direction, so that
the system becomes a one-dimensional chain along the first, and take the Wilson loop of its
occupied bands, the product of the overlaps of the Bloch states around that closed loop. The
phases of its eigenvalues are the positions of the hybrid Wannier centers, the charge centers
of states localized along the first direction and extended along the second, and as $k_2$
goes once around the Brillouin zone their sum moves by $C$ lattice constants, meaning that the
Chern number is the charge pumped across the unit cell in one cycle

```python
from pyqula import geometry
g = geometry.honeycomb_lattice() # create honeycomb lattice
h = g.get_hamiltonian() # create hamiltonian of the system
h.add_haldane(0.05) # Add Haldane coupling
C = h.get_chern(integration="wannier") # winding of the hybrid Wannier centers
```

The result is $C=2$, the same as the mesh sum. The winding is an integer by construction, and
it is the right integer as long as the sum of the centers moves by less than half a lattice
constant between two consecutive values of $k_2$, of which there are `nt`, 100 by default, with
`nk` k-points along each Wilson loop, 30 by default. The centers move fastest where the gap is
smallest, so it is the gap that sets `nt`: with a Haldane coupling of $0.02$ in the snippet
above, `nt=20` returns 0 and `nt=50` already returns 2. The $Z_2$ invariant of the next section
is the same count taken over half of the Brillouin zone, from $k_2=0$ to $k_2=1/2$, the number
of times the centers cross a fixed line on the way, whose parity says whether the Kramers
partners switch. The flow of the centers itself, the object one plots to see the pumping, is
returned by `topology.z2_wannier_centers(h,full=True)`, with the momentum $k_2$ in its first
row and one row per occupied band after it, each center as the phase $2\pi x$ of its fractional
coordinate $x$ along the loop. `loop=` and `pump=` choose the two reciprocal directions, and
`gauge="atomic"` places each orbital at its position in the cell rather than at its origin, as
pyqula's Bloch Hamiltonian does, which leaves every winding as it is but moves the centers to
where the charge actually sits, so it is the choice whenever the positions themselves matter,
a polarization for instance. See `examples/2d/wannier_centers/main.py` for a runnable version
that plots the flow.


### Z2 invariant

When time-reversal symmetry is preserved the Chern number vanishes, and the invariant that
remains is the $Z_2$ index, with only two values: trivial, or a quantum spin Hall insulator
with a helical pair of states at each edge. We will now see how to compute it for the
Kane-Mele model, a honeycomb lattice with intrinsic spin-orbit coupling, with
`topology.z2_invariant()`. It can be computed with the following code

```python
from pyqula import geometry
from pyqula import kdos
g = geometry.honeycomb_lattice() # create honeycomb lattice
h = g.get_hamiltonian() # create hamiltonian of the system
h.add_soc(0.05) # Add spin-orbit coupling
from pyqula import topology
z2 = topology.z2_invariant(h) # Z2 invariant
```

The result is a parity, $-1$ for a topological insulator and $+1$ for a trivial one: the
spin-orbit coupling of the snippet gives $-1$, and a honeycomb lattice gapped by a sublattice
imbalance instead gives $+1$. What the invariant counts is the parity of the number of
helical pairs at each edge, since an odd number of Kramers pairs crossing the gap cannot be
removed by any perturbation that preserves time-reversal symmetry, while an even number can.
It is evaluated from the pumping of the Wannier centers across half of the Brillouin zone, on
a mesh of `nk` momenta and `nt` pumping steps (60 each by default). `h.add_soc()` is the same
term as `h.add_kane_mele()`. Note that `h.get_topological_invariant()` chooses between the two
invariants for you, returning the $Z_2$ invariant when the Hamiltonian is time-reversal
symmetric and the Chern number otherwise, and the Berry phase in one dimension.

The same index classifies a superconductor that preserves time-reversal symmetry, what is
called a class DIII superconductor, where it gives the parity of the number of helical pairs of
Majorana states at each edge, in-gap states that a tunneling probe at the edge would see. For a
Bogoliubov-de Gennes Hamiltonian the invariant is computed from its negative-energy states, and
time-reversal symmetry is checked on the electron and on the hole part of the Nambu spinor, up
to a global phase of the pairing, since a uniform phase of the order parameter is only a gauge
choice. Let us take the simplest case, a helical p-wave superconductor on the square lattice,
where spin up pairs as $p_x-ip_y$ and spin down as $p_x+ip_y$, so that the two spins carry
opposite Chern numbers and time reversal exchanges them

```python
import numpy as np
from pyqula import geometry
g = geometry.square_lattice() # square lattice, band between -4 and 4
h = g.get_hamiltonian() # spinful first-neighbor Hamiltonian
h.add_onsite(2.0) # chemical potential at -2, above the band bottom
def helical(r1,r2): # helical p-wave pairing, d-vector along the bond
    dr = r1-r2 # bond vector
    if abs(dr.dot(dr)-1.)>1e-4: return np.zeros((2,2)) # first neighbors only
    return 1j*np.array([[0.,dr[0]-1j*dr[1]],[dr[0]+1j*dr[1],0.]]) # i d.sigma
h.add_pairing(mode=helical,delta=0.3) # add the pairing, h is now a BdG Hamiltonian
z2 = h.get_topological_invariant() # Z2 invariant of the superconductor
```

The result is $-1$ as long as the chemical potential is inside the band, and $+1$ once it is
moved below the band bottom, `h.add_onsite(5.0)` instead of $2$, where there is no Fermi surface
left to pair. Note that at half filling, `h.add_onsite(0.0)`, the Fermi surface goes through the
$X$ points, where this pairing vanishes, so the gap closes there and the invariant is not
defined. As a reference, the Kane-Mele insulator of the snippet above gives $+1$ once a
small s-wave pairing is added to it, `h.add_swave(0.01)`, although its normal state gives
$-1$: the Nambu doubling carries its helical edge twice, and a pairing gaps the pair, so a
quantum spin Hall insulator is a trivial superconductor. Note that the Chern number of each
spin cannot be read with `h.get_chern(operator="sz")`: for a Nambu Hamiltonian the name `"sz"` is
the spin operator $\sigma_z\tau_0$, which this pairing does not conserve, and the curvature
weighted by it integrates to zero, while the operator that labels the two decoupled sectors is
$\sigma_z\tau_z$, `Operator(np.diag([1.,-1.,-1.,1.]))`, with which the weighted Chern number is
$C_\uparrow-C_\downarrow=-2$. See `examples/2d/z2_helical_superconductor/main.py` for the
invariant across the band.

### Spin Chern number and mirror Chern number

The $Z_2$ invariant needs time reversal, and it is only a parity. We will now see two invariants
that count the helical pairs themselves and survive when time reversal is broken, with
`h.get_spin_chern()` and `h.get_mirror_chern()`. Both are built the same way: at every
k-point the occupied states are split into two halves by the sign of an operator restricted to
them, and each half has a Chern number of its own. The simplest case is the spin. When $s_z$ is
conserved the halves are the two spin sectors, and the spin Chern number

$$
C_s = \frac{C_+ - C_-}{2}
$$

is half the difference of their Chern numbers, meaning that it counts the helical pairs at each
edge, each spin going around the sample in the opposite direction. When $s_z$ is not conserved,
by a Rashba coupling for instance, the halves are the eigenstates of $P s_z P$, the spin
projected on the occupied states, and $C_s$ stays an integer as long as that projection keeps a
gap around zero, which is checked; with time reversal its parity is the $Z_2$ invariant

```python
from pyqula import geometry
g = geometry.honeycomb_lattice() # create honeycomb lattice
h = g.get_hamiltonian() # create hamiltonian of the system
h.add_kane_mele(0.1) # Add spin-orbit coupling
h.add_rashba(0.1) # Add Rashba coupling, so that the spin is not conserved
Cs = h.get_spin_chern() # spin Chern number
```

The result is $C_s=1$. Note that `topology.spin_chern` computes something else, the integral of
the $s_z$-weighted Berry curvature, which is $C_\uparrow-C_\downarrow=2C_s$ while $s_z$ is
conserved and is no longer an integer here, 2.097 with `nk=24`.

The mirror Chern number splits the occupied states by the eigenvalues $\pm i$ of the mirror
$z\to -z$ instead, which maps every momentum of a two-dimensional system onto itself, and it is
$C_M=(C_{+i}-C_{-i})/2$. It is the invariant of a topological crystalline insulator, where it is
the mirror, not time reversal, that protects the edge states. The mirror is found and verified
from the geometry and the Hamiltonian, and the call raises when there is none, as with the
Rashba coupling above, which breaks it

```python
from pyqula import geometry
g = geometry.honeycomb_lattice() # create honeycomb lattice
h = g.get_hamiltonian() # create hamiltonian of the system
h.add_kane_mele(0.1) # Add spin-orbit coupling
h.add_exchange([0.,0.,0.3]) # out-of-plane exchange: breaks time reversal, keeps the mirror
CM = h.get_mirror_chern() # mirror Chern number
```

The result is $C_M=-1$: for a single layer the mirror is $-i\sigma_z$, so $C_M=-C_s$, and it
stays quantized under an out-of-plane exchange field, which keeps the mirror while breaking time
reversal, so that no $Z_2$ invariant is defined at all. For two coupled layers, the AA bilayer
of `specialhamiltonian.multilayer_graphene(l=[0,0])` with the same spin-orbit coupling, one
finds $C_s=2$ with a trivial $Z_2$ invariant, two copies of the quantum spin Hall state, and
$C_M=0$, since the two mirror sectors carry opposite Chern numbers. Both calls take `nk`, the
k-points per direction of the mesh, 40 by default. See
`examples/2d/spin_mirror_chern/main.py` for the two invariants as the Rashba coupling and the
exchange field grow, and
`jupyter-notebooks/functionalities/topological_characterization/08_spin_mirror_chern.ipynb`
for an executed notebook.

### Quadrupole moment and the nested Wilson loop

We will now see how to compute the invariant of a quadrupole insulator, a two-dimensional
insulator with no polarization and no Chern number whose topology shows up as a zero mode at
each corner of a flake, a peak in the local density of states that STM would see at the
corners and nowhere else, with `h.get_quadrupole_moment()`. In the section on the winding of the hybrid Wannier centers we saw
that the Wilson loop along $\vec b_1$ has, at each momentum $k_2$ along the other direction,
the hybrid Wannier centers $\nu_1(k_2)$ as the phases of its eigenvalues, meaning the positions
along $\vec a_1$ of the occupied states once they are localized in that direction. When these
Wannier bands split into two groups separated by a gap, one with $0<\nu_1<1/2$ and one with
$-1/2<\nu_1<0$, each group is a sector of electrons sitting on one side of the unit cell along
$\vec a_1$, and we can ask where the electrons of one sector sit along $\vec a_2$. The answer is
a second Wilson loop, along $\vec b_2$, over the states of that sector alone, what is called the
nested Wilson loop, and the phase of its determinant averaged over $k_1$ is the polarization of
the sector along $\vec a_2$, written $p_2^{\nu_1^\pm}$. With a mirror symmetry along each
direction every one of these polarizations is $0$ or $1/2$, and the quadrupole moment is

$$
q_{xy} = p_2^{\nu_1^-}p_1^{\nu_2^-} + p_2^{\nu_1^+}p_1^{\nu_2^+} \mod 1
$$

meaning that it is $1/2$ when the Wannier sectors are displaced by half a lattice constant along
both directions, and $0$ otherwise. Let us take the Benalcazar-Bernevig-Hughes model, a square
lattice with a flux $\pi$ through each plaquette and four sites per unit cell, with a hopping
$1-\delta$ inside the cell and $1+\delta$ between cells

```python
from pyqula import specialhamiltonian
h = specialhamiltonian.square_2OTI(delta=0.3) # quadrupole insulator, four sites per cell with pi flux
q = h.get_quadrupole_moment() # quadrupole moment from the nested Wilson loops
p = h.get_wannier_sector_polarization(loop=0,sector="+") # polarization along a2 of one Wannier sector
```

The result is $q_{xy}=1/2$, and each of the four sector polarizations is $1/2$. With the stronger
hopping inside the cell, `delta=-0.3`, all of them are $0$, a trivial insulator, which is the same
dimerization argument as in the Su-Schrieffer-Heeger chain applied along both directions at
once, and an open flake of the first model has a zero mode at each corner while the second has
none. With the two directions dimerized differently only one pair of sector polarizations is
$1/2$, the Wannier sectors are displaced along one direction only, and $q_{xy}=0$. What makes
this invariant different from the ones above is that the phase can change without any closing
of the bulk gap, through a closing of the gap between the Wannier bands: for `delta=-1.0` the
plaquettes decouple, every Wannier center sits at $0$, and the two sectors cannot be told apart.
The call raises in that case, when a Wannier band comes closer than `tol` to $0$ or to $1/2$, and
`topology.wannier_gap(h)` gives that distance. The polarizations are computed with every orbital
at the origin of its unit cell, which is the convention that defines the quadrupole moment;
`gauge="atomic"` places each orbital at its position instead, and then the sector polarizations
are positions in the unit cell. The quantization of $q_{xy}$ relies on the two mirrors, and
without them the formula above returns a number with no meaning. Both calls take `nk`, the
k-points per direction, 40 by default. See `examples/2d/quadrupole/main.py` for the invariant
across the transition, `examples/0d/cornermodes/main.py` for the corner modes of the flake, and
`jupyter-notebooks/functionalities/topological_characterization/09_quadrupole_moment.ipynb`
for an executed notebook.

### Strong and weak Z2 indices in three dimensions

We will now see what the $Z_2$ invariant becomes in three dimensions, where
`h.get_topological_invariant()` returns four of them. Let us take a plane of the Brillouin zone
at a fixed momentum $k_i$ along one reciprocal lattice vector. At $k_i=0$ and at $k_i=1/2$ time
reversal maps the plane onto itself, so each of these six planes is a two-dimensional
time-reversal-symmetric insulator with a $Z_2$ invariant of its own, computed as in the previous
section. Four combinations of them are independent, written $\nu_0;(\nu_1\nu_2\nu_3)$: the weak
index $\nu_i$ is the invariant of the plane $k_i=1/2$, and the strong index $\nu_0$ is the
product of the invariants of the planes $k_i=0$ and $k_i=1/2$, which comes out the same for the
three directions. You can think of a weak topological insulator, $\nu_0=0$ with some $\nu_i=1$,
as quantum spin Hall layers stacked along $\nu_1\vec b_1+\nu_2\vec b_2+\nu_3\vec b_3$, with
Dirac cones only on the surfaces that cut the layers, while a strong one, $\nu_0=1$, has an odd
number of Dirac cones on every surface, which is what ARPES sees. Let us take the Fu-Kane-Mele
model, the diamond lattice with spin-orbit coupling between second neighbors, where the gap
opens once one of the four first-neighbor bonds differs from the other three

```python
from pyqula import geometry
g = geometry.diamond_lattice_minimal() # diamond lattice, two sites per cell
h = g.get_hamiltonian() # first-neighbor hopping
h.add_kane_mele(0.05) # spin-orbit coupling between second neighbors
h.intra = 1.3*h.intra # make the bond inside the unit cell stronger
z2 = h.get_topological_invariant() # strong and weak Z2 indices
```

The result is `(1, (1, 1, 1))`, a strong topological insulator. With that bond weaker than the
other three instead, `h.intra = 0.7*h.intra`, it is `(0, (1, 1, 1))`, a weak topological
insulator of layers stacked along the bond, and once the bond dominates, `h.intra =
3.1*h.intra`, it is `(0, (0, 0, 0))`, a band insulator of decoupled dimers. Note that the
indices are 0 or 1 here, rather than the $\pm 1$ of the two-dimensional invariant, since the
weak ones are the components of a vector, and that they are given in the reciprocal lattice
vectors of the geometry, so the weak vector depends on the choice of unit cell while $\nu_0$
does not. Each of the six planes costs one two-dimensional invariant, with `nk` and `nt` as
there, 60 by default, and the call raises if the strong index comes out different from the
three directions, which means that the flow of some plane was not resolved or that the gap
closes.

When time reversal is broken, the same call returns the Chern vector $(C_1,C_2,C_3)$ instead,
the Chern numbers of the planes at fixed $k_1$, $k_2$ and $k_3$, each counted with the Wilson
loop along the next reciprocal lattice vector in cyclic order. For Chern insulator layers
stacked along $\vec a_3$ it is $(0,0,C)$, with $C$ the Chern number of one layer, the Hall
conductance of each layer in units of $e^2/h$. Both are also available directly, as
`topology.z2_invariant_3d(h)` and `topology.chern_vector(h)`. See
`examples/3d/z2_diamond/main.py` for the three phases of the diamond model, and
`jupyter-notebooks/functionalities/topological_characterization/10_z2_three_dimensions.ipynb`
for an executed notebook.

### Winding number of a chiral chain

In one dimension `h.get_topological_invariant()` returns the Berry (Zak) phase, which is $0$ or
$\pi$ and so only tells whether the number of zero modes at each end is even or odd. We will now
see how to count them, with `h.get_winding_number()`. Let us take the simplest case, the
Su-Schrieffer-Heeger chain, two sites per cell with an intracell hopping $t_1$ and an
intercell hopping $t_2$. Its Hamiltonian only connects the two sublattices, so it anticommutes
with the sublattice operator $S$, what is called a chiral symmetry, and in the eigenbasis of $S$
the Bloch Hamiltonian is off-diagonal, with a block $q(k)$ between the two sublattices. The
winding number is

$$
W = \frac{1}{2\pi i}\oint dk\, \partial_k \log\det q(k)
$$

meaning that it counts how many times the phase of $\det q(k)$ goes around as $k$ crosses the
Brillouin zone, and it is the number of zero modes at each end of an open chain, all on one
sublattice, which is what STM sees as a zero-bias peak at the end

```python
from pyqula import geometry
g = geometry.bichain() # two-site chain with a sublattice
def fun(r1,r2): # intracell hopping 0.5, intercell hopping 1
    dr = r1-r2
    if abs(dr.dot(dr)-1.)>1e-4: return 0. # first neighbors only
    return 0.5 if abs((r1[0]+r2[0])/2.)<1e-4 else 1.0 # the intracell bond is centered at x=0
h = g.get_hamiltonian(fun=fun,has_spin=False) # SSH chain
W = h.get_winding_number() # winding number, with the sublattice operator
```

The result is $W=-1$, one zero mode per end, and it becomes $0$ once the intracell hopping is
the stronger one; the sign follows which sublattice $S$ counts as positive. By default the
chiral operator is the sublattice operator of a normal Hamiltonian, and for a Nambu
Hamiltonian it is $\sigma_y\tau_y$ on every site, the chiral symmetry of a Bogoliubov-de Gennes
Hamiltonian that is real in real space, so the same call counts the Majorana zero modes of a
Kitaev chain

```python
from pyqula import geometry
h = geometry.chain().get_hamiltonian() # spinful chain
h.add_onsite(20.) # shift both spin bands up
h.add_zeeman([0.,0.,20.]) # and bring one of them back to the Fermi energy
h.add_pairing(mode="pwave",delta=0.3,d=[1.,0.,0.]) # p-wave pairing: a Kitaev chain
W = h.get_winding_number() # sigma_y tau_y is the chiral operator by default
```

which gives $W=1$ for a chemical potential inside the band and $0$ outside it. The difference
with the Zak phase appears with two such chains coupled by a weak rung hopping: they carry two
Majorana zero modes at each end, $W=2$, and the Zak phase reads $0$, the same as a trivial
chain. Any other chiral operator can be passed as `chiral=`, a name, a matrix or an `Operator`,
and the call raises if it does not anticommute with the Hamiltonian, or if the gap closes. See
`examples/1d/winding_kitaev_ladder/main.py` for the single chain and the ladder across the band,
and `jupyter-notebooks/functionalities/topological_characterization/11_one_dimensional_invariants.ipynb`
for an executed notebook.

### Z2 invariant of a helical superconducting wire

A wire that preserves time-reversal symmetry, a class DIII superconductor, has instead a
Kramers pair of Majorana zero modes at each end, a zero-bias peak that a magnetic field splits,
and neither of the invariants above sees it: the Zak phase adds up the two Kramers partners and
is always trivial, and the winding with $\sigma_y\tau_y$, when the wire has that symmetry at
all, counts them with opposite signs and gives $0$. We
will now see how to compute the $Z_2$ invariant that does, which `h.get_topological_invariant()`
returns for a time-reversal-symmetric Nambu chain. It is the Kramers polarization

$$
\nu = \det U \, \frac{\mathrm{Pf}\,\theta(0)}{\mathrm{Pf}\,\theta(\pi)}
$$

where $U$ is the product of the overlaps of the negative-energy states from $k=0$ to $k=\pi$
and $\theta$ is the time reversal restricted to those states at the two time-reversal-invariant
momenta, meaning that $\nu$ compares how the Kramers partners are paired at the two ends of half
of the Brillouin zone, and it is $-1$ when they switch. Let us take a helical p-wave wire, where
the two spins pair with opposite chirality, with Rashba spin-orbit coupling so that spin is not
conserved, and with a competing s-wave pairing

```python
from pyqula import geometry
h = geometry.chain().get_hamiltonian() # spinful chain
h.add_onsite(0.5) # shift the band, keeping the Fermi energy inside it
h.add_rashba(0.2) # Rashba spin-orbit coupling, so that spin is not conserved
h.add_pairing(mode="pwave",delta=0.5j,d=[1.,0.,0.]) # helical p-wave pairing
h.add_swave(0.2) # a competing s-wave pairing
nu = h.get_topological_invariant() # Z2 invariant of the wire
```

The result is $-1$, a topological wire, and with the s-wave pairing above the p-wave one,
`h.add_swave(1.3)`, it is $+1$, the gap closing in between, near $0.9$. When spin is conserved,
without the Rashba coupling, $\nu$ is the Zak parity of one spin block, a Kitaev chain. Any basis
of the negative-energy states is allowed at every $k$-point, so there is no gauge to fix, and
the only resolution is the number `nk` of k-points from $0$ to $\pi$, 200 by default; a value of
$\nu$ far from $\pm 1$ raises, asking for more. It is also available as
`topology.z2_invariant_1d(h)`. See `examples/1d/z2_helical_wire/main.py` for the invariant as the
s-wave pairing grows, and the `11_one_dimensional_invariants.ipynb` notebook above.

See `examples/2d/z2_kane_mele/main.py` and `examples/2d/z2_transition/main.py` (the invariant
across a transition driven by a sublattice imbalance) for runnable versions, and the
`01_chern_z2.ipynb` notebook above.
