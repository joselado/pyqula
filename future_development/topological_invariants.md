# Topological invariants: what is built, what is missing, and in which order

This roadmap settles which topological invariants pyqula computes, which ones it does not,
and in which order the missing ones are worth building. It comes out of a survey of arXiv
done on 25 September 2026, in four families (one dimension and superconductors, two
dimensions with crystalline and higher-order topology, real space and disorder, and three
dimensions with the non-Hermitian, driven, interacting and bosonic cases), where every
reference below was checked against its arXiv record, and where the test values marked as
measured were computed with pyqula itself in scratch prototypes, not recalled. The next
decision point is at the end: which of the first packages to start with.

## What the package computes today

In two dimensions the Chern number is the backbone. `h.get_chern()` sums the Berry
curvature over a k-mesh by counting the Berry-phase vortices on its plaquettes, so it is
exactly quantized at every `nk`; `integration="qtci"` integrates the same curvature
adaptively, and is accurate only for a smooth curvature; `integration="wannier"` counts the
winding of the hybrid Wannier centers; the quantum geometric tensor gives it a fourth time,
from the integral of its antisymmetric part. An `operator=` restricts the curvature to a
spin, a valley or a sublattice, and `topology.get_chern_operator_sector` computes the Chern
number of the states with a definite eigenvalue of an operator that commutes with $H$,
which is how the mirror Chern example is built. The $Z_2$ invariant comes from the flow of
the hybrid Wannier centers over half of the Brillouin zone, for insulators and, since this
survey, for superconductors that preserve time reversal (class DIII). In one dimension
there is the Berry (Zak) phase, which is also what the Kitaev-chain test uses as its
Majorana criterion. In real space there is the Bianco-Resta local Chern marker, on a finite
island only, and the Berry curvature resolved in energy and in space through Green's
functions, and the Li-Haldane entanglement spectrum of a real-space region.

`h.get_topological_invariant()` dispatches on the dimension: in one dimension the $Z_2$ of
a time-reversal-symmetric superconductor and the Zak phase otherwise, the $Z_2$ invariant in two when the Hamiltonian is time-reversal symmetric and the
Chern number otherwise, and in three the strong and weak $Z_2$ indices or the Chern vector,
since the first-tier package below was built.

What is missing, in one sentence per family: in one dimension, since the second package
below, only the disordered versions (the scattering and the real-space invariants); in two dimensions there is no polarization, only the Chern number modulo $n$ and the parity $Z_2$ from symmetry
eigenvalues, nothing for fragile topology, and higher-order topology has the quadrupole
moment only, since the nested Wilson loop was built; in real space there is no
$Z_2$ marker and nothing that scales past dense diagonalization, since the Bott index was built; in three
dimensions there is nothing beyond the $Z_2$ indices and the Chern vector; and the non-Hermitian, Floquet and interacting
cases have no invariant of their own. The collective modes are the exception: the Chern number of a magnon band and of an exciton band of the mean field, and of the magnon of the moments of a Kondo lattice, is built since 13 November 2026 (`h.get_magnon_chern()`, `h.get_exciton_chern()` and `get_moment_magnon_chern()`, see the section on collective modes), while a spin-wave Hamiltonian with terms that do not conserve the magnon number is still missing.

## Closed while the survey ran

Three things the survey found were repaired on the spot, each with a regression test that
fails on the unmodified source.

- Time reversal of a BdG Hamiltonian: `has_time_reversal_symmetry()` returned False for every Nambu Hamiltonian, so a class DIII superconductor got a Chern number (zero) instead of its $Z_2$. It now applies $T=\tau_0\, i\sigma_y K$, the right operator in the basis $(c_\uparrow, c_\downarrow, c^\dagger_\downarrow, -c^\dagger_\uparrow)$, after gauging away a global phase of the pairing. Checked against the Chern numbers of the two conserved spin sectors of a helical p-wave superconductor (`tests/topology/test_z2_superconductor.py`)
- Chern number from the Wannier-center winding: the branch was dead code marked as wrong, and `z2_wannier_centers` dropped `full=True`, so the full flow was never returned. It is now `topology.wannier_winding(h,full=True)`, reached as `h.get_chern(integration="wannier")`, and it agrees with the plaquette sum on seven models (`tests/topology/test_wannier_winding.py`)
- The $Z_2$ count itself: it counted crossings of the largest gap between the centers, which with several occupied bands changed with the resolution (a 3x3 supercell of the Kane-Mele model gave $-1$ at `nk=nt=60` and $+1$ at 100 and 150). It is now the parity of the signed number of crossings of a fixed line over half of the zone, the same function as the Chern number with `full=False`, and the half flow now ends on the time-reversal-invariant momentum $k_2=1/2$ it used to stop short of. On 44 models the two counts agree except at the supercells, where the new one is right, and at one point that sits exactly on a gap closing

The count works without following the individual centers: the sum of the centers moves by
a small step between two consecutive $k_2$, the sum of the centers measured from the fixed
line in $[0,2\pi)$ moves by the same step except for a jump of $\mp 2\pi$ at each crossing,
so the number of crossings is the difference of the two, and only the two ends of the path
enter. The line goes in the largest gap of the centers at the two ends.

## Built after the survey: the general Wilson loop and the 3D invariants

The first package the maintainer picked, on the same day, was the general Wilson loop with
the three-dimensional $Z_2$, and it is built. `topology.wannier_centers` takes the direction
of the loop and of the pumping (`loop=`, `pump=`), the momentum of the remaining direction in
3D (`kfix=`), so that the flow is the one of any plane of the zone, and `gauge="atomic"`,
which places each orbital at its position through the factor $e^{-2\pi i k\cdot x_j}$ on each
component of the occupied states. The loop then has to close on the periodic image
$e^{-2\pi i x}u(0)$ of the first states, not on the states themselves: parallel transport
follows the atomic phases along the loop, and closing on $u(0)$ gave the lattice-gauge centers
back unchanged, which was the first attempt. The centers are now returned as positions, the
phase $+2\pi x$ of a center at the fractional coordinate $x$ along the loop; before, the code
returned minus that, which is invisible in a winding and in the $Z_2$ but not in a plotted
flow, so the sign of that output changed. `topology.wannier_winding` counts downwards minus
upwards, so its Chern number keeps the orientation of `h.get_chern()`, and exchanging `loop`
and `pump` reverses it.

`topology.z2_invariant_3d` runs the half-zone count on the six planes $k_i=0,1/2$ and raises
if the strong index comes out different from the three directions; `topology.chern_vector`
runs the full-zone count on the three planes at `kfix`, with $C_i$ taken with the loop along
$b_{i+1}$ and the pumping along $b_{i+2}$. `h.get_topological_invariant()` returns the first
for a time-reversal-symmetric 3D Hamiltonian and the second otherwise. The acceptance test
(`tests/topology/test_z2_3d.py`) makes each of the four first-neighbor bonds of the
Fu-Kane-Mele diamond weaker (0.7) and stronger (1.3) in turn, which gives weak vectors along
(100), (010), (001) and (111) of the primitive reciprocal basis, and compares every case with
the Fu-Kane parities computed from the inversion operator of `compile_symmetry` about the bond
midpoint; the eight agree, and a bond of $3.1t$ gives $0;(000)$ in both. The Chern vector is
tested on stacked Haldane layers in the $(a_1,a_2)$ and in the $(a_2,a_3)$ plane. What the
package leaves for later is the polarization itself, the sum of the atomic-gauge centers,
now one line on top of the flow.

## Built after the survey: the one-dimensional invariants

The second package, picked next, is built as well. `h.get_winding_number()`
(`topology.winding_number`) diagonalizes the chiral operator once and winds $\det q(k)$,
with $q=P_+^\dagger H P_-$, around the zone; it takes `chiral=` as a name, a matrix or an
`Operator`, defaults to the sublattice operator of a normal Hamiltonian and to
$\sigma_y\tau_y$ on every site of a Nambu one (exact, to $6\cdot 10^{-17}$, on pyqula's
Kitaev chain), and raises if the operator does not anticommute with $H(k)$ or if the gap
closes. `topology.z2_invariant_1d` is the Budich-Ardonne formula with each link of the Kato
propagator replaced by its unitary part, so that $\nu$ is of unit modulus at any `nk` and a
value far from $\pm 1$ can be refused; the Pfaffian is a port of pfapack's Parlett-Reid
routine (`topologytk/pfaffian.py`), checked against $\mathrm{Pf}^2=\det$ and
$\mathrm{Pf}(B^TAB)=\det B\,\mathrm{Pf}A$. `h.get_topological_invariant()` returns it for a
time-reversal-symmetric Nambu chain, where it used to return the Zak phase, which is always
trivial there.

The tests (`tests/topology/test_one_dimensional_invariants.py`) check the parity of the
winding against the Zak phase on SSH and Kitaev chains, the ladder of two Kitaev chains with
$W=2$ and a trivial Zak phase, the $Z_2$ of the helical wire against the Zak parity of a
Kitaev chain at the same chemical potential (spin is conserved there, so this is exact), its
independence of the phase of the pairing, and, with Rashba coupling, the switch to trivial
as an s-wave pairing grows, across a gap closing measured at $\Delta_s=0.97$ without Rashba
coupling and $0.89$ with $\lambda_R=0.2$. The survey's recommendation to take the sign of an
unconverged $\nu$ was not followed: with unitary links there is no need to.

## Built after the survey: the spin and mirror Chern numbers

The third package is built too, on one general routine with two thin wrappers
(`topologytk/topologicalsector.py`). `get_chern_operator_sign_sector` keeps, at every k-point,
the occupied states on which $POP$ is positive or negative, through the existing
`filter_state`, which diagonalizes the operator inside the occupied states, with an
acceptance by sign rather than by eigenvalue, which is what `occ_states_sector_generator`
could not do; `split_chern` returns $(C_+-C_-)/2$ after checking, with `operator_gap`, that
$POP$ keeps a gap around zero and that the number of occupied states does not change.
`h.get_spin_chern()` is that with $s_z$, and `h.get_mirror_chern()` with the mirror
$z\to -z$ about the middle plane of the geometry, built and verified by `compile_symmetry`
and passed as the Hermitian $iM$ (spinful, $M^2=-1$) or $M$ (spinless), with the sign turned
into the standard $(C_{+i}-C_{-i})/2$. For a single layer $M=-i\sigma_z$, so $C_M=-C_s$.
`topology.spin_chern`, the $s_z$-weighted curvature, is left as it is, and it is $2C_s$, not
$C_s$, where $s_z$ is conserved.

The tests (`tests/topology/test_spin_mirror_chern.py`) check $2C_s$ against the weighted
integral where $s_z$ is conserved, $(-1)^{C_s}$ against the $Z_2$ invariant with Rashba
coupling (where the weighted integral drifts, to 2.095 at $\lambda_R=0.1$), both invariants
under an out-of-plane exchange field that breaks time reversal but keeps the mirror, and an AA
bilayer of Kane-Mele layers, which gives $C_s=2$ with a trivial $Z_2$ and $C_M=0$. The
`examples/2d/mirror_chern` script, whose layer operator is a mirror only at `ti=0.0`, is left
as the sector-Chern example it is; an AB bilayer has no $M_z$ at all, and
`h.get_mirror_chern()` says so.

## Built after the survey: the nested Wilson loop

The fourth package, picked after the three above, is the nested Wilson loop of Benalcazar,
Bernevig and Hughes (arXiv:1611.07987, with the algorithm of section VI of arXiv:1708.04230,
read from the paper itself), in `topologytk/nestedwilson.py`, with
`topology.wannier_sector_polarization`, `topology.wannier_gap`, `topology.quadrupole_moment`
and the delegators `h.get_wannier_sector_polarization()` and `h.get_quadrupole_moment()`. It
is a module of its own rather than an extension of `wannier_centers`, because it needs the
eigenvectors of the Wilson loop at every base point, not only its eigenvalues. The occupied
states are computed once on an `nk` x `nk` grid, as columns; the links are
$F_k=\langle u_{k+dk}|u_k\rangle$, replaced by the unitary part of their polar
decomposition, so that the loop $F_{N-1}\cdots F_0$ has eigenvalues $e^{2\pi i\nu}$ with
$\nu$ a position (the sign convention of the paper, and the $+2\pi x$ convention that
`wannier_centers` returns); the loop at base point $i+1$ is $F_iW_iF_i^\dagger$, so each line
costs one product, and a Schur decomposition gives orthonormal eigenvectors. The sector "+"
keeps $0<\nu<1/2$ and "-" keeps $-1/2<\nu<0$, and the count of each has to be the same at every
k-point; the nested loop closes on the periodic image of the first Wannier-band state, which is
the state itself in the lattice gauge and carries $e^{-2\pi i y}$ per orbital in the atomic
gauge. The polarization at each $k_1$ is averaged after unwrapping, and a sector whose
polarization winds (a Wannier sector with a Chern number) raises. Results are returned modulo 1
in $[-1/4,3/4)$, so that neither 0 nor 1/2 sits at an edge where roundoff would flip it, and
$q_{xy}$ is eq. VI.47, $p_y^{\nu_x^-}p_x^{\nu_y^-}+p_y^{\nu_x^+}p_x^{\nu_y^+}$. The routine
raises when a Wannier center comes within `tol` (1e-4 by default) of 0 or 1/2; the Wannier gap
of `square_2OTI` is 0.018 at $\delta=-0.3$, $2.4\cdot 10^{-4}$ at $-0.9$, $5.7\cdot 10^{-5}$ at
$-0.95$ and zero at $-1$, where the plaquettes decouple with the bulk gap open.

The checks (`tests/topology/test_nested_wilson_loop.py`):

- An independent implementation: nestedWilsonLib (github.com/kuansenlin/nested_and_spin_resolved_Wilson_loop, GPL-3, on PythTB 1.7.2, installed in a scratch directory only) on a `square_2OTI(0.3)` whose mirrors are broken by onsite energies and complex intracell hoppings, so that nothing is quantized. In the lattice gauge the nested phase agreed at each of the 40 base points to $9\cdot 10^{-6}$ and its average to $3\cdot 10^{-6}$; in the atomic gauge, with the orbitals at their positions, to $3\cdot 10^{-5}$ at `nk=40` and $8\cdot 10^{-6}$ at 80. The difference falls by four when the mesh doubles, the discretization difference between its raw overlaps and the unitary links here. Its 2D routine returns the nested phase at one base point only, so the average was built by moving the base point; the four averaged values are pinned in the test
- The nested loop over all the occupied states is the ordinary Wilson loop along the other direction, which `wannier_centers` computes by another code path: equal to $10^{-10}$ in both gauges, on the mirror-broken model. This is the check that catches a conjugated eigenvector or a wrong order of the product
- The Wilson-loop eigenvalues do not depend on the base point, and agree with `wannier_centers` at base 0, to $10^{-12}$
- An atomic limit in the atomic gauge, whose sector polarizations are the orbital coordinates, which fixes the sign and the origin
- `square_2OTI` with $\delta=0.3$ and 0.6 gives the four polarizations and $q_{xy}$ equal to 1/2, and $\delta=-0.3$ gives 0, at `nk=20` and 31; with the two directions dimerized differently ($\gamma_x,\gamma_y=1.25,0.25$ with $\lambda=1$) only $p_y^{\nu_x}=1/2$ and $q_{xy}=0$, the classes of figs. 25 and 29 of the paper
- The sum rule $p^{\nu^+}+p^{\nu^-}=p$, the total polarization, to $2\cdot 10^{-4}$ at `nk=40` on the mirror-broken model (a discretization error, since the Berry connection of the whole manifold traces over both sectors only in the limit)
- A flipped sign of the nested phase fails six tests, conjugated Wilson-loop eigenvectors eight, and a nested loop closed without the periodic image three

A real-space check of the corner charge was tried and not finished: on a 12 x 12 flake the four
corner modes of $\delta=0.3$ are there (at $|E|=8\cdot 10^{-4}$, 84% of their weight within three
sites of a corner, and none for $\delta=-0.3$), but a perturbation of $\pm 0.1$ on alternate
quadrants, meant to split them, gave quadrant charges of $\pm 0.86$ and $\pm 0.07$ in the two
phases rather than $\pm 1/2$ and 0, since it polarizes the bulk and the edges as well. A corner
charge wants the perturbation near the corners only, or the filling-anomaly count of
arXiv:1809.02142. What is left: the nested loop on a plane of a 3D zone (the `kfix` of
`wannier_centers`), the octupole moment (a third nesting), sectors chosen by a window rather
than by the lines 0 and 1/2, for Wannier gaps elsewhere, and the $C_4$ route of eq. VI.53,
which needs the symmetry-eigenvalue primitive of the second tier.

## Built after the survey: the symmetry eigenvalues, the Chern number modulo n, the parity Z2, the corner charge and the Z4 indicator

The sixth package, picked on 5 November 2026 as the start of the symmetry branch of the
higher-order plan below, is in `topologytk/symmetryeigenvalues.py`, with the delegators
`h.get_symmetry_eigenvalues(op,k)`, `h.get_chern_mod_n()` and `h.get_parity_z2()`, and
`topology.rotation_operation`, `topology.inversion_operation` and `topology.invariant_momenta`
to build the operation and find its momenta. The primitive compiles the operation with
`compile_symmetry`, checks that it maps $k$ onto $k+G$, writes it in the occupied states as
$U^\dagger P(k) U$ and raises when $PU\neq UA$, which is what a closing gap or an `nocc` through a
multiplet gives. In pyqula's lattice gauge $H(k+G)=H(k)$ exactly, so no position phase enters
and the eigenvalues are those of the physical operator about the given center. The invariant
momenta are found as the fixed points of $k\to M^{-T}k$ modulo 1 among the fractions 0, 1/2,
1/3, 2/3, 1/4, 3/4, 1/6 and 5/6, not from a table.

The Chern number modulo $n$ follows eqs. ChernC2, ChernC4, ChernC3 and ChernC6 of the TeX
source of arXiv:1207.5767, including the factor $(-1)^F$ per occupied band, which the paper
writes for $C_3$, $C_4$ and $C_6$ only. The counterclockwise rotation of
`rotation_operation` gives the Cartesian Chern number, and the result is multiplied by the sign of $(\mathbf a_1\times\mathbf a_2)_z$ so that it has the sign of `h.get_chern()`, which follows the order of the lattice vectors (before the seventh bug audit a left-handed pair gave $-C$ modulo $n$). The parity
$Z_2$ counts the occupied states of parity $-1$ at each time-reversal-invariant momentum (TRIM), $\delta_i=(-1)^{n_-/2}$, and
searches the inversion center among the midpoints $(r_0+r_j+R)/2$, which covers every inversion
of the crystal; the result does not depend on the center, since moving it by half a lattice
vector flips the parities at an even number of momenta of each product.

The checks (`tests/topology/test_symmetry_eigenvalues.py`):

- QWZ on the square lattice (spin-dependent hopping $t s_z$, Rashba 1, exchange $m$), C4 acting on the spin: C mod 4 and mod 2 equal `get_chern` for $m=\pm1,\pm3$ ($C=\mp1$) and $m=5$ ($C=0$)
- Haldane, spinless and spinful with one occupied band (exchange 4, onsite 4), $t_2=\pm0.1$: C6 and C3 about a hexagon center, C3 about a site, C2 about the bond center and about a hexagon center, all equal to `get_chern` modulo $n$. The spinful case at odd filling is what fixes the $(-1)^F$
- The eigenvalues satisfy $\lambda^n=1$ spinless and $-1$ spinful
- A Fermi level inside the bands raises, from a guard on the number of occupied states at the invariant momenta, which each pass the representation check on their own
- Kane-Mele with the bonds along $x$ scaled by $s$: parity $Z_2$ equal to the Wannier flow of `z2_invariant` (which returns $(-1)^\nu$) at $s=1,1.5$ (1) and $2.5,4$ (0), and to the spin Chern number in a prototype. The Dirac cones merge at $s=2$, at a TRIM, so the transition is at 2 and not at 3; at $s=2$ the routine raises
- Diamond Fu-Kane-Mele, a weak, a strong and a trivial phase: equal to `z2_invariant_3d`
- The parity $Z_2$ about three inversion centers that differ by half lattice vectors is the same
- A rotation with the opposite orientation fails 8 tests, and dropping the $(-1)^F$ fails 7

The corner charge of arXiv:1809.02142 followed the same day, as `h.get_corner_charge()`, from
eq. CornerIndices of the TeX source of the main text for $C_2$, $C_4$ and $C_6$. The index of
the paper is the electronic charge in units of $e<0$, so the routine returns $-Q/e$ modulo 1,
the total charge of electrons and ions in units of $|e|$; the stack $h_{1b}\oplus h_{2c}$ at
3/8 filling gives 3/4, the $3|e|/4$ of fig. 1(a) of the paper, which fixes the sign. The
indices for $C_2$, $C_4$ and $C_6$ use only real eigenvalues once time reversal is imposed
($[M_2^{(4)}]=[M_4^{(4)}]$), so they do not depend on the orientation of the rotation. The
$C_3$ index $\frac{e}{3}[K_2^{(3)}]$ uses a complex eigenvalue at $K$, so it depends on which
valley is called $K$, which the paper fixes only in a figure; it raises `NotImplementedError`.
The paper's own $C_3$ generators (the breathing kagome models) have a nonzero polarization, so
they cannot fix it either, and a $C_3$ model with $P=0$ and a flake oracle would be needed. The
rotation center has to be the center of the unit cell from which the flake is built (Wyckoff
1a); a vanishing polarization is checked from the $C_2$ eigenvalues and raises otherwise. The
$C_2$ index is the charge of each of the two halves of a flake, $n/2$ times the $C_n$ corner
charge modulo 1, which the tests check.

The oracle is a flake of whole unit cells, cut into $n$ pinwheel sectors of whole cells related
by the rotation, with the central cell shared, ions $n_{occ}$ per cell, and the Fermi level in
the largest spectral gap around the neutral filling, so that the corner multiplet is all filled
or all empty. With chiral symmetry and every negative-energy state filled, the density is
exactly 1/2 on every site and the sector charge is zero whatever the phase, which is why the
multiplet has to be treated as a whole. Checked: the Kekulé model $H_2^{(6)}$ of the paper
(1/2 for $t_0<1$, 0 above), the stack (3/4 and 0, and a raise when only one block is
topological, where the flake shows an edge charge that grows with its size), and
`honeycomb_lattice_C6()` with `h.add_kekule(-0.3)` (1/2) and `+0.3` (0). The paper's $C_4$
generator $H_2^{(4)}$ was tried and dropped: as written it is gapless at half filling, its
second and third bands touch at $E=0$. Returning the electronic charge instead of the total
fails the stack test. A side finding: `h.remove_sites()` failed on a Hamiltonian with a sparse
`intra`, since the numba crop received a `csc_matrix`; the crop now slices a sparse matrix
and keeps it sparse, with a test that the sparse and dense paths give the same Bloch matrix.

The $Z_4$ indicator followed, as `h.get_z4_indicator()`: $\kappa_1=\frac14\sum_K(n_K^+-n_K^-)$
mod 4 with time reversal and spin (eq. defk1 of the TeX source of arXiv:1711.11589), and
$z_4=\frac12\sum_K(n_K^+-n_K^-)$ mod 4 otherwise, with $\theta=\pi z_4/2$ from eq.
magnetoelectric of arXiv:1010.4335. The same eigenvalues give $z_4=2\kappa_1$ on a
time-reversal-symmetric insulator, so the method returns $\kappa_1$ when the Hamiltonian is
spinful with time reversal and $z_4$ otherwise, which the docstring and the guide say. The parity
constraint of arXiv:1010.4335 (eq. constraint, an even total number of occupied odd states in
an insulator) makes $z_4$ even, since $z_4=-\sum_K n_K^-$ mod 4; an odd $z_4$ raises, naming
Weyl points. A first version tested the parity of $\sum_K(n_K^+-n_K^-)$, which is always even,
and returned odd values: the notebook's map of the magnetized diamond model showed $z_4=1$ and 3
in two wedges around the transitions at $s=1$ and $s=3$, and at $s=1$, $J=0.3$ the direct gap
closes to $10^{-12}$ at $(0.524,0,0.524)$ and its images, Weyl nodes away from the TRIMs.

Checked (`tests/topology/test_symmetry_eigenvalues.py`): $\kappa_1=\nu_0$ mod 2 against
`z2_invariant_3d` on a weak, a strong and a trivial phase of the diamond model with one bond
scaled; two copies of each, the second at positions scaled by 0.8 about the bond center, which
keeps the inversion, with an onsite coupling 0.1: $\kappa_1=2$ for the doubled strong phase
with `z2_invariant_3d` $=(0,(0,0,0))$, the second-order insulator, and 0 for the doubled weak and
trivial ones; a uniform exchange 0.02 and 0.1 on the strong phase gives $z_4=2$ with a zero
Chern vector, the axion insulator; 0.3 closes the gap and raises; the Weyl case raises. With a
nonzero weak index $\kappa_1$ depends on the inversion center: moving it by $\mathbf a_i/2$ changes it by $2\nu_i$ modulo 4, which maps 1 onto 3 and 2 onto 0 as a reversed sign does. The test checks that rule against the weak indices of the Wannier flow, and the sign through $\kappa_1=-\frac12\sum_K\mathrm{sign}f_K$ about the bond center, built from the four hoppings alone. The weak phase of the guide's model has $\kappa_1=2$ about the first center found and 0 about the bond center, so $\kappa_1=2$ is a second-order insulator only with zero weak indices. Not
verified: the hinge modes themselves, on a rod, for either the second-order or the axion case.

Not built here, and the next decision: the $C_3$ corner charge, and a real-space check of the
hinge modes. A test of the
primitive by a gauge rotation inside a degenerate multiplet was not possible from outside,
since `eigh` returns $s_z$ eigenstates for Kane-Mele, which are also $C_3$ eigenstates.

## Built after the survey: the Bott and spin Bott indices

The fifth package closes the first tier: the Bott index of Loring and Hastings
(arXiv:1005.4883) and the spin Bott index of Huang and Liu (arXiv:1810.00081, Eqs. (5) to (11),
read from the TeX source), in `topologytk/bott.py`, with the delegators `h.get_bott_index()` and
`h.get_spin_bott_index()`. The sample is a two-dimensional periodic supercell taken at $k=0$,
the occupied states are those below zero energy, and $X$, $Y$ are the fractional coordinates
along a1 and a2 from `qgt._orbital_fractions`. $U$ and $V$ are written in the occupied states,
$W^\dagger e^{2\pi iX}W$ of size $N_{occ}$, which gives the same index as the full matrices with
$1-P$ added, since that block only adds eigenvalues equal to one; both are replaced by their
unitary part $ZW^\dagger$ from the singular value decomposition, as Huang and Liu do, and $B$ is
the sum of the phases of the eigenvalues of $VUV^\dagger U^\dagger$ over $2\pi$. The result is a
float, not rounded, so a non-integer says the sample is too small. The spin Bott index
diagonalizes $Ps_zP$ in the occupied states and takes the Bott index of each sign;
`split_bott(h,operator=...)` is the general function and `spin_bott_index` the thin wrapper, as
for `split_chern` and `spin_chern`.

The sign. Swapping a1 and a2 flips $B$, and it also flips `h.get_chern()`, which integrates in
the reduced coordinates of the same vectors: Haldane $t_2=0.1$ gives $C=B=+1$ on the default
honeycomb lattice and $C=B=-1$ with the two vectors swapped. The maintainer's decision of 4
November 2026 keeps both following the order of the lattice vectors, so $B$ equals `get_chern`
on every lattice, and neither is the Cartesian $\sigma_{xy}$ on a left-handed pair. PyBott (GPL-3)
takes the product in the other order, $UVU^{-1}V^{-1}$, and so returns $-B$; its `compute_uv`
also takes the period as max minus min of the positions unless `lx`, `ly` are given, one
lattice spacing short of the torus. Its code was read, not run: executing the downloaded
package was refused in this session, so the external oracle is still to be run. The
independent route in the tests is the k-space Chern number.

The tolerance on a level at zero energy is 1e-8, not the 1e-4 of `split_chern`: in a disordered
sample a localized level came within $3\cdot 10^{-5}$ of zero in one of the strongly
disordered $12\times12$ Haldane supercells of the guide figure, and the index stays defined in that mobility gap; what has to be refused is a
level exactly at zero, as the Dirac points of a clean honeycomb supercell whose size is a
multiple of three.

The checks (`tests/topology/test_bott_index.py`): $B$ equals `get_chern` on the Haldane model
for both signs of $t_2$ and on both sides of $M_c=4.5\,t_2$ (pyqula's Haldane hopping carries the factor $\sqrt3/2$ of the Kane-Mele one, so the clean gap is $9t_2$ and not $6\sqrt3 t_2$), to $10^{-8}$ at supercells 4
and 7; both flip together with a1 and a2; a rigid shift of every site leaves $B$ unchanged to
$10^{-10}$; Anderson disorder $W=1$ keeps $B=1$ for three seeds and $W=16$ gives 0; the spin
Bott index equals `get_spin_chern` on the Kane-Mele model with Rashba couplings up to
$\lambda_R=\lambda_{SO}$ and 0 above the critical imbalance, with $B=0$, at supercells 4 and 6;
and the refusals of a spinless, a one-dimensional and a gapless Hamiltonian. Measured while
building it: on a $12\times12$ Haldane supercell ($t_2=0.15$, clean gap 1.35) every sample is an
integer, the average over samples is 1 up to $W=4$ and 0 from $W=7$, and the same for the
Kane-Mele model with Rashba coupling; at $8\times8$ the average inside the window moves by up to
0.2 with six samples. The plane of sublattice mass and disorder at $10\times10$ shows the
topological Anderson insulator: for $M$ between 0.7 and 1.0, above $M_c=0.675$, the average over eight samples reaches 0.4 to 1 for $W$ between 1.5 and 5. One $12\times12$
spinless supercell takes about 0.02 s; the cost is the dense diagonalization at $k=0$, so the
reach is a few thousand orbitals.

What is left: the external PyBott run; a sparse route for large samples (the occupied
projector from a KPM or a shift-invert window, which `sparse_kpm_mean_field.md` would supply);
the Bott index of an open flake, which the edge states spoil, and of the amorphous samples of
`geometry.amorphous_lattice()`, which needs a periodic amorphous geometry first; and a check that
$U$ and $V$ are far from singular, so that a non-integer result raises instead of returning.

## The candidates at a glance

Effort is small (a function and a test on top of what exists), medium (a new building
block) or large. The tier is the recommended order, argued in the last section.

| Invariant | Class, dimension | Builds on | Test, expected value | Effort | Tier |
|---|---|---|---|---|---|
| General Wilson loop: any direction, a plane of a 3D zone, atomic gauge | any, 2D and 3D | `wannier_centers`, `qgt._orbital_fractions` | atomic-limit centers at the orbital positions | small | done |
| Strong and weak $Z_2$, $(\nu_0;\nu_1\nu_2\nu_3)$ | AII, 3D | the Wilson loop on six planes | Fu-Kane-Mele diamond, each bond weaker and stronger, against the Fu-Kane parities | small | done |
| Polarization from the atomic-gauge centers | any, 1D to 3D | the Wilson loop | SSH $P=1/2$ and 0 | small | 2 |
| Chiral winding number | AIII, BDI, 1D | the chiral operator, `get_hk_gen` | SSH $\lvert W\rvert=1$, Kitaev ladder $W=2$ where Zak reads 0 | small | done |
| $Z_2$ of a helical wire | DIII, 1D | the BdG time reversal, a Pfaffian | the Zak parity of one spin block, for every $\mu$ | small | done |
| Spin Chern number with Rashba (Prodan) | none needed, 2D | `filter_state` by sign | Kane-Mele-Rashba $C_s=1$, with $(-1)^{C_s}=Z_2$ | small | done |
| Mirror Chern with the mirror found | $M_z$, 2D | `pointgroup.compile_symmetry` | Kane-Mele $C_M=-1$, also under an out-of-plane field | small | done |
| Bott and spin Bott index | A, AII, 2D torus | a supercell at $\Gamma$ | Haldane $B=\pm 1$; Kane-Mele spin Bott 1 below $m_c$ | small | done |
| Symmetry eigenvalues at high-symmetry momenta | point groups, 2D and 3D | `pointgroup.compile_symmetry` | the parities above | small | done |
| Fu-Kane parity $Z_2$, Chern mod $n$ | inversion, $C_n$, 2D and 3D | the eigenvalues | the Wannier-flow $Z_2$, `get_chern` | small each | done |
| Corner charge, $C_2$, $C_4$, $C_6$ | $C_n$ and time reversal, spinless, 2D | the eigenvalues | Kekulé 1/2, $h_{1b}\oplus h_{2c}$ 3/4, against a flake | small | done |
| $Z_4$ indicator, $\kappa_1$ and magnetic $z_4$ | inversion, 3D | the eigenvalues | doubled strong TI $\kappa_1=2$, magnetized strong TI $z_4=2$ | small | done |
| $C_3$ corner charge | $C_3$, 2D | the eigenvalues | a $C_3$ model with $P=0$, against a flake | small | 2 |
| Local $Z_2$ and spin Chern markers, Katsura-Koma index | AII, 2D, disordered | `real_space_chern` | Kane-Mele island, bulk 1 below $m_c$ | small | 2 |
| Scattering invariant $\mathrm{sign}\det r$ | D (DIII, BDI), 1D disordered | `transporttk.smatrix` | Kitaev wire $-1$ inside, $+1$ outside (measured) | small, after a guard | 2 |
| Point-gap winding, biorthogonal Chern | non-Hermitian, 1D and 2D | `get_hk_gen`, `mesh_chern` | Hatano-Nelson $w=\mathrm{sign}(g)$ | small | 2 |
| 3D Chern vector | A, 3D | the Wilson loop on three planes | stacked Haldane layers, $(0,0,C)$ and $(C,0,0)$ | small | done |
| Weyl-point chirality | A, 3D | the sliced plaquette sum | chirality $\pm 1$ per node | small to medium | 2 |
| Topological Hamiltonian $-G^{-1}(0,k)$ | interacting A, AII | a custom generator | Haldane plus flat bath levels | small | 2 |
| Nested Wilson loop, quadrupole | $M_x,M_y$, 2D | the Wilson loop | `square_2OTI` $q_{xy}=1/2$ | medium | done |
| Euler class | $C_2T$, 2D | the Wilson loop | three-band model $\pm 2$ (measured) | medium | 3 |
| KPM stochastic Chern marker, Streda formula | A, 2D and 3D, millions of sites | `kpmtk` | Haldane at $10^5$ sites against `get_chern` | medium | 3 |
| Spectral localizer | every class, gapless too | positions, a signature | Haldane island index 1 in the bulk | medium | 3 |
| Magnon Chern number, thermal Hall | bosonic BdG, 2D | a paraunitary solver (there: `solve_pseudo_hermitian`) and a spin-wave Hamiltonian | Kitaev ferromagnet in a [111] field, $C=\pm 1$ | medium to large | 3 |
| Magnon Chern number from TDHF and the RPA | Casida and RPA kernel, 2D | `MagnonProblem` lifted to the pair basis, the null vectors and the winding of $\chi_0^{-1}-V$, the plaquette sum | honeycomb ferromagnet $\pm 1$, the same in the three routes; Lieb ferrimagnet $\pm 1$ in both metrics; Kondo spin exciton $\mathrm{sign}(t_2)$ | medium | done |
| Exciton Chern number | BSE, 2D | `BSE` amplitudes lifted to the pair basis, the plaquette sum | Frenkel kagome oracle $-1$; exciton of a flat localized hole, $C=0$ exactly | medium | done |
| Floquet $W_3$, axion angle, GBZ winding, Hopf | various | various | see below | medium to large | 3 |
| Symmetry indicators, topological quantum chemistry | space groups | irreps, EBR tables | TBG flat bands | large | deferred |

## One dimension and superconductors

The chiral winding number is the integer the Zak phase only knows modulo two, the number of
zero modes at each end on one sublattice, which STM sees as a zero-bias peak at the end of a
chain (defining work arXiv:0912.2157; for wires arXiv:1111.6592). With the chiral operator
$S$ diagonalized once, $q(k)=P_+^\dagger H(k)P_-$ and $W$ is the winding of $\det q(k)$
across the zone, so no eigenvector of $H$ enters and there is no gauge to fix. What matters
is that $S$ is an explicit argument: the natural choice for a real BdG matrix is
$\sigma_y\tau_y$ per site, which has opposite signs on the two Kramers blocks, so a helical
chain gives $W=0$, and the bulk value depends on the choice of unit cell. pyqula's
`get_hk_gen()` is in the lattice gauge, periodic in $k$, which is what the winding needs.
The survey measured $\lvert W\rvert=1$ for SSH (`geometry.bichain()`, $S$ from
`operators.get_sublattice`) with the intercell hopping stronger, 0 otherwise, $W=1$ for the
Kitaev chain at $\lvert\mu\rvert<2$, and $W=2$ for two Kitaev chains coupled by
$t_\perp=0.1$, where the Zak phase is 0 and the Pfaffian number is $+1$, which is the case
that makes it worth having.

The $Z_2$ of a helical wire counts a Kramers pair of end Majoranas, a zero-bias pair that
any field breaking time reversal splits, and it is invisible to both the Pfaffian number
and the winding with the natural $S$. The formula of arXiv:1308.1256 (built on
cond-mat/0606336) is $\nu=\det(U^K)\,\mathrm{Pf}\,\theta_o(0)/\mathrm{Pf}\,\theta_o(\pi)$,
with $U^K$ the ordered product of the overlaps of the negative-energy states from $k=0$ to
$\pi$ (the loop of `topology.berry_phase`, stopped at $k=1/2$) and $\theta_o=W^\dagger U_T
W^*$ the time reversal restricted to them at the two ends, where it is antisymmetric. It
needs the BdG time reversal of the section above, exact including the phase, and a complex
Pfaffian (pfapack, MIT, arXiv:1102.3440). On a finite mesh $\lvert\nu\rvert$ is slightly
below 1 (0.988 at 400 k-points), so the sign is what is returned. Measured:
`add_pairing(mode="pwave",delta=0.3j,d=[1,0,0])` gives $-1$ at onsite $-1$, $0.5$ and $1.5$
and $+1$ at $\pm 3$, and a p-wave of $0.5j$ with an added s-wave stays at $-1$ up to
`add_swave(0.8)` and turns $+1$ from 1.2, with the gap closing in between. The weak-pairing
Fermi-point formula of arXiv:0908.3550 is a cross-check where the pairing is small.

The Kitaev Majorana number (cond-mat/0010440, arXiv:1306.4459),
$\mathrm{sign}\,\mathrm{Pf}A(0)\,\mathrm{sign}\,\mathrm{Pf}A(\pi)$ in the Majorana basis,
gave the same answer as the Zak phase on every Kitaev chain tried, so it is a two-k-point
shortcut and a consistency check rather than new information; on a finite island,
$\mathrm{sign}\,\mathrm{Pf}A$ alone is the ground-state fermion parity, which is what a
parity switch of a Yu-Shiba-Rusinov state changes, and that is its real use.

The scattering invariant $Q=\mathrm{sign}\det r$ of the reflection matrix off a long
superconductor at $E=0$ (arXiv:1101.1749, arXiv:1106.6351, arXiv:1009.5542) is the one that
works with disorder, and it reuses the transport code as it is: the survey measured
$\det r=+1.000$ at $\mu=\pm 3$ and $-0.999$ at $\mu=0,\pm 0.5, 1, \pm 1.5$ on a normal lead
attached to a Kitaev lead. It also found a failure that has to be understood first, see the
loose ends below. $\mathrm{sign}\,\mathrm{Pf}(ir)$ for DIII and the number of negative
eigenvalues of $r$ for BDI need the lead basis where $r$ is antisymmetric or Hermitian.

The real-space winding $\nu=-\mathcal T\{Q_{-+}[X,Q_{+-}]\}$ with $Q=1-2P$
(arXiv:1311.5233, arXiv:1402.7116) counts end modes of a disordered chain, including the
topological Anderson insulator of arXiv:1802.02109; measured on SSH with 200 cells, 0.000
and $-1.000$ in the two phases and $-0.999$ with a hopping disorder as large as the
hopping. It follows the pattern of `real_space_chern`.

## Two dimensions, from the Wilson loop

The general Wilson loop is the building block for most of what follows (arXiv:1101.2011,
arXiv:1102.5600, Z2Pack arXiv:1610.08983, arXiv:1312.6940). What exists is the flow along
$k_1$ at fixed $k_2$ in a 2D zone; what is missing is the loop direction, a plane at fixed
$k_3$ of a 3D zone, and a switch to the atomic gauge. The reason for the last one is that
pyqula's $H(k)$ carries no intracell positions, so the centers come out as if every orbital
sat at the origin of the cell: windings (Chern, $Z_2$, Euler) do not care, but the
polarization, the Wannier-sector polarizations and the corner charge do. The atomic gauge
is one diagonal matrix $\mathrm{diag}(e^{-2\pi i\tau_j})$ in the closing link, which
`topologytk/qgt.py` already builds for `gauge="atomic"`. Z2Pack's movement and gap
tolerances are the recipe for refining $k_2$ adaptively.

The spin Chern number of Prodan (cond-mat/0603054, arXiv:0904.1894) splits the occupied
states by the sign of the eigenvalues of $P s_z P$, which is valid when $s_z$ is not
conserved, and gives $C_s=(C_+-C_-)/2$ together with the gap of $P s_z P$, whose closing
changes $C_s$ without any closing of the energy gap. The existing `topology.spin_chern` is
an $s_z$-weighted curvature, equal to it only without Rashba coupling. The sector machinery
is almost there: `occ_states_sector_generator` accepts only $\lvert s-e\rvert<10^{-3}$, and
under Rashba coupling the eigenvalues of $P s_z P$ sit near $\pm 0.9$, so it drops every
state; it needs to accept by sign. Oracle: BerryEasy (MIT). The single-point version
(arXiv:2301.02612) belongs with the real-space tools.

The mirror Chern number (arXiv:0804.2664, arXiv:1202.1003, arXiv:1310.1044) is the same
split by the eigenvalues $\pm i$ of a mirror $M_z$, which maps every 2D momentum onto
itself. `find_point_group` finds $M_z$ and `compile_symmetry` gives its matrix, so the only
work is to feed $iM_z$ to the sector Chern: `filter_state` reads the real part of the
operator's matrix elements, so $M_z$ itself puts both sectors at zero. The current example
gets $\mp 1$ from a hand-built layer operator, which is a mirror only because `ti=0.0`
decouples the layers (AB stacking has no $M_z$). Tests: Kane-Mele $C_M=\pm 1$, which stays
at 1 under an out-of-plane exchange field until the gap closes, where the $Z_2$ is no
longer defined; an AA bilayer with Haldane couplings of the same sign gives 0. In-plane
mirrors hold only on mirror lines, so this is an $M_z$ tool.

The nested Wilson loop (arXiv:1611.07987, algorithm in arXiv:1708.04230) takes the
eigenvectors of $W_x$ in one gapped Wannier sector, builds the Wannier-band states and runs
a second Wilson loop along $y$; for `square_2OTI`, which is exactly the
Benalcazar-Bernevig-Hughes model (four orbitals, $\pi$ flux, intracell hopping $1-\delta$
and intercell $1+\delta$, checked in the survey), `delta=0.3` should give $p=1/2$ and
$q_{xy}=1/2$ and `delta=-0.3` zero, with the corner charge checked in real space as in
`examples/0d/cornermodes`. Two caveats come with it. The model changes phase through a
closing of the Wannier gap without a closing of the bulk gap (arXiv:1908.00011), so the
routine has to report the Wannier gap; and the placement of every orbital at the origin
that BBH assume is exactly pyqula's convention, so here the lattice gauge is the right one.

The Euler class (arXiv:1808.05375, arXiv:1804.09719, arXiv:2005.02044) is the invariant
of a pair of bands with $C_2T$ symmetry, the obstruction to a two-band symmetric Wannier
model of the twisted bilayer graphene flat bands (arXiv:1807.10676, arXiv:1808.02482). In
a real gauge the Wilson loop of the pair is a rotation, and the Euler class is the winding
of its angle. The survey measured it on the real three-band model
$H(k)=2dd^T-\lvert d\rvert^2$ with $d=(\sin k_x,\sin k_y,m-\cos k_x-\cos k_y)$: $+2$ for
$m=1$ and $0.5$, $-2$ for $m=-1$ and 0 for $m=3$. For TBG the atomistic flat bands mix the
valleys, so they must first be split by the sign of $PVP$, as for the spin Chern number,
and they need the sparse `max_waves` path.

## Symmetry eigenvalues

One primitive, the eigenvalues of a point-group operation on the occupied states at the
high-symmetry momenta, gives several invariants cheaply, each of which has an exact
cross-check in an existing routine. At a high-symmetry point $k'=k+G$, so in pyqula's
convention $P(k)$ from `compile_symmetry` commutes with $H(k)$ exactly, and what is needed
is $U^\dagger P U$ diagonalized in the occupied space.

- The Fu-Kane parity formula (cond-mat/0611341): one parity per Kramers pair, multiplied over the time-reversal-invariant momenta, in 2D and in 3D
- The $Z_4$ indicator (arXiv:1010.4335, arXiv:1707.01903): half the sum of $n_+-n_-$ over the eight momenta, modulo four, which is 2 for an axion insulator when the weak Chern numbers vanish
- The Chern number modulo $n$ from $C_n$ eigenvalues (arXiv:1207.5767): the product at $\Gamma$, $M$, $X$ for $C_4$, at $\Gamma$, $K$, $K'$ for $C_3$
- The corner charge (arXiv:1809.02142, arXiv:1907.10607, arXiv:2101.04322): $Q_4=\frac{e}{4}([X_1]+2[M_1]+3[M_2])$ and the analogues for $C_2$, $C_3$, $C_6$, valid only at vanishing polarization

Two traps. `find_point_group` searches about the origin by default, and on the diamond
lattice it finds no inversion there; with the bond midpoint passed as `centers=` it
compiles inversion with parities exactly $\pm 1$. And the corner-charge formulas assume
$C_n^n=+1$, which the $\pi$ flux of `square_2OTI` breaks, so they do not apply to it; the
$C_6$ Kekulé honeycomb at half filling (filling anomaly 3, corner charge $e/2$) is the test.
The full symmetry indicators and the elementary band representations of topological quantum
chemistry (arXiv:1703.00911, arXiv:1703.02050) are deferred: `pointgroup.py` has no glides
or screws and treats every orbital as a scalar, the tables live on the Bilbao server, and
IrRep reads only DFT output, so it cannot serve as an oracle for a tight-binding model. The
entries above already cover the 2D indicators that matter.

## Three dimensions

The strong and weak $Z_2$ indices (cond-mat/0607699, cond-mat/0611341, arXiv:1102.5600)
come from the flow of the Wannier centers on the six planes $k_i=0$ and $k_i=1/2$:
$\nu_i=Z_2(k_i=1/2)$ and $\nu_0=Z_2(k_i=0)+Z_2(k_i=1/2)$ modulo two, and the sum has to be
the same for the three $i$, which is a free consistency check. This is now built, see the
section on what was built after the survey. The survey measured the Fu-Kane
parities on `diamond_lattice_minimal()` with `add_kane_mele(0.05)`: scaling the intracell
bond by 1.3 gives $-1$ at seven time-reversal-invariant momenta and $+1$ at
$(\frac12,\frac12,\frac12)$, meaning $1;(111)$, by 0.7 gives $0;(111)$, and by 3.1 gives
$0;(000)$, which is the phase diagram of cond-mat/0607699, and $Z_4=2$ in the first case
and 0 in the other two. Oracles: Z2Pack and WannierTools (GPL-3, the license of pyqula, so
they can be mirrored).

The Chern vector of a layered quantum anomalous Hall insulator is the Chern number of a
plane at fixed $k_i$, now `topology.chern_vector`; stacked Haldane layers give $(0,0,1)$.
The chirality of a Weyl point (arXiv:1007.0016,
arXiv:1105.5138) is the plaquette sum on a small sphere around it, or the jump of the slice
Chern number across it, and locating the points needs a multistart minimization of the
direct gap, where `nodes.degenerate_points` finds only one. The Berry phase on a small loop
linking a nodal line is 0 or $\pi$ and `berry_phase(h,kpath=...)` already computes it.

The axion angle $\theta$ has three levels: the $Z_4$ indicator above, nearly free; the
hybrid-Wannier formulation of arXiv:1912.11887; and the Chern-Simons integral over the zone
and a cycle parameter of arXiv:0810.2998, for which PythTB 2.0 (GPL-3) has `axion_angle`
and ships a Fu-Kane-Mele model, so it is a direct oracle. The Hopf invariant of two-band
insulators with vanishing Chern numbers (arXiv:0804.4527, arXiv:1307.7206) has an analytic
test model ($\chi=1$ for $1<\lvert h\rvert<3$, $-2$ for $\lvert h\rvert<1$, 0 above 3) and
little use in the models pyqula builds.

## Real space and disorder

The Bott index (arXiv:1005.4883, equal to the Chern number up to corrections of order
$1/L$, arXiv:1708.05912) is computed on a periodic supercell at $\Gamma$: with
$U=Pe^{2\pi iX}P+(1-P)$ and $V$ the same with $Y$, both replaced by the unitary part of
their polar decomposition, $B=\mathrm{Im}\,\mathrm{Tr}\log(VUV^\dagger U^\dagger)/2\pi$. It
is an integer by construction and needs no normalization per area, which is what makes it
the natural invariant of a disordered or amorphous Chern insulator. The spin Bott index
(arXiv:1810.00081) splits $P$ by the sign of $Ps_zP$ first, and fails where that gap closes
(for $\lambda_R/\lambda_{SO}$ near 3 in the Kane-Mele model). Periodic boundaries are
required, because on an open flake the edge states spoil $U$ and $V$, the same way the
Bianco-Resta marker traces to zero there. The single-point Chern and spin Chern numbers of
arXiv:0705.3771 and arXiv:2301.02612 use the same $\Gamma$ states. Oracles: PyBott (GPL-3),
StraWBerryPy (MIT, github.com/strawberrypy-developers; the PyPI package of that name is an
unrelated animation module).

The local $Z_2$ and spin Chern markers (arXiv:2404.04598, periodic version arXiv:2310.15783)
split $P$ into $P_\pm$ by $Ps_zP$ and take
$C_\pm(r)=4\pi\,\mathrm{Im}\langle r|P_\pm[x,P_\pm][y,P_\pm]|r\rangle$, the Bianco-Resta
marker of each half, which is a small extension of `real_space_chern` (whose `operator=`
branch is a symmetrized commutator, not this). The Katsura-Koma index (arXiv:1508.05485,
numerics in arXiv:1709.05853) counts the eigenvalues at 1 of $P-D_a^*PD_a$ restricted to a
disk around a point $a$, with $D_a=(z-a)/\lvert z-a\rvert$, and its parity is the $Z_2$; it
is the smallest correct disordered $Z_2$ found, and no public code for it was found.

The KPM local Chern marker is built, close to linear in the number of sites; the measurements, and the stochastic route of arXiv:1905.02215 with the Streda and Kitaev formulas it would give, are in "The local Chern marker, built" of [`sparse_kpm_mean_field.md`](sparse_kpm_mean_field.md).

The spectral localizer (arXiv:1502.03498; sparse numerics arXiv:2601.13598; gapless systems
arXiv:2112.08623) is the one tool that works in every tenfold class, in any dimension, with
open boundaries and without a projector:
$L=(H-E)\otimes\sigma_z+\kappa(X-x)\otimes\sigma_x+\kappa(Y-y)\otimes\sigma_y$, the index is
half its signature and its smallest singular value is the protection gap, which closes
where a state sits at $(x,E)$, what STM sees at that bias. The signature needs no
eigenvectors, so dense `eigvalsh` works up to `limits.densedimension` and a sparse
$LDL^T$ beyond; $\kappa$ of order $t/a$ sets a window, and too small a value sees the
boundary while too large a value reads trivial.

The Resta polarization $\mathrm{Im}\log\det\langle\psi_m|e^{2\pi iX/L}|\psi_n\rangle/2\pi$
(cond-mat/9709306) and the corner charge are cheap on a ring or a flake. The quadrupole
operator $e^{2\pi ixy/L^2}$ of arXiv:1812.06990 and arXiv:1812.06999 is not recommended, see
the next section.

## Non-Hermitian, driven, interacting and bosonic

The point-gap spectral winding $w(E_b)=\frac{1}{2\pi i}\oint d\log\det(H(k)-E_b)$
(arXiv:1802.07964, arXiv:1812.09133) is the invariant behind the non-Hermitian skin effect,
the direction in which every state piles up on an open chain, and it needs nothing but the
existing non-Hermitian `get_hk_gen` and `slogdet`; Hatano-Nelson gives
$w(0)=\mathrm{sign}(g)$ inside the spectral loop and 0 outside. The biorthogonal Chern
number of a line-gapped band (arXiv:1706.07435, arXiv:1804.04672) is the plaquette sum with
left and right eigenvectors, and it cross-checks the existing `mode="Green"` curvature. The
non-Bloch winding on the generalized Brillouin zone (arXiv:1803.01876, arXiv:1902.10958,
arXiv:1912.05499) needs high-degree root finding and $H(\beta)=\sum_R h_R\beta^R$ built
from the multihopping, since `get_hk_gen()` casts a complex $k$ to a real one with only a
`ComplexWarning`; it is large.

The Floquet invariants (arXiv:1010.6126, arXiv:1212.3324, with the efficient $W_3$
algorithm of arXiv:1702.04181) need a time-evolution operator $U(k,T)$ built from
`get_hk_gen`, since `keldyshtk/floquet.py` is the Sambe space of a junction and not a band
Floquet operator; the five-step model at $JT/5=\pi/2$ has $U(T)=1$ and all band Chern
numbers zero, yet $W=1$. The same $W_3$ kernel gives the 3D winding of classes DIII and
AIII (arXiv:0803.2786).

The Wang-Zhang topological Hamiltonian $h_t(k)=H_0(k)+\Sigma(k,0)$ (arXiv:1004.4229,
arXiv:1203.1028) turns any existing invariant into one of an interacting insulator, and
pyqula's embedding self-energies can supply $\Sigma$; zeros of the Green's function change
it without any gap closing and without edge states (arXiv:2301.05588), which the
documentation would have to say. An exact test is the Haldane model with one flat bath
level per site, whose topological Hamiltonian is $H_0-V^2/\epsilon_b$.

The magnon Chern number and the thermal Hall conductivity (arXiv:1204.3349,
arXiv:1106.1987) need a paraunitary diagonalization, a Cholesky factor $H=K^\dagger K$
followed by the eigenvectors of $K\sigma_3K^\dagger$, and $H$ must be positive definite, so
the Goldstone mode needs a small gap. A ferromagnet with a Dzyaloshinskii-Moriya
interaction conserves the magnon number and is the Haldane model, so `mesh_chern` already
covers it; the case that needs more is the one where terms that do not conserve the magnon
number carry the topology, the Kitaev ferromagnet in a [111] field ($C=\pm 1$ for
$h/S\geq 4$). What that case lacks is the spin-wave Hamiltonian with those terms, which
`spinwaves.py` does not build; the paraunitary solver is already there, since
`bsetk/solve.solve_pseudo_hermitian` takes the Cholesky factor of $SM$ and diagonalizes
$L^\dagger SL$ with `eigh`, which is the same construction. The Casida matrix of
`bsetk/spinflip.py` has the same paraunitary form, so a magnon Chern number from
time-dependent Hartree-Fock needed a lift of the eigenvectors and a plaquette sum, not a new
solver. That is built since 13 November 2026, `h.get_magnon_chern()`, with the exciton and
the Kondo-lattice cases, and the section on collective modes below has the details; its
Casida metric gives the same integers as the Euclidean overlap on the two models with a
de-excitation half that were measured. What is still missing here is the spin-wave
Hamiltonian with terms that do not conserve the magnon number, and the thermal Hall
conductivity. Oracles: SpinW (GPL-3), Sunny.jl (MIT), neither run yet.

## Collective modes: excitons, magnons and the Kondo lattice

This section records how the Chern number of a collective mode is computed: an exciton of
the Bethe-Salpeter equation, a magnon of time-dependent Hartree-Fock or of the RPA, and the
spin exciton of a Kondo lattice. It was written on 13 November 2026 as a plan, and the three
routes it describes were built, reviewed and compared with each other on the same day. Each
route below says what is built, what was measured and what stayed derived only; the section
on the three routes on one magnet has their comparison, and the section on what is left has
the open points. The seven design decisions that followed (the methods and their names, the
defaults of route 3, the guards, the shared engine, the gap diagnostics, the temperature and
the documentation) were taken the same evening and are built: route 1 is
`h.get_magnon_chern()` and `h.get_exciton_chern()`, route 2 is
`KondoLatticeHamiltonian.get_moment_magnon_chern()` for the moments of a Kondo lattice and a
module-level tool otherwise, and route 3 stays a module-level check. The user guide has a
section for each method (spin-response, excitons and pseudofermions pages, and a pointer in
topological-invariants), the README a functionality bullet with the notebook
`topological_characterization/15_magnon_exciton_chern.ipynb`, and `examples/` three scripts.

The question is what plays the role of the Bloch state. For an electron band it is an
eigenvector of $H(k)$ in a fixed orbital basis, and `h.get_chern()` counts the winding of
its phase on the plaquettes of a k-mesh. For a collective mode with center-of-mass momentum
$Q$ the wavefunction is either an eigenvector in a pair basis that is rebuilt at every $Q$,
or the residue of a pole of a response function. This gives three routes. The first two
use a plaquette sum; the third winds the response function over a contour of complex
frequencies, the generalization of `mode="Green"` that a response function needs.

The plaquette sum they share is `topologytk/plaquette.py`:
`chern_from_states(states, nq, metric, tol, shift)` and `berry_fluxes(...)` take a callable
from $Q$ to the states of a group of modes in a space that does not depend on $Q$, compute
the states of every corner once, and raise when the number of states changes or a link
overlap falls below `tol` (Fukui, Hatsugai and Suzuki, cond-mat/0503172). The sign is the one
of `h.get_chern()`. Since the decisions the engine publishes the mesh, `mesh(nq, shift)`, and
the fluxes of states already computed on it, `mesh_berry_fluxes(V, metric)`, so the three
routes share one definition of the mesh; it also holds `require_2d`, `check_bands` and the
record of routes 1 and 2, `ModeChern(fluxes, min_overlap, qs, energies, gap, bound)`, with
`chern` and `max_flux`, the largest flux of a plaquette in modulus, computed from the fluxes.
No threshold is put on `max_flux`: the flux is the argument of a product, so a true flux of
$1.2\pi$ comes back as $-0.8\pi$, and no fraction of $\pi$ is a sharp test (the single Kondo
mode has a largest flux of $0.645\pi$ at $n_q=6$ and the right integer). The default mesh is
`nq=12` in `mesh`, `berry_fluxes`, `chern_from_states`, `mode_chern` and `kernel_chern`,
since an unshifted mesh with $n_q$ a multiple of 6 holds $\Gamma$, M and K of a hexagonal zone,
and a touching on a point of the mesh raises: on the Kondo lattice with $t_2=0$, whose two
spin excitons touch at K and K' (1.53057), $n_q=6$ and 12 refuse the lowest band and
$n_q=10$ returns an integer with no error and a gap of 0.14 on its record (`nk=8` and 10).
The tests are `tests/topology/test_plaquette_chern.py` (11).

### Route 1: the eigenvectors of the BSE and of the magnon problem

Built in `bsetk/topology.py`: the general function `mode_chern(h, mode=...)`, with `mode`
one of `get_mode_names()`, `"exciton"` or `"magnon"` (with `full_output=True` it returns the
`ModeChern` record with the fluxes, the energies, the mesh, the smallest overlap, the gap of
the group to the other modes and its distance below the continuum), the wrappers
`exciton_chern` and `magnon_chern` behind `h.get_exciton_chern()` and `h.get_magnon_chern()`
(and refused by a spinon mean field converged with `decoupling="bond"`, as its siblings),
`mode_berry_fluxes` and `mode_states`, which replaced the four functions named by the
excitation (the old `exciton_states` clashed with `bse.exciton_states`), and the lift itself,
`lift_amplitudes`, with `casida_metric` and `get_metric_names()` for `metric=`. The tests
are `tests/bse/test_exciton_chern.py` (18) and `tests/magnon/test_magnon_chern.py` (16).

The BSE at $Q$ returns the amplitudes $X_{vc}(k)$ and $Y_{vc}(k)$ on its pair basis, and the
magnon problem returns the Casida matrix with halves of size `n1` and `n2`
(`MagnonProblem`). These amplitudes cannot be overlapped between $Q$ and $Q+dQ$ as they are,
since `PairBasis` diagonalizes $H$ at $k$ and $k+Q$ for every $Q$ and `eigh` leaves an
arbitrary phase on each band. The way we fix this is by lifting the amplitudes to the orbital
pair basis,

$$\Phi_Q(k;\alpha,\beta)=\sum_{vc}X_{vc}(k)\,u_{c,k+Q}(\alpha)\,u^*_{v,k}(\beta),\qquad
\Psi_Q(k;\alpha,\beta)=\sum_{vc}Y_{vc}(k)\,u_{v,k+Q}(\alpha)\,u^*_{c,k}(\beta),$$

meaning that both blocks are amplitudes of the same operator
$c^\dagger_{\alpha,k+Q}c_{\beta,k}$, the first between an empty state at $k+Q$ and an
occupied one at $k$, the second between an occupied state at $k+Q$ and an empty one at $k$
(up to a sign: a one-body operator enters the Casida problem as $(X,-Y)$, and the sign of a
block changes no overlap). The phase of a Bloch state cancels between the amplitude and the
two $u$, so the lifted state does not depend on the Bloch gauge, and since the Bloch Hamiltonian of pyqula is
periodic in the reduced momentum, the state at $Q+G$ is the one at $Q$ and the plaquettes
close on the torus. Which conjugation the $Y$ block takes, left open by the plan, is settled
by an exact check: a one-body operator written in the pair basis lifts to
$P_{emp}(k+Q)\,O\,P_{occ}(k)$ and $P_{occ}(k+Q)\,O\,P_{emp}(k)$ to $10^{-15}$, with and without
random phases on the Bloch states. The integer is not a test of the $Y$ lift. On the Lieb
ferrimagnet of the magnon tests, whose lowest magnon has $|Y|$ up to half of $|X|$, three
wrong lifts of $Y$ (electron and hole exchanged, conjugated, dropped) all left $C=+1$
($n_q=6$, `nk=6`); the lifted state under random phases does see the first two (overlap
0.965 and 0.949 against 1). The reason for this is that $|X|>|Y|$ on the excitation branch, so
the $X$ block alone spans a bundle isomorphic to the one of the mode (derived). The
convention of an electron at $k+Q$ and a hole at $k$ is the electron Wilson loop of
arXiv:2507.22983, which shows that the exciton Chern number is the same in the electron,
the hole and every mixed convention, while the curvature of one plaquette is not; the fluxes
returned are those of the electron convention. A split that shares the momentum,
$k+Q/2$ and $k-Q/2$, is not periodic in $Q$ on a lattice and is not used. The map $Q\to-Q$
is a rotation of the plane, so a routine that solved every point at $-Q$ would return the
same integer; what pins the momentum of a flux is the energies of the record, tested on two
models with $E(Q)\neq E(-Q)$.

The Casida metric. The Chern number is a property of the bundle the modes span, so the
Euclidean overlap and the metric $\mathrm{diag}(1,-1)$ give the same integer for a group of
positive norm (derived). This is now measured on two cases with $Y\neq 0$: the full BSE of
the Frenkel model of the section on the tests ($C=-1$, $|Y|/|X|=0.10$) and the two bound
magnons of the Lieb ferrimagnet ($+1$ and $-1$). The fluxes differ plaquette by plaquette
(by up to 0.018 and 0.073) and their sums agree. Whether the fluxes in that metric are the
Berry curvature of a thermal Hall conductivity is not settled.

What was measured with route 1:

- The shared magnet of the section on the three routes on one magnet: $C=+1$ for the lower
  magnon band, $-1$ for the upper one and 0 for the pair, at $n_q=6,8,10$ and `nk=6, 8, 10`,
  and $-1$, $+1$, 0 with `add_kane_mele(-0.15)`. The two bands are 0.27 or more apart on every
  mesh and 1.88 or more below the Stoner edge.
- The sign of these integers against a model that uses no two-particle code: at strong
  coupling one flipped spin of a saturated ferromagnet hops with the amplitude
  $(2/U)\,t_{ij}(\text{flipped})\,t^*_{ij}(\text{majority})$, the superexchange of second order.
  Its splitting of the two bands is the one of the magnon problem within 1.4 % at $U=40$, and
  its Chern numbers are those of `magnon_chern` at $U=40$, and at $U=8$ for both signs of the
  Kane-Mele coupling and both signs of the field (the second sign goes through the $S^-$
  channel). The model is derived in the test file, not taken from a paper.
- The Frenkel oracle, the nonzero exciton test of the section on the tests: $C=-1$ from the
  BSE and from the exciton tight-binding model, at `nk=4, 6` and $n_q=6,8$, and also with the
  full BSE in both metrics.
- The zero model: $C=0$ to $10^{-16}$, also on meshes of $Q$ not commensurate with the
  k-mesh; the raw amplitudes with no lift give $-5$ under random phases.
- The Lieb ferrimagnet (`tij=[1,0.3]`, Kane-Mele 0.15, Zeeman 0.3, $U=4$): $+1$ and $-1$ for
  its two bound magnons in both metrics, at $n_q=6,8,10$ (`nk=6`) and $n_q=6,8$ (`nk=8`),
  flipped by the sign of the Kane-Mele coupling.
- A honeycomb Neel state in a staggered field: its single bound magnon has $C=0$ in both
  metrics ($n_q=6$, `nk=6`).

Four conditions come with it, each now enforced or documented in the module:

- The mode has to be isolated over the whole zone, from the other modes and from the
  electron-hole continuum, the Stoner continuum for a magnon. A group that touches another
  mode at a mesh point raises. A touching point between two mesh points is not seen, and the
  integer comes back with no error: the Frenkel model with no flux, which has time reversal,
  gives $-1$ for its lowest exciton on a shifted mesh ($n_q=4$, `nk=4`, smallest gap on the
  mesh 0.0055). The `gap` and `bound` of the record are the check.
- The internal mesh `nk` and the mesh of $Q$ are independent, and the mode has to be
  converged in `nk` before the integer is read. Only `solver="dense"` is built.
- An unstable reference raises: a complex energy anywhere on the excitation branch, a
  negative energy, a negative Casida norm, or a positive branch of the full BSE with fewer
  modes than pairs. The norm of a mode of complex energy vanishes, so its sign is roundoff:
  on a Neel state converged at $U=4$ with the kernel of $U=4.5$ the unstable magnon has the
  energy $0.84i$ and the norm $+1.5\times10^{-15}$, and the full BSE returns 142 modes for
  144 pairs. As a side effect, an ungapped antiferromagnet on a mesh that holds $Q=0$ is
  refused for every band when its Goldstone energy comes out imaginary; `shift=` is the remedy.
- The Goldstone mode. The plan said that a de-excitation half makes the zero mode defective.
  This holds only for a magnet with no net moment: the zero mode of a ferrimagnet is regular,
  with a norm proportional to the net moment (the Lieb ferrimagnet with no field returns a
  state at $Q=0$). A saturated ferromagnet has no de-excitation half at all. The gap of a
  Goldstone mode from an exchange anisotropy, which `check_su2_interaction` refuses, still
  needs `check_su2=False`.

In a collinear magnet the excitation branch is one sector, $\Delta S_z=-1$, and the mirror of
the de-excitation branch is the other one, so the Chern numbers of both sectors come from
the same solves; this is not built as a separate output.

### Route 2: the null vectors of a kernel between local operators

Built in `chitk/kernelchern.py`: `kernel_modes`, the zeros of a Hermitian kernel
$K(q,\omega)$ in a window and their null vectors; `kernel_chern`, their plaquette sum;
`spin_rpa_kernel`, the site-basis spin RPA of an onsite-$U$ mean field or of a spinon mean
field converged with `decoupling="bond"`; `moment_rpa_kernel`, the moments of a Kondo
lattice with the residual interaction $U'(q)$; and `get_channel_names()`, the operators
`"raising"` ($S^+$, a magnet along $-z$), `"lowering"` ($S^-$, a magnet along $+z$) and
`"full"`; and `moment_rpa_chern`, behind `KondoLatticeHamiltonian.get_moment_magnon_chern()`.
Since the decisions `kernel_chern(kernel, window, bands=None, nq=12, shift, tol,
full_output, ...)` speaks the words of route 1: `bands` is a contiguous group of the zeros of
the window, the mesh is solved once, and `full_output=True` returns the `ModeChern` record,
whose `gap` is the smallest distance of the group to another zero of the window or to an end
of the window, and whose `bound` is the smallest distance of its top below the upper edge of
the continuum, `None` for a kernel with no continuum. A group degenerate with another zero at
a point of the mesh is refused before the count of zeros is checked, since it is the more
specific reason (on the Kondo case at $t_2=0$ the code before this change refused it through
the count). The kernels default to `T=0`, sharp occupations, a state within
`DEGENERATE_TOL` of the Fermi energy taking 1/2, and a degenerate pair at the Fermi energy
with a nonzero matrix element marking a metal; a negative `T` raises. On the shared magnet the
zeros at `T=0` and at `T=1e-3` are equal to the last digit and are the TDHF energies to
$8\times10^{-14}$ (five $Q$, `nk=6` and 8). The tests are `tests/chi/test_kernel_chern.py`
(30).

The kernel is $K=\chi_0^{-1}-V$, whose zeros are the modes and whose null vectors span a line
bundle in the space of local operators, with no lift. Three things turned out different from
the plan:

- The bubble is evaluated with no broadening on the pair set of `PairBasis`, a state at
  $k+q$ and one at $k$. So its zeros are the energies of time-dependent Hartree-Fock at the
  same $q$ to $3\times10^{-14}$, at any $q$ and not only on the mesh, and the sign of $q$ is
  tested on a sublattice imbalance where $E(q)=4.6536$ and $E(-q)=4.6301$.
- The kernel is returned as $\bar K$, whose null vector
  $\psi_i(q)=\langle0|A_i(q)^\dagger|n,q\rangle$ is the site-diagonal part of $\Phi_Q$ of
  route 1. The raw null vector of $\chi_0^{-1}-V$ is its conjugate and gives the opposite
  integer.
- $\partial K/\partial\omega$ is positive definite only when every transition has the same sign
  of $f_a-f_b$, as in the raising channel of a saturated ferromagnet, and not for an
  antiferromagnet, a Kondo lattice or the full channel. So the pole search counts the
  negative eigenvalues of $K$ (Sylvester's law of inertia) and checks, at every frequency it
  visits, that the count does not increase and that each root is a zero and not a jump
  through infinity. A review found that a pole of $K$ was returned as a mode by the first
  version of that check; it is repaired and tested.

When some operators couple to no transition, as $S_z$ in a saturated ferromagnet, $\chi_0$ is
singular and the kernel is written on its range. The case with a de-excitation half was
measured on a honeycomb antiferromagnet in a staggered field (sublattice imbalance 0.4,
$U=6$, $|Y|$ up to $0.27|X|$): the zero is the Casida energy to $10^{-9}$, the null vector is
the site part of the lift of $(X,Y)$ to $10^{-12}$ (0.87 with the wrong sign of $Y$), and its
magnon band has $C=0$ in routes 1 and 2 ($n_q=6,8$, `nk=6`).

Three limits, as planned, and two traps that were found:

- Route 2 works only below the continuum, and on a finite k-mesh any attractive vertex binds
  a spurious zero just below the lowest discrete transition, which moves into the continuum
  as `nk` grows (0.38208 against a pole at 0.38214 at `nk=12` on a spinon state, gone at
  `nk=24`). So the top of the window stays well below the continuum, and a mode is a mode
  only when it is stable in `nk`.
- With one moment per cell the kernel of one channel is a number and the bundle is trivial.
- $U'(q)$ is a parameter of the model, not a result.
- A positive `T` has to be below about the gap over 50, or thermal pairs put the continuum
  near zero; the default `T=0` has no thermal pairs, and the message of a window outside the
  continuum names a lower `T` as the remedy.
- A spin triplet in the full channel has three copies of one null vector, so the flux of a
  plaquette is three times the one of a single mode and passes $\pi$ on a coarse mesh. The
  Kondo exciton gives $C=0,2,1,3,3$ at $n_q=6,8,10,12,16$ there, with no error, against 1 from
  $n_q=6$ on in the raising channel. An integer is read only once it is stable in $n_q$.

The spinon bond RPA refuses the pair ladder, so route 2 is its only route. Its adaptor
reproduces `get_spinchi_full`, but the chiral honeycomb states tried ($J_2=0.5$ and 1.0)
have no mode band isolated over the zone, so there is no integer for them yet.

### Route 3: the response function as a Green's function

Built in `topologytk/greenwinding.py`: `window_chern` and `window_chern_density`, the
winding of any $G^{-1}(z,q)$ over a window, with `contour` and `get_derivative_names()`
(`"spectral"`, `"central"`); and in `chitk/responsewinding.py`: `bare_spin_chi_complex`,
$\chi_0$ at a complex frequency, `spin_kernel` (`kernel="response"`, `"dyson"` or `"bare"`,
`get_kernel_names()`) and `spin_response_chern`. The tests are
`tests/topology/test_green_winding.py` (13) and `tests/chi/test_response_winding.py` (9).
Since the decisions the defaults are `kernel="dyson"` and `channel="full"`, with `channel=`
in place of the old `components=` and taken from the registry of route 2, and
`greenwinding` takes its points from `plaquette.mesh`, keeping its own condition $n_q\geq3$.
Route 3 keeps `T=1e-3`, since its bubble is the one of `localspinchi`, whose Fermi function
divides by `T`.

The three changes of the plan are made. The integrand is the winding
$$N_3=\frac{1}{24\pi^2}\int d\omega\,d^2q\;\epsilon^{\mu\nu\rho}\,
\mathrm{Tr}\left[G\partial_\mu G^{-1}\,G\partial_\nu G^{-1}\,G\partial_\rho G^{-1}\right],$$
whose prefactor is the one of arXiv:1004.4229. The contour is a rectangle that crosses the
real axis at the two ends of a window $[\omega_a,\omega_b]$, and only its upper half is
evaluated when $G^{-1}(\bar z)=G^{-1}(z)^\dagger$. The bubble at a complex frequency comes
from `localspinchi.spin_chi` with `energies=x` and `delta=y`, which is $\chi_0$ at $x-iy$ to
$10^{-17}$, so the upper half plane is its Hermitian conjugate, at one call per imaginary
part. For $z-H$ the residues give $N_3=-C$, so the functions return $C=-N_3$, the sign of
`h.get_chern()` (checked on the Haldane model with both signs). The winding of the kernel of
pyqula is then minus the Chern number of the mode, since the residue vector of $\chi$ is the
conjugate of the amplitude of the mode, and `spin_response_chern` returns minus the winding.

The number is not an integer on a finite mesh. With the spectral derivative it converges
exponentially in $n_q$ ($2.7\times10^{-6}$ from the integer at $n_q=24$ on the Haldane model
with a gap of 0.9), and exponentially in the number of contour nodes `nz`. On coarse meshes
the approach is not monotonic: the lower magnon band of the shared magnet gives 1.166, 0.927,
1.0024, 1.023, 0.986, 1.0012, 1.0030, 0.9971 at $n_q=6$ to 13 (`nz=4`), so one mesh below
$n_q$ of about 12 says which integer it is and not how converged it is.

The trap of the plan, the zeros of $\chi_0$, is measured as a rule. A pole of $G^{-1}$ inside
the window adds minus the Chern number of its residue bundle, also when the residue does not
commute with $H$ (the windows of the test give the poles minus the zeros, 1.0000, 0.0000
and $-1.0000$ at $n_q=28$). On the shared magnet `kernel="bare"` winds to $10^{-12}$ or less
in both windows, but this says only
that the zeros of $\chi_0$ carry no net Chern number there; the direct check is that the
transverse $\chi_0$ is negative definite across the windows (largest eigenvalue $-0.024$ on
an $8\times8$ mesh of $q$).

The difference of $7\times10^{-6}$ between the zero of the kernel and the TDHF energy at a
generic $Q$ is the k-mesh. The bubble at $-Q$ pairs an empty state at $k$ with an occupied
one at $k-Q$, the TDHF an occupied state at $k$ with an empty one at $k+Q$, so the occupied
states are sampled on meshes shifted by $Q$: on a $Q$ of the k-mesh the two agree to
$10^{-14}$, and at $Q=(0.13,0.27)$ the difference is $7\times10^{-9}$ at `nk=12`.

The two limits the first version had are repaired. The old default, `components="xyz"`
with `kernel="response"`, raised for every magnet collinear along $z$ with a conserved $S_z$,
since the zz block of $\chi_0$ is exactly zero there. The Dyson kernel $1-\chi_0V$ inverts
nothing, so it takes a magnet along any direction, and the response kernel, when asked for
with a singular $\chi_0$, names the two remedies, `kernel="dyson"` or the channel of the
magnet. And the spin sector is selected: the block of a channel is built as in route 2, $T\chi_0T^\dagger$
and $T^{-\dagger}VT^{-1}$ with $T$ the matrix of the channel, keeping the leading block and
raising, with `channel="full"` named, when the rest couples to it. On the shared magnet the
raising channel gives the integer of route 2 and the lowering one zero (below $10^{-6}$). On
the Lieb ferrimagnet the raising channel holds the two bands of route 1 (0.487 to 0.908 and
1.099 to 1.270, $n_q=6$, `nk=6`), and with windows read from the record of route 1,
$(0.2437,1.0037)$ and $(1.0037,1.64)$, it gives $+1.0048$ and $-0.9985$ at $n_q=8$, `nz=4`,
and $+0.9992$ and $-0.9993$ at `nz=8`, the integers of route 1 and of route 2 in the same
channel; the lowering channel has one band, 0.324 to 1.185, across the end 1.0037, and the
full channel there gives $1.0021$ and $-0.9964$ at `nz=4`, close only because that band has
a trivial bundle. The price of the Dyson kernel, both halves of the contour, is removed by
evaluating each pair (Re $z$, $|$Im $z|$) once in `bare_spin_chi_complex`, since
$\chi_0(\bar z)=\chi_0(z)^\dagger$. Measured on the shared magnet at `nk=8`, $n_q=8$, `nz=8`,
the lower window, on an idle machine: before the change the default raised, the response
kernel on $(S_x,S_y)$ took 11.4 s and the Dyson kernel 13.6 s; after it the default, the
Dyson kernel in the full channel, takes 10.7 s (13.0 s with the bubble of the old code), the
same as the response kernel in the raising channel, 10.7 s, with $C=1.0036388$.

### The three routes on one magnet

The comparison of the plan, route 1 against route 2 and route 3 against both, is done on
one mean field: the honeycomb lattice with `tij=[1.0,0.3]`, `add_kane_mele(0.15)` and
`add_zeeman([0,0,3])`, converged with $U=8$, `filling=0.5`, `mf="ferro"`. The state is
saturated, `get_vev("sz")` is $-1$ on both sites, so the magnon problem has no de-excitation
half, and the onsite $U$ is the only interaction, so the site-basis RPA is the TDHF. The two
magnon bands are at 4.549 to 4.794 and 5.150 to 5.911 on the mesh with `nk=6` and $n_q=6$, and
at 4.548 to 4.827 and 5.102 to 5.935 with `nk=8` and $n_q=8$, and the Stoner continuum is 1.88
or more above the upper one. So the windows $(4.0,4.964)$, $(4.964,6.687)$ and $(4.0,6.687)$
hold the lower band, the upper band and both, at every $Q$. Every route uses the same `nk`,
the mesh of $Q$ is shifted by $(0.03,0.07)$, and `T=1e-3`. The mean field converged at
`nk=6` and at `nk=8` gives the same numbers to $10^{-14}$, since the down bands are filled
and the up bands empty whatever the mesh.

| Route | `nk`, $n_q$, `nz` | lower | upper | both |
|---|---|---|---|---|
| 1, `magnon_chern` | 6, 6; 8, 8; 8, 10 | $+1$ | $-1$ | 0 |
| 2, `kernel_chern` | 6, 6; 8, 8; 8, 10 | $+1$ | $-1$ | 0 |
| 3, `spin_response_chern` | 6 and 8, 8, 4 | 0.9950 and 0.9951 | $-1.0254$ and $-1.0255$ | $6.5\times10^{-4}$ |
| 3 | 6, 8, 8 | 0.9954 | $-0.9953$ | $-1.5\times10^{-8}$ |
| 3 | 8, 8, 8 | 0.9955 | $-0.9954$ | $-2.9\times10^{-7}$ |
| 3 | 6, 12, 4 or 8 | 0.99989 (`nz=4`) | $-1.0002$ (`nz=8`) | $6.5\times10^{-4}$ (`nz=4`) |

With `add_kane_mele(-0.15)` routes 1 and 2 give $-1$, $+1$, 0 at (6, 6) and (8, 8), and route
3 gives $-0.9950$, $+1.0254$, $-6.5\times10^{-4}$ at `nk=6`, $n_q=8$, `nz=4`. The routes need no
sign or conjugation from outside to agree. The upper window of route 3 is slower in `nz`,
measured as $-1.0254$ at `nz=4` against $-0.9953$ at `nz=8`. A reason for this (derived) is
that it is wider and the default height of the contour is half the width, so the contour is
taller against the distance of its ends to the bands, 0.13 or more over the zone.

The integers are a weak comparison for routes 2 and 3. Their null vectors live in the
two-dimensional space of $S^+$ on the two sites, so the bundles of the two bands add up to a
trivial one: the upper band has minus the integer of the lower one and the pair has 0 for
any two independent vectors (derived for route 3; for route 2 its review checked the pair on
random frames), and the window around both bands of route 3 is zero under any error of sign
or prefactor. Only the lower band and its flip are measurements there. Route 1 lives in the pair
basis, so its upper band and its pair are measurements, and their agreement is what checks
the isomorphism of the bundles of routes 1 and 2 that the plan derived. What links the
modules beyond the integers is measured too:

- The zeros of route 2 are the energies of the record of route 1 at every point of its mesh,
  to $5\times10^{-14}$ to $8\times10^{-14}$.
- The null vector of route 2 at $Q$ is the site part of the lifted state of route 1, the
  element $(i\uparrow,i\downarrow)$ of $\Phi_Q$ summed over $k$, to $10^{-15}$, and not its
  conjugate (0.943 and 0.989 at two generic $Q$).
- On a $Q$ of the k-mesh the kernel of route 3 at $-Q$ is singular at the energy of route 1
  (smallest over largest eigenvalue $2\times10^{-16}$), and its null vector in $(S_x,S_y)$ is
  $(\bar\phi,-i\bar\phi)$, with $\phi$ that site part, to $10^{-16}$, against 0.9998 for the
  unconjugated $(\phi,-i\phi)$. Off the k-mesh the two differ by the k-mesh offset of the
  section on route 3.

`tests/magnon/test_magnon_chern_routes.py` asserts all of this at `nk=6`: the links, the
three integers of routes 1 and 2 for both signs of the Kane-Mele coupling, with the windows
read from the record of route 1, and route 3 within 0.02 of the integer of route 1 at
$n_q=8$ (`nz=4` for the lower window, `nz=8` for the upper one). It runs in 28 s with a peak
of 452 MB, and the test of route 3 (17 s) is marked slow.

### The Kondo lattice

Three different things go under this name:

- The single-particle topology of the heavy-fermion bands. The mean field is a
  `Hamiltonian`, so `h.get_chern()`, the $Z_2$ and the rest apply today. The composite
  fermion Green's function is the mean-field one multiplied by constant amplitudes,
  $G_{FF}=|z|^2G^{mf}_{ff}$ and $G_{cF}=\bar z\,G^{mf}_{cf}$ (`kondotk/compositespectrum.py`),
  so it has the same invariants and adds nothing at the mean-field level.
- The spin exciton of the moments, through route 2 (`moment_rpa_kernel`, which reproduces
  `moment_spinchi` to $5\times10^{-8}$, bare and with $U'$). On the honeycomb Kondo lattice with two
  moments per cell, $J=4$ at half filling, a Haldane term $t_2$ on the conduction electrons
  and $U'=-4$, one spin exciton is bound below the continuum: at $t_2=0.2$ its band is 0.590
  to 1.357, the next band starts at 1.506 and the continuum at 1.695 ($n_q=12$,
  `nk=6, 8, 10`). Its Chern number is the sign of $t_2$ for $t_2=\pm0.1$ and $\pm0.2$, stable
  in $n_q=6$ to 12 and `nk=8` to 14 for $t_2>0$, and at $n_q=6,8$ for $t_2<0$. At $t_2=0$
  the two exciton bands touch at $K$ and $K'$, and the plaquette sum gives integers that
  depend on the mesh, with no error. With
  $U'=-1.5$ or $-2.5$ the band reaches into the continuum and there is no window. One
  independent check: a pair-space eigenproblem built from the Bloch Hamiltonian alone gives
  the same energy, the same state and $C=+1$ at $t_2=0.2$ (`nk=6`, $n_q=6$); the run for
  $t_2=-0.2$ did not fit in the time of one run. The physical reading, the time-reversal
  breaking of the conduction band gapping a Dirac touching of two exciton bands at $K$, is
  not derived. These numbers come from the build and the review of route 2.
- The spinon bond RPA, also through route 2, with no integer yet (see the section on route 2).

### Tests

How each test of the plan came out:

- Gauge. Done. For the BSE, `gauge=None`, `gauge="phase"` and random Bloch phases give the
  same lifted state (overlap 1) and the same integer; for the magnons random phases are put
  on the Bloch states, since `magnon_matrix` takes no `gauge`.
- Mesh. Done: the integers are stable in $n_q$ and `nk` on every model above. The plaquette
  sum is an integer on every mesh, so a single mesh proves nothing.
- Zero by construction. Done: an electron in a Chern band of the Haldane model bound to a
  hole in a filled flat level at the hexagon center, with no hopping, has $C=0$ to
  $10^{-16}$, whatever the Chern number of the electron band.
- A nonzero oracle. Built: the kagome lattice with its up triangles as molecules, a flux
  $\phi=0.3$ in each and a density interaction $V_1=1$. The BSE equals the exciton
  tight-binding model written from the interaction to $10^{-15}$ at $Q$, and misses the one at
  $-Q$ by 0.13, which pins the sign of $Q$; both give $C=-1$.
- Two routes. Done, see the section on the three routes on one magnet.
- External. Not done. The strong-coupling spin-wave model of the section on route 1 stands in
  for it, but it is our derivation; the linear spin-wave theory of SpinW or Sunny.jl is still
  the external oracle, since `spinwaves.py` has no Dzyaloshinskii-Moriya term.

The exciton formalism rests on arXiv:2507.22983, for the electron and hole conventions and
the Chern number that does not depend on them, and on arXiv:2003.11560, for the plaquette sum
on the cell-periodic exciton state. The exciton routes were run on model interactions only
(the density $V_1$); no screened or long-range interaction was tried.

### What to build, in order

1. The plaquette sum on a mesh of $Q$, with the corner states computed once and a raise when
   the count changes. Done, `topologytk/plaquette.py`; it takes `metric=`.
2. The lift of the BSE and of the `MagnonProblem` eigenvectors. Done, `bsetk/topology.py`,
   with the methods `h.get_magnon_chern()` and `h.get_exciton_chern()` on top of the general
   `mode_chern`.
3. Route 2: the pole search and the null vector, for `chitk`, the spinon RPA and the Kondo
   moments. Done, `chitk/kernelchern.py`, with the method
   `KondoLatticeHamiltonian.get_moment_magnon_chern()`.
4. Route 3: the factor $G\,\partial_zG^{-1}$, the window contour and the bubble at a complex
   frequency. Done, `topologytk/greenwinding.py` and `chitk/responsewinding.py`, with the
   channel of the spin sector and the Dyson kernel as default.
5. The user guide (sections in `spin-response.md`, `excitons.md` and `pseudofermions.md`, a
   pointer in `topological-invariants.md`, two reference entries and three figures), the
   README bullet, the notebook `topological_characterization/15_magnon_exciton_chern.ipynb`
   and the scripts `examples/2d/magnon_chern`, `examples/2d/exciton_chern` and
   `examples/kondolattice/moment_magnon_chern`. Done.

### What is left

The decisions of 13 November 2026 settle the shape: route 1 is the public route, with two
methods named by the excitation; route 2 is a method on the Kondo lattice only, since it is
the only route to the mode of the moments; route 3 is an independent check that returns a
number that is not an integer on a finite mesh, so it stays a module-level tool. No threshold
was added anywhere: the checks that cannot be made sharp (the gap, the distance below the
continuum, the largest flux) are fields of one record, and a touching at a point of the mesh
raises. What is deliberately unbuilt, with what would decide it:

- Route 2 as `method=` of `h.get_magnon_chern()`, as `h.get_magnon_bands()` has. For an onsite
  $U$ its zeros are the TDHF energies to $3\times10^{-14}$, so it gives the integers of route 1
  again and asks for a window chosen by hand; the keyword can be added later without breaking
  a call.
- A Chern method on `SpinonHamiltonian`. Its adaptor reproduces `get_spinchi_full`, but no
  mode band is isolated over the zone on the chiral states that were tried ($J_2=0.5$ and
  1.0), so there is no measured integer to document. A spinon mean field converged with
  `decoupling="bond"` refuses `get_magnon_chern`, as it refuses the other pair-basis methods.
- Route 3 as a method. It is the check of the other two, and its number converges to the
  integer only as $n_q$ and `nz` grow.
- The comparison of two meshes with no solve: the plaquette sum on every second point of an
  even mesh gives the integer of the mesh with half the points from the states already
  computed, and a difference between the two flags an unresolved mesh with no threshold. Not
  measured; as a gate it would refuse the tests that run at $n_q=6$, so it would be a field
  of the record and not a raise.
- A gap diagnostic between the points of the mesh. A touching between two points is not seen
  in any route, also not by `gap`, which is taken on the points of the mesh: the Kondo case at
  $t_2=0$ on the mesh with $n_q=10$ returns an integer and a gap of 0.14, as large as the gap
  of the gapped case $t_2=0.2$. A second $n_q$ is the check, and the default $n_q=12$ holds the
  symmetric points of the hexagonal and square zones. The windows of routes 2 and 3 stay chosen
  by hand, since a finite k-mesh binds a spurious zero just below the continuum (0.38208
  against an edge of 0.38214 on a spinon state), which an automatic window would count.
- An ungapped magnet with no net moment on a mesh that holds $Q=0$ raises or not depending on
  the sign of the residual of its mean field (the Goldstone energy comes out as the square root
  of the residual, real or imaginary). No sharp test removes this, since "ungapped" is defined
  only up to that residual; the rule, in the docstring and in the guide, is that such a magnet
  takes `shift=`. On the real side a group that holds the acoustic band gets a nearly defective
  vector at $Q=0$, and no measurement covers its integer.
- Five things stay as they are: only `solver="dense"` in route 1 (a large `nk` needs the
  iterative Tamm-Dancoff solver, whose amplitudes would lift the same way); `parallel.pcall`
  with more than one core is not tested; the memory of route 3 for a large cell, one
  $3N\times3N$ matrix per frequency and per $q$; the thermal Hall convention, whether the
  fluxes of the electron convention in the Casida metric are the Berry curvature that enters
  the thermal Hall conductivity (arXiv:1106.1987, arXiv:1204.3349); and the external check of
  the sign of the magnon integers with SpinW or Sunny.jl, for which the strong-coupling
  spin-wave model of the tests stands in.
- The Kondo result $C=\mathrm{sign}(t_2)$ is measured at $t_2=\pm0.1$ and $\pm0.2$ and has no
  derivation; its independent check with a pair-space eigenproblem exists at $t_2=0.2$ only.
  A degenerate group with a nonzero integer and `metal=True` have no test.

## Ruled out, and why

- The quadrupole operator $e^{2\pi ixy/L^2}$: it depends on the origin and on the boundary, and is quantized only when the protecting symmetries hold (arXiv:1902.07508); the nested Wilson loop and the corner charge measure the same physics without that ambiguity
- A quantized valley Chern number: it is not protected by any symmetry and reaches $\pm 1/2$ only in the Dirac limit (arXiv:1301.4205); the 1.089 that `examples/2d/valley_chern` prints is a valley-weighted curvature, which is fine as long as it is not read as an invariant
- The full elementary band representations: large, and limited by the scalar-orbital point group, see the section on symmetry eigenvalues
- The Kitaev Pfaffian number as a separate invariant in a periodic chain: it repeats the Zak phase, and is kept only as the finite-island parity

## Loose ends found by the survey

These were found while prototyping and were not repaired then. They were all looked at on the
same day as the nested Wilson loop, and each has its resolution at the end of its item: four
repaired, each with a test in `tests/topology/test_survey_loose_ends.py` that fails on the
unrepaired source, two left as they are by the maintainer's decision, two documented, and one
not reproduced.

- `transporttk.smatrix`: on a normal lead attached to a Kitaev lead at $\mu=-1$ (Zeeman 5 or 20, broadening $10^{-4}$ or $10^{-6}$) the Fisher-Lee scattering matrix came out non-unitary by order one ($\lVert SS^\dagger-1\rVert$ between 5.5 and 38), and the default `check=True` of `unitarize.check_and_fix` repaired it without a warning into $\det r=+0.14$, the wrong sign. Reported by the survey and not reproduced here; the Sancho-Rubio wrong-convergence trap at an onsite level is a candidate cause. A scattering invariant must at least refuse $\lvert\lvert\det r\rvert-1\rvert$ above a tolerance. **Not reproduced.** A normal lead (a spinful chain with `add_swave(0.)`, at half filling or with an onsite $-0.5$) attached to a Kitaev lead (onsite $Z+\mu$, Zeeman $Z=5$ and 20, `add_pairing(mode="pwave",delta=0.3,d=[1,0,0])`) at $\mu=-3,-1,0,1,1.5,3$, with `ht.delta` $10^{-4}$ (also its default) and $10^{-6}$: at $10^{-4}$ the lead self-energies are resolved only at that broadening, with a warning, the unitarity error is $3.5$ to $4\cdot 10^{-4}$ and $\det r$ is $-0.999$ inside the topological range and $+1$ outside, before and after `check_and_fix`; at $10^{-6}$ the normal lead at $\mu\neq 0$ raises from the Dyson-residual check of `greentk.rg` instead of answering. The guard above is still what the scattering invariant should carry
- `occ_states_sector_generator`: its tolerance of $10^{-3}$ on the operator eigenvalue rejects every state of $Ps_zP$ once Rashba coupling is present. **Repaired:** it now checks that the operator maps the occupied states into their own span, which it does when it commutes with the Hamiltonian, and raises otherwise, pointing to the split by the sign of $POP$; with Rashba coupling and $s_z$ it used to return 0
- `filter_state`: it reads the real part of the operator's matrix elements, so an operator with eigenvalues $\pm i$ gives zero in both sectors. **Repaired:** it raises when the operator is not Hermitian on the states (and `disentangle_manifold` diagonalizes with `eigh`, which reads one triangle only), saying to pass $iM$
- `examples/2d/mirror_chern`: its layer operator is a mirror only at `ti=0.0`. **Documented:** the script says that it is the sector Chern number of an operator that commutes only because the layers decouple, that AB stacking has no mirror, and points to `h.get_mirror_chern()`
- `h.get_hk_gen()`: a complex momentum is cast to a real one with only a `ComplexWarning`. **Repaired** in the dense generator (`htk/bloch.py`), which raises `TypeError` for a finite imaginary part and accepts a complex array whose imaginary part is zero; the sparse generator evaluates $e^{2\pi i k\cdot R}$ at the complex $k$ and is left as it is, and the jax path is not checked, since its $k$ can be traced
- `find_point_group`: the default center misses the inversion of the diamond lattice, which sits at the bond midpoint. **Left as it is, by the maintainer's decision:** widening the default centers would change what every caller finds, the group averaging of the Wannierization among them, and the docstring already says to pass `centers=`; it now names the diamond case
- `densitymatrix.occupied_projector`: it returns the transpose of $P$, which flips the sign of anything built from $s_z$ or a current, as recorded for `full_dm`. **Left as it is, by the maintainer's decision:** it keeps the index convention of `full_dm`, its docstring says so, and its one caller in the package takes `.T`
- `wannier_centers`, `z2_invariant`: their `nocc` argument is accepted and ignored. **Repaired:** `occupied_states` takes `nocc`, the lowest `nocc` states whatever their energy, and `wannier_centers`, `wannier_winding`, `z2_wannier_winding` and `z2_invariant` pass it on, so `h.get_chern(integration="wannier",nocc=...)` works too
- The operator `"sz"` of a Nambu Hamiltonian is the physical spin, $\sigma_z\tau_0$ in pyqula's basis, which equal-spin pairing does not conserve; the label of the two decoupled sectors of a helical p-wave superconductor is $\sigma_z\tau_z$, which is why `get_chern(operator="sz")` returns 0 there (measured $-2\cdot 10^{-8}$, and $-2.006$ with $\sigma_z\tau_z$ at `nk=20`, on the helical p-wave at $\mu=-2$). Not a bug, but a user will trip on it. **Documented** in the docstring of `topology.chern` and in the guide's section on the $Z_2$ invariant of a superconductor

## Higher-order topology: what is left after the nested Wilson loop

This section, written on 5 November 2026, collects in one place the higher-order pieces that
are still missing, so that the next one can be picked without reading the whole survey again.
What exists is the quadrupole moment $q_{xy}$ and the Wannier-sector polarizations from the
nested Wilson loop, with the corner modes of `square_2OTI` shown on a flake in
`examples/0d/cornermodes`. What is missing is any integer that counts corner states, any
corner charge, anything in three dimensions, and anything that works without translation
symmetry. The arXiv references below were checked against their arXiv records on that day.
There are four candidates, and each one follows the `new-feature` checklist: one function in
`topologytk/`, a delegator on `Hamiltonian`, one test against an independent route, and the
five documentation surfaces.

- **Multipole chiral number** (Benalcazar and Cerjan, arXiv:2109.06892, PRL 128, 127601).
  Built on 13 November 2026 as `h.get_multipole_chiral_number(cells=(Lx,Ly),chiral=S)` in
  `topologytk/multipolechiral.py`, on a supercell like the Bott index. The paper defines it on
  the torus and computes it on the open flake, so both are there (`boundary=`). $x,y$ count
  the unit cells from a corner; a centered origin gives 0 in the topological phase. The
  determinant of $\bar Q_A\bar Q_B^\dagger$ is one, so the number is an integer for any
  sample, and a sample that is too small gives a wrong integer, not a fraction. On
  `square_2OTI` it is 0 for $\delta<0$ and 1 for $\delta=0.5$ from $L=6$ to 14, also for
  $8\times12$; at $\delta=0.2$ the torus gives 1 from $L=6$ and the flake only from $L=10$.
  It equals the zero modes of the open flake, A minus B, at the corners $(0,0)$ and $(L,L)$,
  checked at $\delta=0.5$ for $L=10$ and 14. Chiral bond disorder up to 0.4 of each bond
  keeps it at 1, two coupled copies give 2. `g.get_sublattice()` has a random global sign,
  so `chiral=None` pins the first orbital on A. An independent review re-derived the
  formula from the TeX and the fully dimerized limit by hand,
  $N_{xy}=\mathrm{round}((L_x-1)(L_y-1)/L_xL_y)$ with the corner origin, which is why a
  sample needs at least 4 cells along each direction. Not done: the multipole winding number of
  Lin and Lee (arXiv:2401.03699), the octupole $N_{xyz}$, the $N_{xy}=4$ phases of the paper,
  the README bullet and a notebook.
- **Corner charge and filling anomaly from $C_n$ eigenvalues** (Benalcazar, Li and Hughes,
  arXiv:1809.02142, PRB 99, 245151). The corner charge $Q_n$ in units of $e/n$ comes from
  the rotation invariants $[\Pi_p]=\#\Pi_p-\#\Gamma_p$ of the occupied bands, for instance
  $Q_4=\frac{e}{4}([X_1]+2[M_1]+3[M_2])$, valid at zero polarization and with $C_n^n=+1$.
  It is the second-tier item already listed under symmetry eigenvalues. Its primitive,
  `h.get_symmetry_eigenvalues()`, is built (see the section on it above). The test is the $C_6$ Kekulé honeycomb at
  half filling (the geometry exists in `kekule.py` and `examples/2d/kekule_hopping`), with a
  corner charge $e/2$ checked against the charge counted on a $C_6$-symmetric flake around
  one corner, with the filling anomaly removed. Cost: small once the primitive is there,
  which is itself small.
- **Octupole moment and the nested loop on a plane of a 3D zone** (arXiv:1611.07987, section
  on the octupole, algorithm in arXiv:1708.04230). A third nesting of the Wilson loop in
  `nestedwilson.py`, and the `kfix` of `wannier_centers` passed down to the nested loop. It
  needs a 3D BBH builder next to `square_2OTI` in `specialhamiltoniantk/highorderTI.py`. The
  tests are $o_{xyz}=1/2$ on the cubic BBH model with a positive dimerization and 0 with a
  negative one, and $q_{xy}$ on a $k_z$ plane of a stack of decoupled 2D BBH layers equal to
  the 2D value at every $k_z$. Cost: medium, since the Wannier gap has to be tracked at two
  levels of nesting.
- **Second-order topology in three dimensions** (Schindler and others, arXiv:1708.03636,
  Sci. Adv. 4, eaat0346). The hinge modes of a $C_4T$-symmetric or inversion-symmetric
  insulator are diagnosed by the $Z_4$ indicator, which is equal to 2 with vanishing weak
  indices, together with a vanishing strong $Z_2$. This needs the same symmetry primitive as
  the corner charge, and then it is a few lines. The test is the $Z_4=2$ already measured on
  diamond in the survey, with a staggered magnetization that breaks time reversal but keeps
  $C_4T$, and the hinge states counted on a rod. Cost: small after the primitive.

Two small loose ends of the nested Wilson loop stay as they are recorded above: the Wannier
sectors chosen by a window rather than by the lines 0 and 1/2, and the $C_4$ route of eq.
VI.53, which also waits for the symmetry primitive. As a result the plan has two branches.
The multipole chiral number stands alone. The symmetry-eigenvalue primitive unlocks the
corner charge, the $Z_4$ second-order indicator and the $C_4$ route together, and also the
Fu-Kane and Chern-mod-$n$ items of the second tier. The maintainer picked the symmetry branch
first, and its primitive, the Chern number modulo $n$ and the parity $Z_2$ are built, see their
section above.

## The recommended order

The first tier is chosen by two things, how much of it already exists and how much the
models pyqula users already build ask for it. The Wilson-loop generalization is the
building block of the 3D $Z_2$, the nested loop, the Euler class and the atomic-gauge
polarization, and the 3D $Z_2$ was the missing branch of `get_topological_invariant()`; both
are now built. The chiral winding and the helical-wire $Z_2$, also built, are what the
superconducting examples need (Kitaev chains, Shiba chains, doped topological insulator
wires), and the second one uses the time reversal repaired here. The Prodan spin Chern
number and the automatic mirror Chern number, built third, replace two physics shortcuts in
the current examples at almost no cost. The Bott and spin Bott indices are the invariants of the
amorphous and disordered systems the real-space chapter already works with.

The second tier is the symmetry-eigenvalue primitive and what it feeds, the local $Z_2$
markers, the scattering invariant once the scattering matrix is understood, and the small
non-Hermitian, Weyl and interacting wrappers. The third tier is the medium and large
pieces: the nested Wilson loop and the Euler class, the KPM marker, the spectral localizer,
the magnon and exciton Chern numbers (see the section on collective modes), and the
Floquet, axion, GBZ and Hopf invariants.

The general Wilson loop with the 3D $Z_2$ was picked first, the one-dimensional invariants
second and the spin and mirror Chern numbers third, and all three are built, see the sections
after the repairs; the nested Wilson loop, from the third tier, was built fourth, together with
the loose ends, and the Bott and spin Bott indices fifth, which closes the first tier. The
magnon and exciton Chern numbers, from the third tier, were built next, on 13 November 2026,
with the magnon of the moments of a Kondo lattice: three routes, the methods
`h.get_magnon_chern()`, `h.get_exciton_chern()` and `get_moment_magnon_chern()`, and their
guide sections, see the section on collective modes, whose last part lists what is left of
them. The next decision is which piece of the second tier starts; each wants the
maintainer's sign-off, and each is one function, one test against an independent route and
one section of the guide.

## Repairs of the seventh bug audit

- **`chern_mod_n` did not follow the order of a1 and a2** (topo1#0). On a left-handed pair it gave $-C$ modulo $n$ for $n=3,4,6$. `geometry.honeycomb_lattice(4)`, (7), (9) and (13) are left-handed, and the Haldane model with $t_2=0.1$ on `honeycomb_lattice(4)` gave 1 for $n=3$ against `get_chern()=-1`. Now repaired; tested on the Haldane and QWZ models with a1 and a2 exchanged, and on `honeycomb_lattice(4)`.
- **$\kappa_1$ depends on the inversion center when a weak index is not zero** (topo1#1). The docstring says this now, and the tests check the $2\nu_i$ rule and the sign independently.
- **`corner_charge(n=6)` did not check the filling at K** (topo1#2). `add_kekule(0.3)` on `honeycomb_lattice_C6()` with the Fermi level at 2.41 has 5, 5 and 6 occupied states at Gamma, M and K, and returned 5/6; now it raises.
- **The Kekulé corner-charge test used the default n** (topo1#3); it passes `n=6` now.
- **Haldane normalization in the KPM marker texts** (nov13#2): gap $9t_2=0.9$ and critical mass $4.5t_2=0.45$ for `add_haldane(t2)`, not 1.04 and $3\sqrt3t_2$.
