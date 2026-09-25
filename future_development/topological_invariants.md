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
below, only the disordered versions (the scattering and the real-space invariants); in two dimensions there is no polarization, no invariant computed from symmetry
eigenvalues and nothing for higher-order or fragile topology; in real space there is no
$Z_2$ marker, no Bott index and nothing that scales past dense diagonalization; in three
dimensions there is nothing beyond the $Z_2$ indices and the Chern vector; and the non-Hermitian, Floquet, interacting and magnon
cases have no invariant of their own.

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
| Bott and spin Bott index | A, AII, 2D torus | a supercell at $\Gamma$ | Haldane $B=\pm 1$; Kane-Mele spin Bott 1 below $m_c$ | small | 1 |
| Symmetry eigenvalues at high-symmetry momenta | point groups, 2D and 3D | `pointgroup.compile_symmetry` | the parities above | small | 2 |
| Fu-Kane parity $Z_2$, $Z_4$ indicator, Chern mod $n$, corner charge | inversion, $C_n$, 2D and 3D | the eigenvalues | diamond $Z_4=2$ in the strong phase (measured) | small each | 2 |
| Local $Z_2$ and spin Chern markers, Katsura-Koma index | AII, 2D, disordered | `real_space_chern` | Kane-Mele island, bulk 1 below $m_c$ | small | 2 |
| Scattering invariant $\mathrm{sign}\det r$ | D (DIII, BDI), 1D disordered | `transporttk.smatrix` | Kitaev wire $-1$ inside, $+1$ outside (measured) | small, after a guard | 2 |
| Point-gap winding, biorthogonal Chern | non-Hermitian, 1D and 2D | `get_hk_gen`, `mesh_chern` | Hatano-Nelson $w=\mathrm{sign}(g)$ | small | 2 |
| 3D Chern vector | A, 3D | the Wilson loop on three planes | stacked Haldane layers, $(0,0,C)$ and $(C,0,0)$ | small | done |
| Weyl-point chirality | A, 3D | the sliced plaquette sum | chirality $\pm 1$ per node | small to medium | 2 |
| Topological Hamiltonian $-G^{-1}(0,k)$ | interacting A, AII | a custom generator | Haldane plus flat bath levels | small | 2 |
| Nested Wilson loop, quadrupole | $M_x,M_y$, 2D | the Wilson loop | `square_2OTI` $q_{xy}=1/2$ | medium | 3 |
| Euler class | $C_2T$, 2D | the Wilson loop | three-band model $\pm 2$ (measured) | medium | 3 |
| KPM stochastic Chern marker, Streda formula | A, 2D and 3D, millions of sites | `kpmtk` | Haldane at $10^5$ sites against `get_chern` | medium | 3 |
| Spectral localizer | every class, gapless too | positions, a signature | Haldane island index 1 in the bulk | medium | 3 |
| Magnon Chern number, thermal Hall | bosonic BdG, 2D | a paraunitary solver | Kitaev ferromagnet in a [111] field, $C=\pm 1$ | medium to large | 3 |
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

The KPM stochastic Chern marker (arXiv:1905.02215, with the KPM Kubo-Bastin conductivity of
arXiv:1410.8140) expands $P=\theta(E_F-H)$ in Chebyshev polynomials, applies it to random
vectors on a region, and evaluates $2\pi i\,\mathrm{Tr}_A[PXP,PYP]$, at a cost linear in
the number of sites, with a stochastic error $\sqrt{\xi^d/(R\lvert S\rvert)}$. The
expansion coefficient $\mu_0=1-\arccos(\epsilon)/\pi$ is misprinted in the TeX of
1905.02215. What `kpmtk` lacks is the primitive that applies the expanded $P$ to a vector
and returns vectors rather than moments; with it, the Streda formula
$C=(h/e)\,\partial N/\partial B$ from KPM state counts at commensurate fluxes and Kitaev's
formula $C=12\pi i\sum_{j\in A,k\in B,l\in C}(P_{jk}P_{kl}P_{lj}-P_{jl}P_{lk}P_{kj})$
(cond-mat/0506438) come almost for free. Oracle: the Zenodo record 2667604 of the paper
(BSD-2, on top of Kwant).

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
covers it; the case that needs the new solver is the one where terms that do not conserve
the magnon number carry the topology, the Kitaev ferromagnet in a [111] field
($C=\pm 1$ for $h/S\geq 4$). The Casida matrix of `bsetk/spinflip.py` has the same
paraunitary form, so a magnon Chern number from time-dependent Hartree-Fock, which no other
code offers, is within reach once the solver exists. Oracles: SpinW (GPL-3), Sunny.jl (MIT).

## Ruled out, and why

- The quadrupole operator $e^{2\pi ixy/L^2}$: it depends on the origin and on the boundary, and is quantized only when the protecting symmetries hold (arXiv:1902.07508); the nested Wilson loop and the corner charge measure the same physics without that ambiguity
- A quantized valley Chern number: it is not protected by any symmetry and reaches $\pm 1/2$ only in the Dirac limit (arXiv:1301.4205); the 1.089 that `examples/2d/valley_chern` prints is a valley-weighted curvature, which is fine as long as it is not read as an invariant
- The full elementary band representations: large, and limited by the scalar-orbital point group, see the section on symmetry eigenvalues
- The Kitaev Pfaffian number as a separate invariant in a periodic chain: it repeats the Zak phase, and is kept only as the finite-island parity

## Loose ends found by the survey

These were found while prototyping and were not repaired; each needs a look before the
invariant that depends on it is built.

- `transporttk.smatrix`: on a normal lead attached to a Kitaev lead at $\mu=-1$ (Zeeman 5 or 20, broadening $10^{-4}$ or $10^{-6}$) the Fisher-Lee scattering matrix came out non-unitary by order one ($\lVert SS^\dagger-1\rVert$ between 5.5 and 38), and the default `check=True` of `unitarize.check_and_fix` repaired it without a warning into $\det r=+0.14$, the wrong sign. Reported by the survey and not reproduced here; the Sancho-Rubio wrong-convergence trap at an onsite level is a candidate cause. A scattering invariant must at least refuse $\lvert\lvert\det r\rvert-1\rvert$ above a tolerance
- `occ_states_sector_generator`: its tolerance of $10^{-3}$ on the operator eigenvalue rejects every state of $Ps_zP$ once Rashba coupling is present
- `filter_state`: it reads the real part of the operator's matrix elements, so an operator with eigenvalues $\pm i$ gives zero in both sectors
- `examples/2d/mirror_chern`: its layer operator is a mirror only at `ti=0.0`
- `h.get_hk_gen()`: a complex momentum is cast to a real one with only a `ComplexWarning`
- `find_point_group`: the default center misses the inversion of the diamond lattice, which sits at the bond midpoint
- `densitymatrix.occupied_projector`: it returns the transpose of $P$, which flips the sign of anything built from $s_z$ or a current, as recorded for `full_dm`
- `wannier_centers`, `z2_invariant`: their `nocc` argument is accepted and ignored
- The operator `"sz"` of a Nambu Hamiltonian is the physical spin, $\sigma_z\tau_0$ in pyqula's basis, which equal-spin pairing does not conserve; the label of the two decoupled sectors of a helical p-wave superconductor is $\sigma_z\tau_z$, which is why `get_chern(operator="sz")` returns 0 there. Not a bug, but a user will trip on it

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
the magnon Chern number, and the Floquet, axion, GBZ and Hopf invariants.

The general Wilson loop with the 3D $Z_2$ was picked first, the one-dimensional invariants
second and the spin and mirror Chern numbers third, and all three are built, see the sections
after the repairs. What is left of the first tier is the Bott and spin Bott index, and the next
decision is whether it comes next or the second tier starts; each wants the maintainer's sign-off, and each is one function, one test against
an independent route and one section of the guide.
