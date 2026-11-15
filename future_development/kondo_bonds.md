# Interactions between different cells: the Kondo bonds and the range of the shells

This file records the work of 14 November 2026. The maintainer asked that
the Kondo lattice mean field, its spectral functions and its RPA work when
the Kondo interaction couples sites of different cells, and that the same
holds everywhere else it is relevant, the electronic mean fields included.

## What exists now

Before this, `KondoLatticeHamiltonian` had one Kondo pair per moment, the
moment and the conduction electron of its own site, and `J` was one number.
Now the coupling is a set of Kondo bonds. A bond `(c, m, R, J_b)` couples
the moment `m` of the cell 0 to the conduction site `c` of the cell `R`, and
is decoupled on its own, with the field
$V_b=-(J_b/2)\sum_a\langle c^\dagger_{c,R,a}f_{m,0,a}\rangle$, the term
$V_bf^\dagger_{m,0}c_{c,R}+\mathrm{h.c.}$ and the constant
$2|V_b|^2/J_b$. The multiplier $\lambda$ stays one per moment.

- `J` is a number (the onsite model, bit for bit the numbers of the code
  before, checked on the chain and on the honeycomb lattice) or a function
  `J(rc, rm)` of the position of the conduction site and of the site of the
  moment, with `rcut` (5.0 for a periodic system, every pair for a finite
  one, as for `Vr`). The function does not have to be symmetric, and bonds
  with $J_b=0$ are dropped. The choice of one keyword over `J1/J2/J3` was
  Fable's: `J1` already means a Heisenberg exchange in the electronic mean
  field, and a number is the function that is nonzero at zero distance.
- `h2.kondo_bonds` lists the bonds, sorted by moment and then by distance,
  and `h2.hybridization` has one field per bond in that order, so its
  shape is one per moment for a number `J`. The seed `mf=(V, lam)` takes
  one V per bond and one lam per moment, or a number for all.
- `_kondo_pairs` stays one per moment: the tip of the cotunneling spectrum,
  the composite fermion and the moment susceptibility are at one site.
  Everything that depends on the bonds is read from the Hamiltonian.
- `kondotk.pairs.scale_hybridization` multiplies every element between a c
  and an f orbital in every direction. Before, it scaled only the intracell
  block of each pair, which would have left an inter-cell bond unscaled.
- The `J=` of `get_composite_spectral_function` and `get_cotunneling_didv`
  takes the same number or function and uses the onsite bond of the moment
  (`pairs.onsite_coupling`); a moment with no onsite coupling raises, and
  `J=None` works for any coupling. The cotunneling amplitude now reads V
  from the matrix element, as the composite amplitude already did, instead
  of indexing `h.hybridization` by position, which would have taken the
  wrong bond.

## Conventions

The matrix `m[d]` of a Hamiltonian couples the orbital `i` of the cell 0 to
the orbital `j` of the cell `d`, and `get_density_matrix` returns
`dm[d][i,j]` $=\langle c^\dagger_{i,0}c_{j,d}\rangle$. So the bond of the
cell `R` is the element `[f, c]` of `m[R]`, its conjugate the element
`[c, f]` of `m[-R]`, and its order parameter the element `[c, f]` of
`dm[-R]`. The Fock term of `densitydensity.get_mf_normal` reads the bond
of `v[d][i,j]` from `dm[-d][j,i]` the same way.

The convergence check of V is now relative to the largest field of all the
bonds, not to each field. A bond whose field vanishes by symmetry (the
first-neighbor bond of a particle-hole symmetric chain with an onsite V)
sits at roundoff, where its own relative change is of order one and would
never converge. A trajectory where every V decays to the trivial solution
together still has a constant residual, which was the reason for the
relative check. For one moment per cell, or a uniform cell, nothing moves;
a non-uniform cell can now stop a few iterations earlier, at the same
tolerance.

Changed on 15 November 2026 (the eighth sweep, `bug_audit_8.md`): with a global residual a
small but genuine field converged only to `maxerror` times the largest field, so each field is
again measured relative to its own size, with a floor that the maintainer chose: a field is
measured against at least a tenth of the largest field, so a field zero by symmetry, or one
smaller than a tenth of the largest, counts as converged once it moves by less than
`maxerror/10` times the largest field. The total energy of the loop is the grand potential at
the smearing T, and the loop stores its T and k-mesh on the result.

## What was measured

`tests/kondolattice/test_kondolattice_bonds.py`, 14 tests, about 25 s:

- A function that is J at zero distance gives the onsite numbers to 1e-12.
- A moment coupled only to the conduction site of the cell R = +1, -1 or
  +2 is the onsite model with the conduction sites relabeled: V, lam, n_f
  and the energy agree to 1e-8, and A_cc, A_FF, g_F and the RPA
  susceptibility of the moments agree to 1e-7, also with
  `hybridization_scale=1.5`. This is what pins R against -R.
- A chain with three bonds of different couplings (onsite 0.8, 1.4 to the
  right, 0.5 to the left), with real fields, and the half-filled chain
  coupled to its neighbors only, with the imaginary fields of the guide:
  every field is the order parameter of the Bloch matrix written by hand
  from the definitions, to 1e-6, and the grand potential of that matrix
  with the constant of each bond is stationary in the real and imaginary
  part of every field (first derivative below 1e-4, second above 1e-2).
  The second chain is what tells V from conj(V) at R != 0.
- The half-filled chain with an onsite (4) and a first-neighbor (0.5)
  coupling, where the neighbor fields vanish by symmetry and decay from a
  seed of 1: the loop stops, with the onsite field of the onsite model and
  the neighbor fields at 6e-7 for `maxerror=1e-5`, a value set by the
  tolerance. It takes 1098 iterations against 35 with a seed of 0 on the
  neighbors; the slow part is lam, which the seed pushes off its
  particle-hole symmetric value and which relaxes with its T-scaled step.
- The chain with onsite and first-neighbor couplings in its primitive cell
  and in a cell of two sites: every bond field, lam and the energy per site
  agree to 1e-7, the local A_cc folds, and the sum of the moment
  susceptibility of the doubled cell gives the primitive one at q=0 and at
  the zone boundary, to 1e-7.
- A finite chain of six sites with onsite and first-neighbor bonds, all
  inside the one cell, against the eigenstates of the whole cluster.

The existing Kondo, tunneling and kernel Chern tests (144, with 10 skipped
by their parametrization) pass unchanged.

Physics found on the way, now in the guide with a figure: on the
half-filled chain with the coupling only to the two neighbors (J=2 each),
the loop gives $V_{\pm1}=\mp0.498i$, so $V(k)=0.995\sin k$, odd in k. It
opens a gap of 1.99 where the conduction band crosses the f-level, at
$k=\pm\pi/2$, and vanishes at $k=0$ and $\pi$, where the f-level stays at
the chemical potential: no full gap. A real symmetric seed (an even V(k),
which vanishes at the crossing) relaxes to this odd one, so the even
solution is unstable there; its energy was not compared.

## What is left

- The composite fermion of a bond, $F_{ij}=\tfrac23(\vec\sigma\cdot\vec
  S_j)c_i$, is the operator a non-onsite coupling creates in the equation
  of motion of $c_i$. Only the composite of the site is built, so a moment
  with no onsite coupling has no amplitude with `J=`. The mean-field
  estimate of $F_{ij}$ is $-\tfrac23\sum_s\langle f^\dagger_{js}c_{is}\rangle
  f_j$ for any pair, coupled or not, which would be the way to build it.
- No README bullet and no notebook: the bullet of the Kondo lattice covers
  it, and the notebook `14_kondo_lattice.ipynb` could take a section.
- The Kondo-Heisenberg model is still out of scope, as before.

## The electronic side: the third shell was dropped on small cells

The same day the electronic and spinon routes were checked with
interactions that reach two and three cells, against the same system in a
supercell (scripts kept in the session, not in the repository). Vr and Jr
passed everywhere: `specialhopping.distance_cut_interaction` sizes the
number of cells from the distance. The shells V1/V2/V3 and J1/J2/J3 did
not. Their dense builders called `g.get_hamiltonian(is_multicell=True,
mgenerator=...)` without `nc`, so the multicell builder took its default
`nc=2` and built only the cells two lattice vectors away. Every pair of a
shell further away was dropped, with no warning. On the one-site chain
that is the whole third shell; on a rectangular cell with $a_2=5$ too; and
on the square lattice in the skewed basis $a_2+3a_1$ half of every shell,
V1 included. The square, triangular, honeycomb and kagome lattices and
the chain supercells were complete.

Every route that takes the shells had it, and in each one the V3 or J3
result was exactly the result without that term:

- the mean field: exact diagonalization (spinless 0.24 per site off, the
  spinful VJinteraction 0.98 for V3 and 1.7e-2 for J3), the triangle
  integration, jax, the dense KPM route (the sparse KPM route was right),
  the BdG pairing with an attractive V3;
- the density RPA (`chitk/densitychi.py`), the spin RPA of the pair basis
  (with J3 dropped, `h.V` was onsite only, so the site vertex was used
  without a word), the Bethe-Salpeter density interaction;
- the spinon mean field, bond and Hartree-Fock decoupling, and its RPA.

The six builders (`densitydensity.Vinteraction`, the dense KPM route,
`spinspin._build_v` and `_build_density_v`, `densitychi`, and
`bsetk.interaction.density_interaction`) now go through one function,
`specialhopping.shell_interaction`, which evaluates the shells on every
cell that can hold a pair of the farthest shell with a nonzero coupling
(`specialhopping.cells_within`). The range is set per lattice direction,
the extent of the cell plus the distance in units of the spacing of the
lattice planes, so a long thin supercell does not pay its long side along
the short one. On every geometry where the old builder was complete the
two give the same matrices, and the new one is not slower: 0.008 s
against 2.1 s on a 40 x 1 square stripe (11 directions in both), 0.013 s
against 0.028 s on a 6 x 6 honeycomb supercell, the same on the cubic
lattice and on a slab of six layers. On the one-site chain it adds the two
directions of the third shell.

`tests/scf/test_shell_range.py` (18 tests, about 25 s) holds the shells
against the KD-tree builder of the sparse mean field on seven geometries
(the skewed and rectangular cells included), V3 on the chain against its
cell of three sites, and V3 or J3 against the same coupling given as Vr or
Jr in each route above; all 18 fail on the code before the repair, with a
wrong number. Checked by hand after it and not kept as tests: the magnon
bands of `method="rpa"`, `"pair"` and `"tdhf"`, `get_spinchi_ladder`, the
KPM route of VJinteraction and a 2D rectangular cell, where J3 or V3 now
gives exactly what Jr or Vr gives (1.9e-8 on the 2D cell) and differs from
the run without it. The Goldstone check of the spin RPA is not a test of
the range: it held with J3 dropped too, since the dropped state is
consistent.

Not reached: qtci, a Neel state and a supercell at nonzero Q. Left at
first, since they are neither a mean field nor an RPA: the classical
spin, lattice-gas and Ising models (`add_heisenberg`, `add_interaction`)
took a callable coupling through `get_hamiltonian(tij=Jij)`, whose
single-cell storage in 1D and 2D keeps only the neighboring cells, so a
coupling two cells away on a small cell was dropped there too; a list of
couplings goes through the multicell builder with a range set by the
number of shells and is complete.

Repaired on 15 November 2026, on a design chosen by a Fable agent: a
callable `Jij(r1,r2)` goes through `specialhopping.distance_cut_interaction`,
as the Vr of the mean field, with a new `rcut=None` (every pair of a finite
system, 5.0 for a periodic one) on the three methods, through one helper,
`specialhopping.folded_interaction`, which returns the couplings folded at
k=0. The old builder also made the pairs of a large cell depend on where a
site sits in it (the 3x3 block of cells is not centered on the site), so
the change of range is not a loss: the pairs beyond 5.0 that it kept on a
large cell were not a physical set. Nothing in the package passed a plain
callable. The route is stricter, since a coupling that is not reciprocal,
real and finite raises. `tests/classicalspin/test_coupling_range.py` (10
tests) holds, on the three models, a function against the same shells as a
list on the one-site chain, the energy per site of an alternating state in a
cell of two and of six sites, a uniform local energy on a 6x6 or 4x4 cell,
and every pair of a finite chain; all 10 fail on the code before. The same
night a `HoppingGenerator` (`specialhopping.NNG` and the others), which in 1D
and 2D also landed in the single-cell storage, took the same route:
`specialhopping.generator_within` evaluates its matrix generator on every
cell that holds a pair within `rcut` and drops the pairs beyond it; three
more tests hold `NNG(g,[0,J2])` against the list on the one-site chain, and
the couplings of the finite ladder of
`examples/classicalspin/modulated_ladder` are unchanged. The hoppings built
with `mgenerator` and the default `nc=2`, and more generally with a
function on the single-cell storage, were checked the same night and do
drop couplings on small cells; that is recorded, with a measurement and the
options, in `hopping_range.md`.
