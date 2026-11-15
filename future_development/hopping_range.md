# The range of a hopping given as a function

A hopping given to `g.get_hamiltonian(tij=...)` as a function, or as a `HoppingGenerator`
(`specialhopping.NNG`, `ILG` and the others), on a 1D or 2D lattice and with the default
`is_multicell=False`, was stored in the single-cell form (`inter`, or `tx`, `ty`, `txy`,
`txmy`), which visits only the 9 cells with $|n_1|,|n_2|\le1$. A hopping to a farther cell was
dropped without a warning. The same shells given as a list, `tij=[t1,t2,...]`, go through the
multicell builder with a range set by the number of shells and are complete. This was found
on 15 November 2026, by a read-only audit agent and then by a run.

Repaired the same morning, with the first option below, which the maintainer picked: a
function or a generator in 1D and 2D is built as multicell. Both are evaluated on the cells of
`specialhopping.cells_within(g,rcut)`, every cell with a pair of sites within `rcut=5.0`, unless
a box of cells is asked for with `nc=` or `cutoff=` (the same thing, both at once is a
TypeError); `nc` defaults to `None` now, it was 2. A large cell still builds its neighbor cells
only. This is not the hard cut of the option as written below ("each distance shell kept or
dropped whole"): `rcut` selects the cells, and every pair of sites of a selected cell is
evaluated, so on a cell of several sites a shell just beyond `rcut` is kept in part, chosen by
the extent of the cell, which can break a rotation. A hopping that reaches beyond `rcut` needs a
larger `rcut`; a cut of every pair at `rcut` is a decision left to the maintainer, since it
would drop tails that the builders kept before. Two paths still ignore `rcut=` without a word,
where it was a TypeError before: a list `tij=[...]`, which sets its own box of cells, and a
generator with `nc=` or `cutoff=`. A Hermitian Hamiltonian is evaluated on one cell $R$ of each pair $R,-R$ (the first
nonzero index positive) and the hopping to $-R$ is the adjoint, which is what the single-cell
storage did. This matters for a function that is reciprocal but not periodic in the lattice:
`specialhamiltonian.topological_heavy_fermion_1d` takes the sign of a hopping from the position
of the bond, and a build of both directions gave it a Bloch Hamiltonian that is not Hermitian,
with no warning. A non-Hermitian Hamiltonian (`non_hermitian=True`) evaluates both directions.
On the way, a sparse build from a function makes sparse matrices (the multicell builder made
dense ones), a generator with `spinful_generator=True` is no longer doubled in spin, and
`h.add_peierls` takes a 1D multicell Hamiltonian, so a ribbon from a function still gets its
field; its spinful sparse branch indexed the positions with a float and failed.

Two callers read the single-cell attributes of a Hamiltonian built from a function:
`spinwaves.hp_heisenberg` and `spinwaves.szsz` (`h.inter`, `h.tx`...). They now loop over the
cells of the multicell storage. Other routines still refuse or cannot read a multicell
Hamiltonian and now fail, loudly, on a 1D or 2D one from a function, as they already did on one
from a list: `heterostructures.create_leads_and_central` (`heterostructures.build` converts
first and works), `h.add_peierls` in 2D, and some legacy readers (`kdos.write_surface_kpm` in
1D, `ldos.ldos_finite`, `multilayers.build_honeycomb_bilayer`).

`tests/hopping/test_hopping_range.py` holds a function and `NNG` against the list of the same
shells on the triangular lattice, the chain to the third neighbor, the square lattice to the
third shell and the honeycomb lattice to the fourth (these fail on the code before), a large
cell that reaches its neighbors only, `rcut=` and `cutoff=` on both paths, the hopping to $-R$
equal to the adjoint, the heavy-fermion chain against its single-cell model, a spinful
generator on a chain, and the Peierls phase of a ribbon from a function.
`tests/magnon/test_holstein_primakoff.py` checks the ferromagnetic magnon dispersion
$E(k)=\sum_d J(d)(1-\cos 2\pi k\cdot d)$ to the third shell on the chain and the square
lattice. The tables below are the state before the repair.

## What was measured

The lowest band at the generic momentum $(0.13,0.29)$, first-neighbor hopping $t_1=1$ and
second-neighbor $t_2=0.2$:

| lattice | `tij=[1,0.2]` | `tij=f` | `tij=NNG(g,[1,0.2])` | `tij=f, is_multicell=True` |
|---|---|---|---|---|
| triangular | -1.14647 | -0.66657 | -0.66657 | -1.14647 |
| chain of one site | 1.34398 | 1.36909 | 1.36909 | 1.34398 |

On the triangular lattice, with the 120 degree basis of `geometrytk/lattices.py`, the second
shell at $\sqrt3$ is at the cells $\pm(1,-1)$, $\pm(2,1)$ and $\pm(1,2)$, and only the first pair
is visited, so four of the six bonds are lost and the six-fold rotation is broken. The keywords
`nc`, `cutoff` and `rcut` pass the keyword check of `get_hamiltonian` and are not used on this
path, so `tij=f, rcut=8` still visits the 9 cells.

## What the audit found, most likely first

1. `g.get_hamiltonian(tij=f)` with a function or a generator, in 1D or 2D (`htk/g2h.py`,
   the non-multicell branch). Cases: the one-site chain loses $t_2$ entirely; the triangular
   lattice loses 4 of the 6 second-neighbor bonds; the square lattice loses the third shell at
   distance 2, the cell $(2,0)$; the honeycomb lattice is complete up to the third shell and
   keeps 2 of the 6 bonds of the fourth, at $\sqrt7$.
2. A generator with `is_multicell=True`, and `h.add_hopping_matrix(fm)`, at the default
   `nc=2` (`multicell.parametric_matrix`): the box of $\pm2$ cells. On the chain the third shell
   is lost, on the triangular lattice the fourth and fifth. On this path `cutoff=` or `rcut=`
   raise a TypeError, so only `nc` works.
3. The interlayer hopping `ILG`/`twisted_matrix` on the single-cell storage of a primitive
   bilayer, `multilayer_graphene(real=True)` and `examples/2d/tunable_hoppings`: in the AB
   bilayer two of the three partners at in-plane distance 2 (cells $(2,0)$ and $(0,-2)$) are
   lost, about $1.6\times10^{-3}t$ at `ti=0.3`; in the AA bilayer 4 of 6 of the $\sqrt7$ shell,
   about $5\times10^{-5}t$. This breaks C3 at that level. (Magnitudes from reading, not run.)
4. The `mgenerator` builders at `nc=2` (twisted multilayers, `GrapheneHamiltonian`,
   `triangular_pi_flux`, the in-plane valley operator, `add_kekule`): on a moire cell the box
   of $\pm2$ is enough at the default parameters; on a primitive bilayer passed as the geometry
   a pair is lost only for `lambi` below about 5, against the default 8. Pi flux, valley and
   Kekule are first-neighbor only and safe.

Also seen: `add_pairing` with a function loops over `neighbor_directions()` with
`geometry.ncells=2`, so a pairing beyond two cells is lost (since bfa3e2c a pairing of known
range takes every cell within it; a callable `mode` still visits the box, recorded in
`bug_audit_8.md`); `spinwaves.set_couplings` calls
`parametric_hopping` without `fc` and is broken. Safe: every list `tij`, functions with
`is_multicell=True` (box of $\pm5$ and `rcut=5.0`) of first-neighbor range, the single-cell
storage of first-neighbor or purely vertical couplings in a reduced cell, `add_strain` and
`add_peierls` (they only rescale the stored matrices), every 3D Hamiltonian (always multicell)
and every 0D one. A first-neighbor function is safe only in a reduced cell: a triangular cell
with $\mathbf a_2'=\mathbf a_2+2\mathbf a_1$ puts the first neighbor $\mathbf a_2$ at the cell
$(-2,1)$, which the 9 cells miss.

## The decision

The same problem was solved tonight for the couplings of the classical models
(`specialhopping.folded_interaction`, `generator_within`) and before for the shells of the mean
field (`shell_interaction` with `cells_within`). Options for the hoppings:

- A function or a generator in 1D and 2D goes through the multicell builder by default, on the
  cells of `cells_within(g,rcut)` with `rcut=5.0` and each distance shell kept or dropped whole,
  as `distance_cut_interaction` does. Complete for any range below `rcut`, and a large cell
  still visits only its neighbors, since `cells_within` has a range per direction. The returned
  Hamiltonian is then multicell, which most routines take through `get_multicell()` anyway, but
  code that reads `h.tx` or `h.inter` of a 2D or 1D Hamiltonian would have to be found first.
- Keep the single-cell storage when the function vanishes beyond the 9 cells, checked on the
  cells of `cells_within`, and switch to multicell otherwise: a conditional gate, which the
  maintainer prefers to avoid.
- Only warn when a nonzero coupling is found beyond the 9 cells: no change of result, cheap, but
  the user still has to know to pass `is_multicell=True`.

The test that would fail today and pass after: the triangular and the chain rows of the table,
a function against the list of the same shells, and a generator `NNG` against the list.
Recommendation: the first option, with the whole suite run after it, since `get_hamiltonian`
with a function is behind most model builders.

Decided by the maintainer on 15 November 2026, after the repair: `rcut` becomes a hard cut, every
pair of sites farther than `rcut` is zero, so the range is exact and no shell is kept in part; a
longer tail needs `rcut=` by hand. Since `rcut` is then a distance on pairs, the same kind of
range as a list of shells, `rcut=` with `tij=[t1,t2,...]` raises a TypeError. Built the same
day: `rcut=None` means 5.0 in a periodic system and every pair in a finite one, as for the
interactions (`distance_cut_interaction`), with the same tolerance of 1e-6, so a shell sitting
at `rcut` is kept. A box of cells (`nc=`, `cutoff=`) keeps every pair of its cells unless
`rcut=` is also given. On the honeycomb lattice the shell at $\sqrt{27}$, just beyond 5.0,
is now dropped whole, where the cells within 5.0 kept a part of it
(`tests/hopping/test_hopping_range.py`).
