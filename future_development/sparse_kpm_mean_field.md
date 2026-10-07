# The KPM mean field for large sparse systems

Status, 6 October 2026: **the four steps are built** (the construction
path, the sparse self-consistent loop under both KPM entry points, the
kernel with the lagged Fermi level, and the truncated recursion, below),
and an iteration at $10^5$ sites takes under a minute. This is the plan for a KPM mean field
(`h.get_mean_field_hamiltonian_kpm`, and `integration="kpm"` of
`h.get_mean_field_hamiltonian`) whose memory is linear in the number of
orbitals, so that a self-consistent calculation on $10^5$ sites, with
non-collinear magnetism or with a Nambu Hamiltonian, fits in memory. We will
now see what scales today and what does not, what sets the time, the four
steps in the order they are to be built, the checks, and the four decisions
the maintainer took. The engine this builds on is described in
[`gpu_kpm_mean_field.md`](gpu_kpm_mean_field.md).

## What scales today and what does not

The Chebyshev recursion is already linear in memory. The block recursion of
`kpmtk/pairmomentsjax.py` streams the starting orbitals in blocks capped by
`_MAX_BLOCK`, so at $N=2\times10^5$ orbitals it would run about 55 of them at
a time, on block arrays of about 0.5 GB. Its constant is set by XLA rather
than by that budget: a 3200-orbital spinful island sat at 6 GB, of which the
block arrays and the dense copies below account for about 2 GB, the rest
being XLA's working set (the gathered temporaries of the ELL product, the
`vmap` over k, the compilation). So the footprint of the sparse path is to be
measured, not computed.

What is quadratic is everything the recursion is wrapped in. At $10^5$ sites
a dense $N\times N$ complex matrix is 640 GB spinful and 2.6 TB with Nambu,
and the loop holds several. Found by reading the path and by building
spinful square islands with Rashba coupling at 900 and 10,000 sites (on a
laptop under other load, so the times are a scaling, not a rate):

- `Vinteraction_kpm` and `densitydensity_kpm`
  (`scftk/densitydensity_kpm.py`): `h.get_dense()` at entry, and the
  interaction `v` built as a dense $2n\times 2n$ matrix by a Python double
  loop over sites, $10^{10}$ iterations at $10^5$ sites.
- `neighbor.neighbor_distances`, which `Vinteraction_kpm` calls for
  $V_1$ to $V_3$: an $n\times n$ array of distances, 28 s at 10,000 sites
  and 80 GB at $10^5$.
- `kpmtk/densitymatrix_kpm.py::_dm_kpm_from_needed`: only the needed entries
  are computed, and they are written into a dense $N\times N$ matrix per
  direction; `get_dm_kpm` fills missing directions the same way.
- `scftk/densitydensity.py::random_hermitian_guess`: a dense guess.
- `normal_term_*_jit`, `get_dc_energy_jit` (`scftk/densitydensity.py`) and
  `anomalous_term_ij_jit` (`scftk/superscf.py`): loops over all $n^2$ pairs
  of dense matrices.
- `get_fermi4filling_kpm`: the exact trace by `kpm.full_trace`, one
  recursion per orbital, so the Fermi search costs as much as the density
  matrix; for Nambu it also copies h and removes the doubling every
  iteration.
- The tail of `densitydensity_kpm`: `h.get_total_energy(nk=h.nk)`, which
  densifies and diagonalizes; `get_total_energy_kpm` exists and refuses
  Nambu.
- On the construction path, `magnetism.add_magnetism` (behind
  `add_exchange`), `superconductivity.time_reversal` (behind
  `setup_nambu_spinor`) and the s-wave pairing (behind `add_swave`) build a
  block-diagonal operator with `scipy.sparse.bmat` on an $n\times n$ Python
  list of blocks: 90 s, 183 s and 54 s at 10,000 sites with a 2 GB peak.
  Building the sparse Hamiltonian, the Rashba coupling and `get_hk_gen` scale
  fine.
- `scftk/spinspin.py`: the `VJinteraction` KPM path keeps h sparse
  (`keep_sparse`), and still builds $n\times n$ boolean masks
  (`_build_sparse_pairs`) and a dense density matrix.

`mix_mf`, `diff_mf`, `update_hamiltonian`, `get_eh_sector` and
`build_nambu_matrix` keep a sparse matrix sparse, and return a dense
`np.matrix` only when one input is dense, so they need no rewrite once the
mean field is sparse, and neither does `obj2mf`, which passes a dictionary
through. The guesses and the constraints do, measured on spinful square
islands at 625 and 2500 sites, where a linear routine takes 4 times longer:

- `meanfield.guess` with `"random"`, `"Fully random"` or `"dimerization"`
  returns a dense $N\times N$ array, so the random guess most examples use
  densifies the loop from its first iteration.
- The guesses that stay sparse scale as 10 to 13 times for 4 times the
  sites (`"ferro"`, `"XY"`, `"magnetic"` and `"randomXY"` through the `bmat`
  of `add_exchange`, `"kekule"`, `"swave"`), and `"pwave"` goes through
  `superconductivity.pairing_block`, which calls the pairing function once
  for every pair of sites and assembles an $n\times n$ list of blocks: 39 s
  at 625 sites, 552 s and a 6 GB peak at 2500. This is the builder behind
  `add_pairing`, so every pairing other than the s-wave one is quadratic to
  build, not only the guess.
- `enforce_constrains` with `"no_magnetism"` or `"no_inplane_magnetism"`
  assigns the onsite spin entries one site at a time on a CSC matrix, and an
  assignment that changes the pattern rebuilds it (7 times for 4 times the
  sites); `"no_normal_term"` and `"no_SC"` on a Nambu mean field scale as
  10 to 13 times. `"no_charge"` and `"no_offplane_magnetism"` are linear.

## What sets the time

With every dense piece removed, a recursion started on every orbital is still
$N$ recursions of length $N$, meaning that the time per iteration grows as
$N^2$. Scaling the per-iteration times of the GPU roadmap (18 s on six CPU
cores and 2.8 s on a consumer card at 3072 orbitals; 20 s and 1.9 s for a
3456-orbital Nambu island) by $N^2$ gives roughly 13 min and 2 min at
20,000 orbitals, 20 h and 3 h at $2\times10^5$, and 75 h and 7 h at
$4\times10^5$ with Nambu. The last two are extrapolations by 4000 to 13000
in $N^2$ from one family of Hamiltonians at `npol=200`. Moment doubling,
$T_{2n}=2T_n^2-T_0$, halves the time and keeps the $N^2$. So the exact route
is practical up to about $10^4$ sites.

Linear time needs locality. After $n$ steps the vector started on orbital $j$
is nonzero only within $n$ hops of $j$, meaning that the moments
$\langle i|T_n(H)|j\rangle$ for $i$ near $j$, computed on $H$ restricted to
a ball of radius $R$ around $j$, are exact up to $n=2R$. Running each block
of starting orbitals on such a ball costs $N$ times the size of the ball
instead of $N^2$, and the memory is that of the ball. With `npol=200` on a 2D
lattice the exact radius is 200 hops, which is the whole $10^5$-site island,
so the gain comes only from truncating below it, and this is allowed by the
state, not by the code: the density matrix decays exponentially with a
length $v_F/\Delta$ in a gapped state (a few lattice constants for an
antiferromagnet at $U$ of a few $t$, tens for a superconductor at
$\Delta=0.1t$) and $v_F/T$ at finite temperature, while a metal at $T=0$ has a
power-law tail and no truncation of it is controlled. The radius is then a
keyword converged as `nk` is, by comparing the density matrix at two radii.

## The plan, in build order

**The construction path.** The $n\times n$ `bmat` in `magnetism.py`,
`superconductivity.time_reversal` and the s-wave pairing is replaced by a
direct block-diagonal build (COO arrays, or `kron` with the identity),
`neighbor_distances` by a `cKDTree` query (`neighbor.py` already uses one),
and `pairing_block` evaluates the pairing function only on the pairs within
its range, found the same way, instead of on every pair of sites. The
constraints act on the onsite blocks as arrays rather than entry by entry,
and the dense random guesses are drawn on the sparse pattern. This is
independent of the KPM and repairs those routines for every large sparse
Hamiltonian.

This step is built. Two more quadratic pieces turned up while it was
measured and are repaired with it: the sparse Nambu reordering
(`sctk/reorder.py`) grew its index lists by concatenation, and the Kekule
term (`kekule.py`) evaluated its bond function on every pair and found the
hexagon centers and the registry from arrays of every pairwise difference,
36 GB at 40,000 sites. A spinful square island with Rashba coupling, built
sparse on six desktop cores, now takes, at 10,000 and at 99,856 sites:
0.01 s and 0.05 s for the exchange, 0.01 s and 0.14 s for the Nambu
doubling, 0.03 s and 0.31 s for the neighbor shells, 0.5 s and 5.0 s for a
p-wave pairing, 0.4 s and 4.4 s for the Kekule guess, and below 0.1 s for
every constraint, with a peak of 1.3 GB for the whole sequence at $10^5$
sites. Before, on a laptop under load, the exchange took 90 s, the Nambu
doubling 183 s, the neighbor shells 28 s and the Kekule guess 216 s at
10,000 sites, and the p-wave guess 552 s with a 6 GB peak at 2500, each
growing as the square of the number of sites. Every term is
the same as before to 1e-12, checked on 337 cases (exchange, Zeeman and
antiferromagnetic fields, Nambu doubling, every registered pairing mode
inside a cell and towards every neighboring cell, the neighbor shells of
0D to 3D lattices, the constraints, the Kekule terms and registries), with
four changes a user can see:

- The random guess (`mf="random"`, `"Fully random"`, `"dimerization"`) of a
  sparse Hamiltonian is drawn on the sparsity pattern of h and on the onsite
  block of every site, and stays sparse; of a dense one it is the fully
  random matrix it was.
- `neighbor_distances(n=...)` returns the n shells asked for. It used to
  return every distinct distance, since a local variable overwrote `n`;
  every caller in the package reads only the first few.
- `add_pairing` with `mode="dpid"` or `"chiral_dwave"` and `nn>1` builds the
  pairing of the nn-th shell, where it stopped with an `AttributeError`.
- A Kekule term added to a sparse spinless island keeps its intracell
  matrix sparse, where it became a dense `np.matrix`.

A pairing whose range is unknown, given as a callable `mode`, still
evaluates every pair of sites, as does the antisymmetry check that runs on
it; the registered modes report their range to `pairing_block`, and
`tests/superconductivity/test_pairing_range.py` holds every one of them to
the evaluation of every pair. `tests/hopping/test_linear_construction.py`
bounds the peak memory of each step at 10,000 sites, where any $n\times n$
object is 0.8 GB.

**A sparse self-consistent loop on a fixed pattern.** The interaction
becomes a list of bonds $(d,i,j,V)$ built with `cKDTree`, the Hubbard $U$
being the onsite spin-flip bonds, which fixes once the density-matrix
entries the loop reads. The density matrix is a vector over those entries,
computed for $i\le j$ only, with the partner set by conjugation as the
present engine does: computing both from their own moments lets roundoff
open an anti-Hermitian part that the loop amplifies until the recursion
diverges (`bug_audit_5.md`). The mean field has a fixed pattern too (Hartree
on the diagonal, Fock on the bonds, pairing on the pairing bonds), so
$H_0+\mathrm{MF}$ lives on one CSR pattern built once, and each iteration only
replaces a data vector, with mixing and the convergence error as operations
on that vector. Hartree, Fock, pairing and the double counting become numba
loops over the bonds, linear in their number. The string guesses are drawn on
the same pattern, and the total energy comes from the KPM density of states,
with a BdG version added to `get_total_energy_kpm`. Non-collinear magnetism
comes with no extra work, since the spin-flip entries $(2i,2i+1)$ are
already needed by the Hubbard Fock term, and Nambu comes through
`required_elements_eh`, whose index maps exist, so `integration="kpm"` stops
refusing a Nambu Hamiltonian and the `VJinteraction` KPM path moves onto the
same core. The dense KPM engine stays as it is, and the entry points route
on the Hamiltonian they are given: a sparse h (`h.is_sparse`) goes to the
sparse engine by default, a dense one to the dense engine, so that
`tests/scf/test_kpm_block_density_matrix.py` and
`tests/scf/test_densitydensity_kpm.py` keep checking the dense one unchanged
and the two engines are held to each other by the first of the checks below.
This step gives linear memory, exact results and $N^2$ time.

This step is built, in `scftk/sparsemeanfield.py`, in a simpler form than
the one above: the interaction, the density matrix and the mean field are
dictionaries of CSR matrices holding only the entries the interaction
couples, rather than one fixed pattern with a data vector, and the mean
field is assembled with sparse-matrix operations rather than numba loops,
which already makes every step of an iteration other than the recursion
linear and negligible (the mean field of 20,000 orbitals took below 0.01 s,
as printed to two decimals). Both entry points route on `h.is_sparse`: `Vinteraction_kpm` (and
`hubbard_kpm`) build the interaction with a KD-tree (`site_pairs`), and
`VJinteraction` builds its four channels the same way, with the spin weights
$\pm 1/4$ of the exchange channels, and asks for the density matrix on whole
$2\times 2$ blocks when an exchange channel is decoupled in a rotated frame,
since the rotation mixes the two indexes of each block; its dense KPM path
still refuses a Nambu Hamiltonian, and its sparse one takes it. The density
matrix comes from the same `_kpm_pair_values` as the dense engine, split out
of `_dm_kpm_from_needed`, so the two engines agree to roundoff: the mean
field after six iterations differs by below 1e-15 on a non-collinear Rashba
Hubbard island, on a honeycomb lattice with $V_1$ on a k-mesh, on a Nambu
island with attractive $U$ and an in-plane field, and through `VJinteraction`
with isotropic and anisotropic exchange. The Nambu case with exchange, for
which there is no dense KPM path, converges to exact diagonalization at a
fixed chemical potential as `npol` grows (5e-3 in the normal part and 9e-3
in the pairing at `npol=150`, 2e-4 and 3e-4 at 600). The total energy is
$\mathrm{Tr}(H\rho)$ from the KPM density matrix on the hoppings
(`get_band_energy_kpm`), with the trace of the electron block added and
halved for Nambu as `spectrum.total_energy` does, taken at $T=0$ whatever
the temperature of the loop, since the dense engine's energy is the sum of
the occupied levels and the two have to stay comparable; against the sum of the
occupied levels its relative error is 1e-3 to 2e-5 at `npol=100` and 2e-7 to
5e-9 at 1600, on islands and on a k-mesh, normal and Nambu.

Two quadratic pieces outside the mean field turned up on the way and are
repaired with it: `MultiHopping.dot`, behind every Hermiticity check, made
both matrices dense (2 GB at 1600 Nambu sites), and `kpm.full_trace`, behind
the Fermi search, stacked one dense $N\times N$ block of site vectors; it now
takes them in chunks of $2^{24}$ entries. With the working buffers of the
recursion and of the trace made small, the traced memory of one iteration
grows from 0.012 to 0.020 to 0.044 GB from 400 to 1600 to 6400 sites, and
with the default buffers it saturates at a few hundred MB above the linear
part.

One iteration at 10,000 sites (20,000 spinful orbitals, `npol=100`) on six
desktop cores takes 103 s for the Fermi search and 292 s for the density
matrix, with a peak of 1.9 GB, which is the $N^2$ extrapolation of the dense
engine's measured times, and on a consumer card 43 s and 44 s, with 1.4 GB.
With Nambu (40,000 orbitals, the same island with attractive $U$) the
density matrix takes 1184 s on the CPU and 182 s on the card, four times
the normal case as $N^2$ says, at the same peak memory, while the Fermi
search stays at 103 s and 43 s, since it runs on the electron sector. The
Fermi search costs a third of an iteration in the normal case on the CPU,
which is what the lagged trace of the next step removes.

One difference with exact diagonalization that is a convention and not an
error: for a filling whose number of electrons is not a whole number, as
0.4 of the 18 electron states of a 9-site island, the Fermi search of
exact diagonalization rounds it to whole states (7 electrons) while the KPM
one holds the fractional count (7.2), so the two converge to slightly
different states; at a whole number of electrons the two Fermi levels agree
to 1e-5. The dense KPM engine has the same difference.

**The kernel and the Fermi level.** On the CPU the recursion becomes a numba
kernel, a CSR product with the block of starting orbitals in `prange` over
its columns, compiled once and cached, with jax's ELL kernel kept for the
card. The reasons are measured: jax recompiles for every block shape (one
compile of `_contracted` took 2 min at 3200 orbitals, and 400 orbitals spent
23 s in 19 compiles), XLA's thread count ignores `parallel.set_enabled(False)`,
and an in-place CSR product has none of the gathered temporaries behind the
memory constant above. On the card the blocks are padded to one shape, and
the zeros of the fixed pattern are kept as stored entries, since `_ell`
drops entries that are zero and a pairing amplitude passing through zero
would change the width and recompile the kernel. The moment doubling is added
for the pairs whose two orbitals are both starting orbitals, which is every
pair of an onsite interaction. The Fermi level comes from the exact trace of
the same recursion, the diagonal moments summed as they are produced, used
for the next iteration: free whenever every orbital is a starting orbital,
and exact at convergence, where the Hamiltonian no longer changes. It needs
one trace pass before the first iteration, and the convergence criterion
includes the filling error, since the iterates are not at the requested
filling until the mean field stops moving.

This step is built, with the doubling on the CPU only. The CPU kernel is
`kpmtk/pairmomentsnumba.py`, with the two entry points of the jax one, and
the package-wide switch picks between them. Three things set its speed,
measured on a spinful square island of 10,000 sites with Rashba coupling
(20,000 orbitals, ten stored entries per row) at `npol=100`. The block is
kept as two real arrays, one for the real and one for the imaginary part,
which numba vectorizes: a step of 256 columns took 17 ms against 38 ms with
a complex array (on a laptop), and a Hamiltonian whose entries are exactly
real at a k-point runs on the real part alone, 10 ms; exactly real, since a
tolerance would drop the small imaginary seed of a chiral state in a
mean-field loop. The width of the block: on six desktop cores a step costs
the same per starting column from 64 to 256 columns, and 1.6 and 2.4 times
more at 16 and 8, so a block holds at least 64 columns, and beyond 65,536
orbitals the memory of the blocks grows linearly, at 2 kB per orbital in
double precision. And the doubling, which on the pairs of an onsite Hubbard
interaction took 3.7 ms per starting column against 5.4 ms read on the
rows, 0.69 of the time rather than one half, since the inner products of
the pairs that are not diagonal are taken one pair at a time; those of the
diagonal pairs are the norms of the columns, taken for every column at
once. Whether a block is doubled is decided by a cost model, with the
orbitals of the pairs that are not starting columns added to the block when
this pays, and never for every pair of a small cell, which the dense engine
asks for on a k-mesh and whose inner products would cost $N^3$ per step.
The jax engine of the previous step took 292 s for the same density
matrix on the same cores, 14.6 ms per starting column, so the kernel is
about four times faster. A first version split the rows of every block
among the threads, which on a small cell costs more than the rows
themselves: a two-orbital chain on 8 k-points took 0.29 s per density
matrix against 5 ms with jax, which runs every k-point in one call. So a
cell small enough runs one k-point per thread, and a block of at most 4096
entries runs its rows in one thread, after which, on a laptop, the chain
takes 1.5 ms, a six-orbital cell on 144 k-points 58 ms against 82 ms with
jax, and honeycomb islands of 144 and 576 orbitals 0.04 s and 0.28 s
against 0.10 s and 1.30 s. It agrees with the jax kernel to 1e-12 in double
precision (`tests/scf/test_kpm_block_density_matrix.py`), and becomes
serial with `parallel.set_enabled(False)`, which jax's CPU backend did not.

The Fermi level is lagged in both engines, the dense and the sparse one
(`LaggedFermi` in `kpmtk/densitymatrix_kpm.py`). The diagonal pair of every
orbital is added to the pairs of the recursion, and the sum of their
moments is the trace the Fermi search inverts. The first Hamiltonian gets
an exact search, and so does every Nambu one, whose Fermi level is that of
the electron-only Hamiltonian, a different matrix from the one the
recursion runs on. The convergence check holds the filling error, read
from the same trace, to the tolerance together with the change in the mean
field. Three consequences were measured, all in
`tests/scf/test_kpm_lagged_fermi.py` or `tests/scf/test_vjinteraction_kpm.py`.
The path to convergence is not that of an exact search, so where several
self-consistent states exist the loop can settle in another one: on the
spinful chain at $U=5$, $J_{1z}=-1$ and filling 0.2 the lagged loop reaches
the fully polarized state, at $E=-0.606$, where exact diagonalization from
the same guess stops in a partially polarized one at $E=-0.564$, and stays
in the polarized one when it is started there; the test that compared the
two moved to filling 0.15, where both reach the polarized state. The
converged state is not that of an exact search either, by what the
expansion resolves: the lagged loop ends at the Fermi level whose filling,
read on the expansion of the shifted Hamiltonian, is the requested one,
while an exact search reads it on the unshifted one, whose scale is
different, and on a 16-site Hubbard island at `npol=150` the two differ by
2e-3 in the Fermi level and 1.5e-2 in the mean field. A final exact search,
to return the Fermi level of the returned Hamiltonian, was tried and
dropped, since it gives up the filling the convergence check measured. And
on that island the lagged loop took 40 iterations where an exact search
every iteration took 32.

On the card every call of the kernel has one shape: the last block of
starting columns is padded with zero columns, the pairs of every block to
the most any block has, and the k-points to a whole number of calls, and
the ELL width keeps the one last used for the same dimension when the
pattern loses a few entries, so that the kernel compiles once per
Hamiltonian. The diagonal moments are summed in the same kernel, so the
lagged Fermi level is free on the card as well. The doubling was left out
on the card, where its inner products would be summed in single precision.

One iteration against the previous step, on six desktop cores and on the
consumer card, the square island of 10,000 sites at `npol=100` and the
honeycomb islands of the GPU roadmap at `npol=200` (the card in single
precision; the exact search is the first iteration's, and with Nambu every
iteration's, included):

| case | orbitals | CPU | CPU before | card | card before |
|---|---|---|---|---|---|
| square, Hubbard | 20,000 | 76 s | 395 s | 44 s | 87 s |
| square, Nambu | 40,000 | 334 s | 1287 s | 232 s | 225 s |
| honeycomb, Hubbard | 1728 | 1.1 s | 6.2 s | 0.37 s | 0.82 s |
| honeycomb, Hubbard | 3072 | 3.5 s | 18.0 s | 1.5 s | 2.8 s |
| honeycomb, Nambu | 3456 | 5.7 s | 20.1 s | 1.8 s | 1.9 s |

So on the CPU an iteration is 5 times faster in the normal case, the
kernel and the lag together, and 4 times with Nambu, where the
electron-only search stays (60 s of the 334 s), and the CPU now overtakes
exact diagonalization already at 1728 orbitals (4.1 s) and at the
3456-orbital Nambu island by 6 times (35.6 s). On the card the normal case
is twice as fast, the trace now coming from the density-matrix recursion,
and the Nambu case is unchanged, as it should be, since it neither
lags nor doubles there. A first version of the padding padded a last
block of 220 starting columns to the 3236 of the first one, which made the
3456-orbital Nambu island 1.7 times slower on the card; the blocks and the
groups of k-points are now made equal, so the padding is at most one
column or k-point per call.

Two more pieces of the loop were changed with this step. The convergence
check (`diff_mf`) averaged the change of each matrix of the mean field over
all of its $N^2$ entries, most of them zero, so a change of 0.1 on every
diagonal entry of 20,000 orbitals read 1e-5 and passed the default
`maxerror`, and at $10^5$ sites a loop would have stopped after its first
iteration; a sparse mean field is now averaged over the entries it holds,
and a dense one as before. Measured per entry, the change of the mean
field in single precision stops at about 1e-6 (1e-6 to 3e-6 on the CPU,
where the doubled moments are inner products of single-precision
vectors, 8e-7 to 1e-6 on the card), which the diluted check hid, so a
`maxerror` below that needs `kpm_prec="double"`. On a 16-site Hubbard
island started from a uniform ferromagnet, the CPU run in single
precision then left the state the double-precision run converges to (a
net moment of 0.78, $E=-16.817$) for a less polarized one of lower energy
(0.10, $E=-16.832$ after 200 iterations), so the first is a saddle point
that the roundoff of single precision is enough to leave. And the trace of the exact search, the first iteration's and
every Nambu one's, goes through the same block kernel instead of
`kpm.full_trace`.

**The truncated recursion.** An opt-in radius keyword, with the starting
orbitals tiled in space so that a tile and its halo share one ball. With
truncation the lagged trace inherits the truncation error, which the
convergence in the radius already controls.

This step is built, in `kpmtk/truncation.py`, as the keyword
`kpm_radius` of both entry points, a whole number of hops on the site
graph, two sites joined whenever $H(k)$ has an entry between their
orbitals at some k-point (a decision of 6 October 2026, over a distance in
the units of the geometry): it is the light cone itself, so a spin flip or
a pairing on one site is not a hop and a bond across a periodic cell is
one. The sites of the starting orbitals are cut into tiles of at most 64
orbitals by halving at the median along the longest side, and every tile
runs on the Hamiltonian restricted to its region, the tile and the rows
of its pairs with every site within the radius, found by a breadth-first
search; the restriction keeps the spectrum inside that of $H$, so the scale
of the expansion stays valid. The tiles are independent problems for the
same kernels, so they go to the CPU one per thread (twice as fast as
splitting the rows of each tile among the threads, 7.3 s against 15.0 s at
10,000 sites on a laptop) and to the card as members of one batch, padded
to one shape per kind, so that the kernel compiles once whatever the sizes
of the regions. The density matrix, the lagged trace, the exact search
(the electron-only one of a Nambu loop included) and the band energy all
take the radius.

The error falls exponentially with the radius in a gapped state, as it
should. On a spinful honeycomb island of 200 sites with Rashba coupling
and a sublattice imbalance of 0.6, at half filling and `npol=150`, the
density matrix differs from the full one by 2.8e-2, 6.4e-3, 1.5e-3, 3.2e-4
and 1.9e-5 at 2, 4, 6, 8 and 12 hops, and the self-consistent exchange
field of a collinear antiferromagnet ($U=3$, 128 sites) by 0.11, 0.023,
0.005 and 0.001 at 2 to 8 hops. The Fermi level of the same kind of
island with 392 sites at filling 0.3, inside a band, differs by 1.3e-2 at
2 hops and 5e-4 at 12 at zero temperature, slowly, as in a metal, and by
8.5e-3 and 1e-6 at a temperature of 0.1; at half filling, inside the gap,
it moves by up to 0.36, which changes nothing, since every Fermi level in
the gap gives the same state.

At `kpm_radius=10` and `npol=100` on the square islands of the previous
table, the density matrix of an iteration and the exact search of the
first one take, with the peak memory of the process:

| case | orbitals | where | first search | iteration | memory |
|---|---|---|---|---|---|
| Hubbard, $10^4$ sites | 20,000 | CPU | 2.1 s | 2.5 s | 0.8 GB |
| | | card | 2.3 s | 2.3 s | 1.9 GB |
| Nambu, $10^4$ sites | 40,000 | CPU | 2.1 s | 10.6 s | 0.9 GB |
| | | card | 1.9 s | 7.8 s | 2.2 GB |
| Hubbard, $10^5$ sites | 199,712 | CPU | 19.5 s | 24.8 s | 0.9 GB |
| | | card | 19.3 s | 19.2 s | 2.1 GB |
| Nambu, $10^5$ sites | 399,424 | CPU | 19.5 s | 107 s | 1.5 GB |
| | | card | 19.4 s | 81 s | 2.8 GB |

(the Nambu iteration includes its electron-only search). So at $10^4$
sites an iteration is 30 times faster than the full recursion on the CPU
(2.5 s against 76 s, 10.6 s against 334 s), and from $10^4$ to $10^5$ sites
the time grows by 10 and the memory barely moves, which is what linear
means here: the memory is that of the tiles handed over at once
(`_TILES_PER_CALL`), since handing over every tile at once took 9 GB at
$4\times10^5$ orbitals. On the CPU the recursion is three quarters of a
call at $10^5$ sites, the rest finding the regions (4.4 s, kept from one
call to the next, since the site graph does not change along a loop) and
restricting the matrices to them (2 s). The card is only 1.3 to 1.4
times faster than the CPU here, against 3.4 for the full recursion; where
its time goes was not measured. Four self-consistent iterations at
$10^5$ sites, with the first search and the final energy, took 136 s on
the CPU and 116 s on the card for the Hubbard model (a peak of 1.2 GB on
the CPU), and 538 s and 399 s with Nambu (2.3 GB), the two backends giving
the same errors of the loop to 1e-8.

The doubling on the card is built too, with the same cost model as the
CPU, and only in double precision. In double precision a density matrix
takes 0.68 to 0.73 of the time it takes read on the rows (0.92 s against
1.34 s for a 1728-orbital island, 4.3 s against 6.3 s for a 3456-orbital
Nambu one, 94 s against 128 s for 20,000 orbitals). In single precision,
the default on the card, it was no faster (0.26 s against 0.31 s, 1.35 s
against 1.25 s, and 35 s against 22 s) and moved the density matrix by
5e-8 instead of 1e-8, since its inner products are summed in single
precision over every orbital, so there the blocks stay read on the rows.
The 22 s of the 20,000-orbital island read on the rows in single
precision (21 s again in a second run) is half the 44 s of the table of
the previous step, with the same kernel apart from the padding; the
difference was not traced.


## The checks

The sparse engine against the present dense KPM engine where both run, which
use the same moments and so agree to roundoff, as in
`tests/scf/test_kpm_block_density_matrix.py`: a Rashba Hubbard non-collinear
state, a $V_1$ Fock term, and a Nambu island with attractive $U$ and an
in-plane field. The numba CPU kernel against the jax kernel to roundoff. The
peak memory at $N$ and $4N$ within a factor of about 4.5, as an invariant of
linear scaling. For the truncation, the error against the full recursion
decreasing with the radius in a gapped state.

## Decisions taken, 6 October 2026

- Scope: the exact path first (the construction path, the sparse loop and
  the kernel), the truncated recursion after it as an opt-in keyword. Kept
  apart from it were "exact only", which leaves $10^5$ sites at hours per
  iteration, and "truncation built in from the start".
- Fermi level: the lagged exact trace. The alternatives were a stochastic
  trace with fixed random vectors, which shifts the filling by about
  $1/\sqrt{RN}$, and storing every moment of every needed pair, exact with
  no lag at $2\,$`npol` moments per pair, roughly 2 GB at $2\times10^5$
  orbitals in double precision.
- CPU kernel: numba on the CPU, jax on the card. This revisits `3854931`,
  which made the jax ELL kernel the engine on both backends.
- The dense engine: kept, with the sparse engine the default whenever the
  Hamiltonian given is sparse, and the dense one for a dense Hamiltonian.

Taken while the kernel and the truncation were built, the same day:

- The lag in both engines, so that they stay equal to roundoff at every
  iteration. Kept apart were the lag in the sparse engine only, which
  keeps the dense one on the path of exact diagonalization, and an opt-in
  keyword with an exact search by default, at about 1.8 times the
  density-matrix time per iteration.
- On the card, the padding to one shape and the trace in the same kernel,
  without the doubling.
- The convergence check of a sparse mean field averaged over its stored
  entries, the dense check unchanged. Kept apart were the stored entries
  everywhere, which moves the iteration counts of every dense loop, and
  leaving it with a `maxerror` that has to shrink with the size.
- With Nambu, the Fermi level of the electron-only Hamiltonian searched
  every iteration, the convention of the dense engine and of exact
  diagonalization. Kept apart was the number equation on the electron
  count of the BdG density matrix, free from the same recursion but a
  different state at the same filling.
- The truncation radius `kpm_radius` in hops on the site graph, the light
  cone itself. Kept apart was a distance in the units of the geometry,
  like `rcut`, which needs minimum images in a periodic supercell and
  whose exactness depends on the range of the hopping.

## Left open

- `scf.dm` of the sparse engine holds only the entries the loop read, as
  the dense KPM engine's does, and a Nambu one lacks the raw, unmapped
  entries the dense engine also computes and never reads.
- The per-site (array) filling of `VJinteraction` is still refused with
  `integration="kpm"`, sparse or dense.
- An a-posteriori error estimate for the truncation radius, beyond comparing
  two radii.
- With Nambu, the Fermi level of the electron-only Hamiltonian is still a
  trace of its own every iteration, truncated with the rest when a radius
  is given, a fifth of an iteration.
- The doubling on the card in single precision, its default, where it was
  no faster; a summation of the inner products in double precision would
  remove the precision loss but costs the card's float64.
