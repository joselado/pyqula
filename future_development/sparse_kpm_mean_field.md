# The KPM mean field for large sparse systems

Status, 6 October 2026: **the first step is built** (the construction path,
below), the KPM engine itself is not. This is the plan for a KPM
mean field (`h.get_mean_field_hamiltonian_kpm`, and `integration="kpm"` of
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

**The truncated recursion.** An opt-in radius keyword, with the starting
orbitals tiled in space so that a tile and its halo share one ball. With
truncation the lagged trace inherits the truncation error, which the
convergence in the radius already controls.

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

## Left open

- The memory constant of the sparse path, which has to be measured at
  $10^4$ to $10^5$ sites, not inferred.
- An a-posteriori error estimate for the truncation radius, beyond comparing
  two radii.
