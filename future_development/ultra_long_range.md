# The ultra long-range method in a tight-binding basis: a plan

This roadmap settles how the ultra long-range method, the ultracell, goes into pyqula. It
covers the formalism, what it buys in a tight-binding basis and what it does not, the
stages, the tests, and the decisions left to the maintainer. The method is that of
T. Müller, S. Sharma, E. K. U. Gross and J. K. Dewhurst, Phys. Rev. Lett. 125, 256402
(2020), arXiv:2008.12573, read from the TeX source. defuqumat has the plane-wave version
in its `ultracell` subpackage, which is where the design choices below were compared. The conventions were checked in a scratch prototype on 14 November 2026,
against the explicit supercell of pyqula. Stages 1 and 2 were built the same day (see
"What stage 1 built" and "What stage 2 built"); stages 3 and 4 are not. The next decision
point is the list at the end.

## What stage 1 built

`h.get_ultracell(hs,emax=None)` returns an `Ultracell` (`src/pyqula/ultracell.py`, with
the internals in `ultracelltk/`: `basis.py`, `matrix.py`, `observables.py`). It takes the
modulated supercell `hs` and projects it, `P^dagger H_S(k_S) P`, one column block at a time
with an FFT over the cells, so the clean part comes out as the band energies on the
diagonal without any subtraction. Its methods are `get_hk_gen`, `get_kdos_bands`,
`get_dos`, `get_ldos`, `get_vev` and `get_harmonics`, each with the normalization of its
counterpart on the unit cell or the supercell. The cut is `emax=`, every state up to an
energy, with a degenerate level kept whole. Diagonal supercells only at first (any
supercell matrix since 15 November 2026, see "Non-diagonal ultracells"); a supercell with a
site removed, a non-Hermitian Hamiltonian and `get_vev` with Nambu raise.

`tests/ultracell/test_ultracell.py` (17 tests) pins: the supercell spectrum with every
state, at 1e-12, for chains with odd and even $n$, spinless and spinful honeycomb with
Rashba coupling, an exchange field and modulated hoppings, a cubic lattice, and a Nambu
honeycomb with a modulated s-wave pairing; the unfolded weight against `operator="unfold"`;
the unit cell's `get_kdos_bands` and `get_dos` at zero modulation; the supercell's
`get_ldos` and `get_vev` (with and without `"sz"` and `"sx"`); the interlacing and the
nested bases of the cut; with a cut, the real-space states against the Rayleigh-Ritz
energies in the supercell, the LDOS summed over the sites against the DOS, the unfolded
spectral function against the kept levels of the unit cell, and the error of `get_vev`
when `emax` is below zero; the helical exchange against `generate_spin_spiral`; and the
harmonics of a cosine. The linear-response test against `densitychi_RPA` is not written,
and belongs with stage 2.

Two facts found while building it. The supercell builders center the positions, so a
function of position is shifted with respect to the cell index $\mathbf R$ that the
harmonics use: the phase of $f_{\mathbf Q}$ depends on that shift and its modulus does not
(stated in the docstring). And the cost is in the per-$\mathbf k_S$ work, not in the
unit-cell diagonalizations: the LDOS and the occupation of a $30\times30$ spinless
graphene ultracell, 1800 states, at four $\mathbf k_S$ each, take about a minute on two
threads, since `get_ldos` and `get_vev` each rebuild and diagonalize the matrix; a cache
of the eigenstates per $\mathbf k_S$ should halve it (an estimate, not measured), at
the price of storing them.

The guide page is `documentation/guide/ultracell.md`, with the figures
`ultracell_unfolding.py` and `ultracell_ldos.py`, the example
`examples/2d/ultracell/main.py`, and the notebook
`jupyter-notebooks/functionalities/single_particle_hamiltonians/14_ultracell_long_range.ipynb`.

## What stage 2 built

`u.get_mean_field_hamiltonian(...)` runs the mean field of `scftk` on the supercell with a
new density-matrix backend, `integration="ultracell"` (`ultracelltk/meanfield.py`). The
plan above named three changes; the code needed different ones. `VJinteraction` needed no
change: with an integration name other than `"ed"` or `"kpm"`, `h.get_mean_field_hamiltonian`
already routes a spinful Hamiltonian to `Vinteraction`, which takes U, V1, V2, V3 and Vr. What
`scftk/densitydensity.py` needed is three entry points, each a branch of the same shape as
the `"qtci"` and `"triangle"` ones: the registry of `get_dm`, the Fermi search for a filling
(from the ultracell eigenvalues, the states above `emax` counted as empty, and a
`ValueError` when the filling needs them), and, after the loop, the full density matrix
and the band energy. The folded basis travels on the supercell Hamiltonian as the
attribute `ultracell_basis`, which `copy()`, a deepcopy, keeps. The density matrix comes
out of `densitymatrix._accumulate_dm_batch`, the kernel of `"ed"`, fed with the ultracell
states written on the supercell orbitals, so it has the conventions of `"ed"` by
construction (the transpose of `full_dm` included).

Decision 5 was measured and then taken by the maintainer: the basis is rebuilt at every
iteration from the lattice-periodic part of the current Hamiltonian, the block of each
folded momentum on itself, with as many states per momentum as the clean cut keeps,
a degenerate level kept whole, and skipped without a cut (`rebuild_basis=True`, the default of the mean field; the result keeps it, so its
observables use the basis of the loop). The measurement, the antiferromagnet of graphene at
$U=3t$ under $0.3\cos(2\pi x_1/8)$, $8\times1$ cells, `nk=3`, moment about 0.47:

| emax | rows of 32 | moment error, clean basis | moment error, rebuilt basis |
|---|---|---|---|
| 0.5 | 16 | 0.41 | 0.004 |
| 1.0 | 18 | 0.37 | 0.002 |
| 1.5 | 22 | 0.29 | 0.002 |
| 2.0 | 26 | 0.08 | 0.0005 |
| 2.5 | 30 | 0.05 | 0.0003 |

The energy errors follow the same pattern (0.46 against 0.009 at `emax=0.5`, 0.15
against $10^{-5}$ at 2.5). The clean basis loses the antiferromagnet because the
staggered field mixes the two bands of graphene at every momentum, and the cut removes
part of the upper one. The whole rebuilt runs took 1.3 to 3.7 times longer than the clean
ones (7.0 s against 1.9 s at `emax=0.5`, 6.9 s against 5.5 s at 2.0), but the two
converged to different states in different numbers of iterations, so the cost per
iteration was not measured; the extra work was the $N$ projections of the orbital Bloch
sums, in a Python loop.

That loop is gone since 15 November 2026. The block of a folded momentum on itself, in
the orbital Bloch basis, is the Bloch Hamiltonian at $\mathbf p_{\mathbf q}$ of the
hoppings of the supercell averaged over its cells, the same for every $\mathbf k_S$. So
the rebuilt basis is the clean basis of that averaged unit cell, built once per
Hamiltonian, and it agrees with the projection to $5\times10^{-15}$ with modulated
hoppings, Rashba coupling and an exchange field. Two other costs went the same night.
The Fermi search and the density matrix diagonalized the ultracell twice per kpoint, and
the tail of the loop twice more; a uniform shift leaves the eigenvectors as they are, so
the search keeps its states and the rest takes them. And the density matrix sums each
pair once per kpoint for all directions, since only the Bloch phase depends on the
direction. On spinful graphene, $24\times24$ cells, 2304 orbitals, `emax=1.0` keeping
1440 states, `nk=1`, one call of the loop went from 30 s to 15 s, of which one `eigh` of
the ultracell is 4.7 s; on a $12\times1$ ultracell, fifteen iterations went from 2.5 s
to 1.0 s. What is left is the dense loop of `scftk`: the same call peaks at 3.4 GB,
against 0.7 GB for the ultracell itself, so a supercell of about 4000 orbitals no longer
fits in 6 GB, while its ultracell would still be a minute per iteration.

`tests/ultracell/test_ultracell_meanfield.py` (6 tests) pins: the Hubbard antiferromagnet
of a $3\times2$ cell with every state against `integration="ed"` from the same guess, energy
and Hamiltonian to $10^{-8}$, for both bases; a first-neighbor V1 on a chain, whose Fock
terms modulate the hoppings, against `"ed"`; the linear response to $10^{-4}\cos(2\pi
x_1/4)$ on a gapped honeycomb lattice against `chiAB_q` at $U=0$ and against
`densitychi_RPA` about the converged mean-field state at $U=1$, both on the same mesh, to
$10^{-3}$ relative (the measured agreement is at the printed digits, $10^{-10}$
absolute); the cut converging with `emax` with the rebuilt basis and the clean one losing
the moment; and the errors. The first comparison of the response was wrong on two
counts, both in the test and not in the code: the potential's harmonic carries the phase of
each site's position, so $V_{\mathbf Q}$ has to be taken from `get_harmonics` of the
potential; and at finite $U$ the response is the RPA about the mean-field state, whose gap
is smaller than the bare one, so its response is larger than the bare one even though
the Hartree kernel screens. A Hartree loop written by hand agreed with `scftk` to $10^{-10}$ on
that second point.

The guide has a section with a figure, and its example changed twice on the way, which is
worth recording. The first one, graphene at $U=2.5t$ under $0.6\cos(2\pi x_1/24)$ on the
mesh `nk=3`, showed the staggered moment reversing between antiphase domain walls where
the potential dopes the lattice. It does not survive a finer mesh: with `nk=[2,12]` there
is no reversal, with `[2,24]` a small one, and at $U=3t$, $V_0=1.5t$ the loop did not
converge on `[2,24]`. The ultracell is 24 cells long along $\mathbf a_1$ but one cell
along $\mathbf a_2$, so a region the potential dopes into a metal needs a fine mesh along
$\mathbf a_2$, and `nk=3` gave three points there. A charge density wave chain under a
slow potential converged in the mesh but into an irregular pattern in the doped regions,
which may be a metastable state of the loop. The example kept in the guide is a
paramagnetic insulator, the honeycomb lattice with a sublattice imbalance of $\pm1$, under
an exchange field $0.05\cos(2\pi x_1/24)\sigma_z$: $U=1.5t$ enhances the induced moment by
1.822 on `nk=[2,12]` and 1.821 on `[2,24]`, with the loop at `maxerror=1e-5` and `1e-9`
within $4\times10^{-5}$. A random initial guess left an in-plane moment of $1.8\times10^{-5}$
that the default tolerance did not remove, and gave 1.85 instead; the guess `ferroZ` gives
an in-plane moment of $10^{-16}$. To make the mesh per direction possible, the ultracell
observables now take `nk` as one number or one per direction, as `klist.kmesh` does.

Open after stage 2: the spin-spin exchange `J1`, `J2`, `J3`, `Jr` (refused with a
`NotImplementedError`; only `VJinteraction` has it, and it fixed
`integration="ed"` for its own loop); Nambu; the constraint
$\int\mathbf m_{\mathbf Q=0}=0$; and size, since the loop of `generic_densitydensity`
makes the supercell dense (`get_dense`), so its memory grows as $(N N_\mathrm{orb})^2$ per
direction whatever the cut.

## The spinful mean field on the sparse engine (15 November 2026)

The exchange and the size were solved together, on a decision taken by a Fable agent for
the maintainer. `VJinteraction` already had a sparse engine for the KPM
(`scftk/sparsemeanfield.py`): the interaction from a KD-tree, the density matrix at the
entries the mean field reads as csr matrices, the mean field sparse, Nambu included. Its
gate mixed two things, the sparse storage and the KPM numerics (the lagged Fermi level,
the band energy from the Chebyshev moments). It is now split: `sparse_engine` is the
storage, for `integration="kpm"` or `"ultracell"` on a sparse Hamiltonian, and the
ultracell supplies its own numerics with `ultracelltk.meanfield.SparseBackend`, which
diagonalizes the ultracell once per kpoint and returns the csr density matrix with the
Fermi level, as `full_dm_accumulate_sparse_with_fermi` does for "ed", and the band energy
from the same states. A spinful ultracell always takes this engine, so it has `J1`, `J2`,
`J3`, `Jr` and no dense matrix of the supercell in the loop. `VJinteraction` is spinful
only, so a spinless ultracell takes the sparse loop of `generic_densitydensity_kpm`, with
the same backend passed as `backend=`: it is there the `fermi=` object of that loop, with
the interface of `LaggedFermi`, whose Fermi level is exact and whose error is zero. The
dense route of stage 2, three branches of `scftk/densitydensity.py` and the states kept on
the Hamiltonian between the Fermi search and the density matrix, is gone.

Tests: the backend's entries against the dense density matrix of the same states to
$10^{-12}$ at $d$ and $-d$, with and without a cut, and with every state against "ed" on
the supercell to $10^{-8}$; $U$ and $J_1$ from a tilted guess against "ed" on the
supercell, energy to $10^{-8}$, with an in-plane moment above 0.05; the cut converging
with `emax` with $U$ and $J_1$. Two facts found on the way. The two engines stop at
different distances from the same fixed point, so a comparison at $10^{-8}$ needs
`maxerror` well below it: at $10^{-9}$ they differed by $8\times10^{-8}$ in energy, at
$10^{-12}$ by $8\times10^{-11}$. And a staggered moment tilted from $z$, with $U$ alone,
relaxes by about 0.5% per iteration in both engines alike, with errors equal to $10^{-14}$
over 300 iterations, so it is a slow mode of that state and not of the ultracell.

Two things are worth knowing about this route. The spinless loop goes through
`Vinteraction_kpm` and `densitydensity_kpm`, which are named after the KPM and now carry the
ultracell as well, a naming wart rather than a design choice. And a random guess is drawn
on the pairs the interaction couples only, so the spinless test against "ed" passes one
explicit guess to both. Nambu and the constraint are decisions 6 and 7 below.

The size it reaches, measured the same night on spinful graphene under $0.3\cos(2\pi
x_1/n)$, $U=3t$, $J_1=0.2t$, one call of the loop (one iteration and the energy) at
`nk=1`, on two threads:

| cells | orbitals | `emax` | kept states | dense loop before | sparse engine |
|---|---|---|---|---|---|
| 24x24 | 2304 | 1.0 | 1496 | 30 s, 3.4 GB | 7 s, 0.63 GB |
| 36x36 | 5184 | 0.5 | 2716 | | 24 s, 1.3 GB |
| 48x48 | 9216 | 0.3 | 4684 | | 94 s, 1.9 GB |

Two more changes went into the last two rows: the ultracell matrix is built on the kept
states, and not padded to the largest block, and the states go to the supercell orbitals a
chunk at a time. What is left is the dense `eigh` of the ultracell matrix, 19 of the 24 s
at 2716 states, at the BLAS limit of two threads (the evr, evd and numpy drivers take
16.6, 16.4 and 15.4 s on a random matrix of that size, and half of the eigenvectors 14.8
s). Graphene keeps about half of its states at any useful cut, since every occupied state
has to be there, so these rows are the worst case of the method; a unit cell with many
orbitals and a few bands near the Fermi level is where the cut pays. The eigenvectors of
every kpoint are kept for the density matrix, (kept states)$^2$ per kpoint, which is fine
on the meshes an ultracell needs, a few points for a large one. Going further needs another
route to the occupied states than the dense `eigh`, which is the question of stage 3; ARPACK
near the Fermi level helps the local density of states there, not the density matrix, which
needs every occupied state. The new wall is the time of the dense `eigh` of
the ultracell matrix, about 5000 kept states, and its eigenvectors, which are quadratic in
the kept states.

## Non-diagonal ultracells (15 November 2026)

A supercell along other vectors than those of the unit cell, $\mathbf A_S=M\mathbf A$ with
an integer matrix $M$, the $\sqrt3\times\sqrt3$ cell of a Kekule order or a commensurate
moiré cell, works since the night of 15 November 2026. The folded momenta are
$\mathbf p_{\mathbf q}=M^{-1}(\mathbf k_S+\mathbf q)$ with $\mathbf q\in\mathbb
Z^d/M\mathbb Z^d$, and the phase of a cell $\mathbf R$ is $\mathbf R\cdot
M^{-1}\mathbf q$. A diagonal form $PMQ=\mathrm{diag}(\mathbf n)$ with unimodular $P$
and $Q$ (`basis.diagonal_form`, a Euclid reduction of rows and columns, not the full
Smith form, since the FFT does not need $n_i$ to divide $n_{i+1}$) gives
$\mathbf R\cdot M^{-1}\mathbf q=(\mathbf RQ)\,\mathrm{diag}(1/\mathbf n)\,(P\mathbf
q)$, so the cells are a grid in $\mathbf RQ$ and the momenta a grid in $P\mathbf q$,
and every transform stays an FFT. The averaged hoppings take $\mathbf D M$ for the
displacement of the supercell $\mathbf D$, the unfolded spectral function takes
$\mathbf k_S=M\mathbf p$, and the harmonics come at $M^{-1}\mathbf q$. A diagonal $M$
gives $P=Q=1$ and the code of before.

Tests: the supercell spectrum with every state to $10^{-11}$, and the Rayleigh-Ritz
bound with a cut, for $\sqrt3\times\sqrt3$ and a seven-cell rotated honeycomb cell, a
seven-cell and a sheared six-cell triangular cell, all with $P$ and $Q$ not the identity,
and a chain; the unfolded spectral function against `operator="unfold"` and the
harmonics of a plane wave on the cells; the rebuilt basis against the block of each
momentum; the V1 mean field of the Kekule cell against "ed". The same night
`h.get_supercell(M)` learned any matrix (`htk/supercell.py`, `matrix_supercell`): every
hopping of the unit cell from the cell $\mathbf R$ to $\mathbf R+\mathbf d$ goes to the
copies in the cell $\mathbf R_j$ of the supercell with $\mathbf R+\mathbf d=\mathbf
R_j+\mathbf DM$, found with the same diagonal form, which is the inverse of the averaged
hoppings. Its test is the spectrum at a generic $\mathbf k_S$ against the unit cell at the
folded momenta, with Rashba coupling, an exchange field, Haldane hoppings with a
sublattice imbalance (without the imbalance, inversion maps $\mathbf k$ to $-\mathbf k$
and a reversed direction of every hopping went unnoticed), s-wave pairing, a chain and a
two-cell cubic supercell.

## The design of window= and of the Nambu mean field (15 November 2026)

The maintainer decided on 15 November 2026 to add `window=(emin,emax)` and a Nambu mean field
with the cut $|E|\le E_\mathrm{max}$ (decisions 2 and 6). A Fable agent designed both from the
code and from the TeX source of arXiv:2008.12573, and both were built the same day (ebaa136,
f58bd25, b7572fa, f3e0c08), with the tests of `tests/ultracell/test_ultracell_window.py` and
`test_ultracell_nambu.py`. A reviewer found one crash, a Nambu cut below the quasiparticle gap,
repaired in f3e0c08. The window and the Nambu cut are one
mechanism: each folded momentum gets an index range $[i_0,i_1)$ of its sorted levels, frozen
below $i_0$, kept in between, dropped above. The Nambu cut is the window $(-E_\mathrm{max},
E_\mathrm{max})$ on the BdG levels, and its deep states are the frozen density in the Nambu
space.

The Hellmann-Feynman test of the design came later the same day (f744b88,
`tests/ultracell/test_ultracell_hellmann_feynman.py`). At a fixed $\mu$ the slope of the
grand potential in the strength of the interaction is the band energy minus the grand
potential. It holds to $2\times10^{-10}$ for a modulated honeycomb with $V_1$ and a window on
the clean basis, and to $2\times10^{-7}$, the $\epsilon^2$ residue of the central difference,
for a modulated chain with $U=-2t$, a pairing of 0.36 and every state kept. Halving the band
energy of the frozen states breaks the first by 0.24, and doubling the pairing double counting
of the sparse engine breaks the second by 1.7. With a cut and the rebuilt basis the kept
states move with the interaction, so no such identity holds there, and the check is a table in
`emax`. `examples/1d/ultracell_superconductor` gives one, and `examples/2d/ultracell_window`
gives one for a window. On the chain with a slow potential of amplitude $0.4t$ (period 60,
$U=-2t$, filling 0.3) the cut overestimates the modulation of the pairing, and it converges to
the exact one from above as `emax` grows: 0.079 at `emax=1`, 0.047 at 2, 0.036 at 3, against
0.032 with every state. With every state kept, the ultracell and "ed" on the supercell agree to
$2\times10^{-9}$ at `maxerror=1e-11`. At the default tolerance they differ by $10^{-3}$, which
is the distance at which each loop stops.

Still not built: a shell filling for a gapless paired state.

The paper has no frozen density and no window: Eq. (2) sums every band at the central k, and
the runs keep the occupied states plus a number of empty ones, which is our `emax=`. In its
language the frozen states enter only $\rho_{\mathbf Q=0}$ of Eq. (5), so the frozen density
is an extension of ours.

- API: `h.get_ultracell(hs,emax=None,window=None,rebuild_basis=False)`, with `emax=x` the same
  as `window=(None,x)` and both at once a ValueError. For a normal state the window is absolute,
  in the energies of the clean cell, as `emax=` is now; for Nambu, `emax=x` means $|E|\le x$
  from the zero of the BdG Hamiltonian, and a Nambu window with `emin != -emax` raises, since
  the kept space would not be closed under particle-hole symmetry. The spectral routines give
  no weight outside the window; `get_vev` adds the frozen part.
- The frozen states and the kept ones must come from the same diagonalization, or the density
  matrix is not a projector, so the frozen density follows the basis: computed once on the
  clean basis, rebuilt at every iteration otherwise. It is the unit-cell density matrix of the
  frozen states tiled on the supercell, the inverse map of `FoldedBasis.periodic_hoppings`,
  $\mathrm{dm}_f[D]_{ij}=\frac{1}{N_kN}\sum_{k_S,q}e^{2\pi i\,\mathbf p_q\cdot
  \mathbf d_{ij}}\rho_0(\mathbf p_q)_{\alpha_i\alpha_j}$ with $\mathbf d_{ij}=\mathbf R_j-
  \mathbf R_i+DM$, summed on the folded momenta of the same mesh (`frozen_entries` in
  `meanfield.py`, added in `dm_entries` and `dm_full`). The Fermi search counts the frozen
  levels at the bottom and raises when the filling does not reach above them; the band energy
  adds $\mathrm{Tr}[P_f H_P]$ with $H_P$ the lattice-periodic part. The double counting reads
  the density matrix and needs no change.
- Nambu: the basis is the BdG states of the lattice-periodic part of the shifted Hamiltonian,
  not electrons and holes apart, because the gap equation sums over every state: with BdG
  states the frozen density carries an anomalous block, and a uniform superconductor is exact
  for any `emax`. A Nambu shift is not rigid, so the states are solved after the shift
  (`ultracell.dm`), and the ranges are recounted by energy at every iteration. Particle-hole
  symmetry maps $k_S$ to $-k_S$, and the mesh is closed under it, so the kept space is closed
  over the mesh. The Fermi level of the paired state is `superscf.BdGFermiLevel`, with an
  electron-sector search on the ultracell and a compressibility from the levels of the
  periodic part; the band energy is $(S+\mathrm{Tr}\,h_e)/2$. A Nambu mean field with a cut on
  the clean basis raises NotImplementedError, since a clean basis means nothing once $\mu$
  moves.
- Tests: `frozen_entries` against the explicit sum of the frozen basis vectors on the Kekule
  cell; a window against "ed" for two decoupled subsystems with the modulation on the upper
  one; the band energy against $\sum_D\sum_{ij}H_D[i,j]\,\mathrm{dm}[D][i,j]$; zero modulation
  with a uniform mean field against "ed"; a uniform attractive U at small `emax` against "ed",
  which must fail with the frozen anomalous block set to zero; a window holding every state
  equal to the supercell, normal and Nambu; particle-hole symmetry of a Nambu cut without
  inversion; nested windows never below "ed" and monotonic; Hellmann-Feynman in a fixed basis.
- Risks: the modulated part of the pairing converges only logarithmically in `emax`, since
  frozen states do not respond (about a quarter of the pair response at $E_\mathrm{max}=W/3$,
  $\Delta=W/100$), so a pair density wave with zero average needs a measured table in `emax`;
  the Nambu recount can change the kept dimension by a multiplet between iterations. Not built:
  a shell filling, Nambu `get_vev`, asymmetric Nambu windows, a response of the frozen states.

## The method, and what it becomes in a tight-binding basis

The question is how to compute a crystal whose potential, exchange field or order
parameter changes slowly over many unit cells, while the atoms stay periodic. The paper
generalizes the Bloch state with a sum over a fine grid of momenta $\boldsymbol\kappa$
around each $\mathbf k$,
$\Phi(\mathbf r)=\sum_{n\boldsymbol\kappa}c_{n,\mathbf k+\boldsymbol\kappa}\,
u_{n\mathbf k}(\mathbf r)\,e^{i(\mathbf k+\boldsymbol\kappa)\cdot\mathbf r}$, so that
the densities and the potentials become Fourier series
$f(\mathbf r)=\sum_{\mathbf Q}f_{\mathbf Q}(\mathbf r)\,e^{i\mathbf Q\cdot\mathbf r}$
with lattice-periodic coefficients. The smallest cell that holds the whole modulation is
the ultracell, $N=n_1n_2n_3$ unit cells.

In a tight-binding basis this is simpler than in DFT, and the simplest way to see it is
to start from the supercell. Take the ultracell as a supercell with Bloch momentum
$\mathbf k_S$, in reduced coordinates of the ultracell. Without the modulation, its
eigenstates are the Bloch states of the unit cell at the $N$ folded momenta
$\mathbf p_{\mathbf q}=(\mathbf k_S+\mathbf q)/\mathbf n$ (componentwise,
$\mathbf q\in\mathbb Z_{n_1}\times\mathbb Z_{n_2}\times\mathbb Z_{n_3}$), so Bloch's
theorem gives them from $N$ diagonalizations of the unit cell. The basis is

$$
|\mathbf q,m\rangle=\frac{1}{\sqrt N}\sum_{\mathbf R}e^{2\pi i\,\mathbf p_{\mathbf q}\cdot\mathbf R}
\sum_\alpha u_{m\alpha}(\mathbf p_{\mathbf q})\,|\mathbf R,\alpha\rangle ,
$$

with $u_m(\mathbf p)$ the eigenvectors of `h.get_hk_gen()` at $\mathbf p$, in pyqula's
lattice gauge (phase $e^{2\pi i\,\mathbf k\cdot\mathbf d}$). For an onsite modulation
$W(\mathbf R)$, a matrix on the orbitals of one cell that may include spin, the matrix to
diagonalize at $\mathbf k_S$ is

$$
H_{\mathbf q m,\mathbf q'm'}=\delta_{\mathbf q\mathbf q'}\delta_{mm'}\,\varepsilon_m(\mathbf p_{\mathbf q})
+u_m^\dagger(\mathbf p_{\mathbf q})\,W_{\mathbf q-\mathbf q'}\,u_{m'}(\mathbf p_{\mathbf q'}),
\qquad
W_{\mathbf s}=\frac1N\sum_{\mathbf R}e^{-2\pi i\,(\mathbf s/\mathbf n)\cdot\mathbf R}\,W(\mathbf R),
$$

meaning that the diagonal is the band structure of the unit cell and the modulation only
couples folded momenta that differ by one of its harmonics. For a general modulation
$\delta H$, hoppings included, the element is $P_{\mathbf q}^\dagger\,\delta H(\mathbf
k_S)\,P_{\mathbf q'}$, with $P_{\mathbf q}$ the folded Bloch states written on the orbitals
of the supercell. An eigenstate goes back to real space with one FFT over $\mathbf q$,
$\Phi(\mathbf R,\alpha)=N^{-1/2}\sum_{\mathbf q m}c_{\mathbf q m}\,e^{2\pi i\,\mathbf
p_{\mathbf q}\cdot\mathbf R}u_{m\alpha}(\mathbf p_{\mathbf q})$.

With every band kept this is the modulated supercell, exactly, rewritten in the eigenbasis
of the clean supercell. So the method has one approximation, the truncation of that basis
to the states of the unit cell in an energy window, and one convergence parameter, the
window. You can think of it as the modulated supercell solved by Rayleigh-Ritz in the
low-energy states of the clean supercell, which Bloch's theorem gives from the unit cell
alone.

The paper and Elk do three things that a tight-binding basis does not need. They keep one
set $u_{n\mathbf k}$ at the central $\mathbf k$ for every $\boldsymbol\kappa$, which the
paper chooses because it makes the densities and the matrix elements cheaper (defuqumat's
notes add that in LAPW a state at $\mathbf k+\boldsymbol\kappa$ cannot be handed to
$H(\mathbf k)$); they then build the diagonal block as $O^\dagger\varepsilon O$ with the
overlap $O$ made unitary by an SVD; and they assume $|\boldsymbol\kappa\cdot\mathbf
r|\ll1$ inside a cell. In a tight-binding basis $H(\mathbf p)$ acts on any vector, and the
band basis at every folded momentum costs one diagonalization of the unit cell. This is
the route defuqumat calls the direct route, the exact Rayleigh-Ritz matrix in the stated
basis. Its blocks depend on $\mathbf q$ and $\mathbf q'$ separately, so the build is $N^2$
products of size $N_b\times N_\mathrm{orb}\times N_b$, against $(N N_b)^3$ for the
diagonalization. The ratio of the two is $N_\mathrm{orb}/(N N_b)$, so the central-k basis,
whose blocks depend on $\mathbf q-\mathbf q'$ only, saves nothing as long as $N
N_b>N_\mathrm{orb}$. For a large unit cell, a moiré cell with $10^4$ orbitals, the
balance changes: the build dominates, and the direct route needs the $N_b$ states of the
window at each of the $N$ folded momenta (ARPACK, $N$ runs), where the central-k basis
needs them at one momentum, at the price of a slower convergence in the window. That
measurement decides the route of stage 3. The paper also uses a $\boldsymbol\kappa$
window of half the $\mathbf Q$ grid, which for an even ultracell never forms one
harmonic; we use the full folded set, and only with it is the method equal to the
supercell.

Three conventions were checked in the prototype and are part of the design. The momenta
are reduced, with the $2\pi$ in the phase, as in `geometrytk/bloch.py`. The modulation is
sampled at the positions of the supercell's own geometry and mapped to $(\mathbf R,\alpha)$
by `unfolding.get_supercell_map`, which is what `h.add_onsite(fun)` sees on the same
supercell, so a function written for a supercell gives the same physics here. The
unfolded weight at $\mathbf p=\mathbf p_0$ (take $\mathbf k_S=\mathbf n\,\mathbf p$) is
$\sum_m|c_{0m}|^2$, and `operator="unfold"` on the supercell gives $N$ times that.

## What was measured

The prototype is three short scripts outside the repository. The error is the largest
difference between eigenvalues of the ultracell matrix and of the dense supercell at the
same generic $\mathbf k_S$. The spinless rows use the harmonic form of the matrix above,
and the spinful row uses the general element $P_{\mathbf q}^\dagger\,\delta H\,P_{\mathbf
q'}$, with $\delta H$ the difference of the modulated and the clean supercell.

| model | orbitals per cell | ultracell | basis | dimension | error |
|---|---|---|---|---|---|
| spinful honeycomb, Rashba, modulated exchange field and modulated hoppings | 4 | 5x3 | every band | 60 | 1e-14 |
| same exchange field, general element against harmonic form, entry by entry | 4 | 5x3 | every band | 60 | 3e-16 |
| spinful chain, helical exchange field, supercell against `generate_spin_spiral` | 2 | 7 | every band | 14 | 1e-15 |
| honeycomb, three-harmonic potential | 2 | 7x1, 6x1, 5x4, 6x6 | every band | 14 to 72 | 1e-14 |
| same, unfolded weight against `operator="unfold"` | 2 | same | every band | 14 to 72 | 1e-13 |
| honeycomb, $0.3\cos$ potential | 2 | 31x1 | $\lvert\varepsilon\rvert<1.5$ | 40 of 62 | 9e-5 for states in $[-1,1]$ |
| triangular 4x4 moiré cell, slow $0.1$ potential | 16 | 9x9 | 1 band | 81 of 1296 | 0.23 |
| same | 16 | 9x9 | 2 bands | 162 of 1296 | 2e-4 for the lowest 81 |
| same | 16 | 9x9 | 6 bands | 486 of 1296 | 5e-6 |
| same | 16 | 9x9 | $\varepsilon<-1.7$ | 150 of 1296 | 3e-4 for states below $-1.9$ |

Two lessons come out of the moiré rows. A cut by band count fails without any warning when
the kept band overlaps the next one in energy: here band 1 spans $[-2.14,-1.72]$ and band 2
$[-1.95,-1.48]$, so one band cannot hold the lowest 81 states. A cut in energy with a margin
of about twice the amplitude of the modulation works, which is what perturbation theory
suggests, since states further away than that mix in only at second order.

## Where it pays off, and where it does not

The gain of the paper is the ratio of plane waves to bands, $10^2$ to $10^3$. In a
tight-binding model with two orbitals per cell there is no such ratio: the honeycomb
window above keeps 40 of 62 states for an error of $10^{-4}$, and a sparse supercell with
the KPM already reaches $10^5$ to $10^6$ sites. So for a few-orbital model the method
gives no speed. What it still gives is the unfolded spectral function at no extra cost
(each basis state has a definite primitive momentum), the harmonics $\rho_{\mathbf Q}$ and
$\mathbf m_{\mathbf Q}$ as direct outputs, and every eigenvector in a dense matrix, which is
what a mean field and an RPA need.

The method pays off when the unit cell has many orbitals and the physics lives in a few
bands: a moiré cell as the unit cell with a super-moiré modulation on top, a Wannier
Hamiltonian from DFT (defuqumat's states brought into pyqula) with a long-wavelength
magnetic texture, a multi-orbital d-electron model. There the dimension is divided by
$N_\mathrm{orb}/N_b$, a factor 8 in the moiré rows above, so about 500 in the cost of a
dense diagonalization. As in defuqumat, this is a prefactor and not a change of scaling:
the dense solve is $(N N_b)^3$ per $\mathbf k_S$.

## What it computes

- The unfolded spectral function $A(\mathbf p,\omega)$ along a path of the primitive zone,
  from the weight of each state on the block of momentum $\mathbf p$. This is what ARPES
  measures, and at zero modulation it must equal `h.get_kdos_bands()` of the unit cell.
- The density of states on a mesh of $\mathbf k_S$, per unit cell.
- The local density of states over the whole ultracell at an energy, one FFT per state,
  which is what STM sees above a domain wall or a gate-defined potential.
- The density and the magnetization per site, and their harmonics $\rho_{\mathbf Q}$ and
  $\mathbf m_{\mathbf Q}$, which are the order parameters of a charge or spin density wave.
- In stage 2, the self-consistent mean field and its total energy, which decides whether a
  modulated state is below the uniform one.

## Two ways of giving the modulation

This is the first decision, and it sets the architecture.

In the first way, you build the modulated supercell with the existing methods
(`hs = h.get_supercell(n)`, `hs.add_onsite(fun)`, `hs.add_exchange(fun)`, strain on the
hoppings, pairing), and the ultracell solves it in the window of folded Bloch states of
the clean `h`, one block of $\mathbf q'$ at a time with an FFT over $\mathbf R$. Every term
pyqula can put on a supercell works at once, and the test against the supercell is a
change of solver on the same object. The mean field can reuse `scftk` as a new
density-matrix backend, since its mean-field Hamiltonian is again a supercell Hamiltonian
(stage 2 lists what that takes). The price is that the sparse supercell is stored, so the
size is bounded by its memory, as for a KPM calculation on the same supercell.

In the second way, you give the onsite potential and the exchange field as functions of
position, and we keep only their harmonics $W_{\mathbf Q}$. The supercell is never built,
so this reaches the largest $N N_\mathrm{orb}$, but every new kind of modulation (a
hopping, a pairing, a bond mean field) needs its own harmonic form, and the mean field
needs its own loop.

The two give the same matrix for an onsite term, to 3e-16 in the prototype. The
recommendation is the projection first, and the harmonics later as the fast path for
onsite terms, checked against the projection.

## Stages

### Stage 1: the non-interacting ultracell

The basis (folded momenta, the states of the unit cell in the window, a degenerate level
at the edge of the window taken whole or not at all, as defuqumat does), the matrix at
$\mathbf k_S$, its eigenstates and their FFT to real space, and the observables of the
list above. Diagonal ultracells in one, two and three dimensions, spinless, spinful and
Nambu.

### Stage 2: the mean field

A local Hubbard $U$ first, collinear and non-collinear, then the intersite $V_1$ and $J$,
whose Fock terms are modulated bonds and which the projection handles with no extra work.
The mean field can reuse `scftk` through a new `integration="ultracell"` of
`scftk.densitydensity.get_dm`, which computes only the entries the mean field reads, as
the `qtci` backend does. This takes three changes, read in the code but not made: an
entry in the registry of `get_dm`; a way for the clean `h` and the window to travel on the
supercell Hamiltonian through `copy()`, since the backend receives only the mean-field
Hamiltonian; and the spinful route, `VJinteraction`, which calls `get_dm` with
`integration="ed"` written into the call, passing its own `integration` through. With
them, the interaction builders, the mixing, the energies and the double counting come from
the existing code. Two physics points belong to this stage.
First, the basis is the band structure of the clean `h`, so a mean field whose
lattice-periodic part changes (a moment that appears on every site) needs a wide window;
in a tight-binding basis we can instead move the $\mathbf Q=0$ part of the mean field into
the unit cell and rebuild the basis at each iteration, at the cost of $N$ small
diagonalizations. Second, in its runs on Cr and on $\gamma$-Fe the paper holds
$\int\mathbf m_{\mathbf Q=0}=0$, so that the loop is not drawn to a lattice-periodic
solution; `scftk/mfconstrains.py` is where such a constraint goes. The paper also warns of charge
sloshing from the $1/Q^2$ of a long-range Coulomb interaction, which is why the long-range
Hartree term is left to stage 4.

### Stage 3: size

For a modulation with a few harmonics the matrix is sparse in blocks.
The unfolded spectral function then follows from the KPM with the $N_b$ start vectors of the
block of momentum $\mathbf p$, deterministic, the same factored moments as the KPM
unfolding of `unfolding.md`, and the states near the Fermi level from ARPACK through
`algebra.arpack_eigh`. The harmonic route for onsite terms goes in here. The non-diagonal
ultracells planned here are built (see "Non-diagonal ultracells"); their folded set is
$\mathbb Z^d/M\mathbb Z^d$ in the column convention of $M^{-1}\mathbf k_S$.

### Stage 4: response

The static RPA needs almost nothing new: the self-consistent response
to a weak modulation is the RPA susceptibility at $\mathbf Q$ in the all-band limit. The
dynamical susceptibility of a modulated state, a matrix $\chi_{\mathbf Q\mathbf Q'}(\mathbf
q,\omega)$ built from the states in the window, and the long-range Coulomb Hartree term in
$\mathbf Q$ space, with the screened-to-bare ramp the paper proposes against charge
sloshing, each want their own design once stage 2 has been used on a real problem.

Batched dense diagonalizations over $\mathbf k_S$ are the GPU-favourable shape, but no GPU
work is planned here, since each tier of `documentation/gpu_porting_plan.md` needs the
maintainer's sign-off.

## Where the code goes

- `src/pyqula/ultracelltk/`, with `basis.py` (folded momenta, window, multiplets, the FFT of
  a state), `matrix.py` (the projection, later the harmonics), `observables.py` (spectral
  function, DOS, LDOS, densities and harmonics) and, in stage 2, `densitymatrix.py` (the
  backend of `get_dm`).
- `src/pyqula/ultracell.py`, the public entry that composes them, and a one-line delegator
  on `Hamiltonian`.
- `tests/ultracell/`, the invariants below.
- `examples/2d/ultracell_*/main.py`, the guide page `documentation/guide/ultracell.md` with
  its line in the `nav` of `mkdocs.yml`, an entry in `reference-*.md`, and a bullet in the
  FUNCTIONALITIES list of `README.md`.

## Tests that can fail

- Every band kept gives the eigenvalues of the supercell at a generic $\mathbf k_S$, with a
  generic modulation, for odd and even ultracells in one, two and three dimensions,
  spinless, spinful with an exchange field and with Rashba coupling, and Nambu with a
  modulated pairing. Measured at 1e-14 in the prototype, spinless in the harmonic form
  and spinful with Rashba coupling, an exchange field and modulated hoppings in the
  general one.
- The unfolded weight equals that of `operator="unfold"` on the supercell, up to the
  factor $N$. Measured at 1e-13.
- At zero modulation the unfolded spectral function equals `h.get_kdos_bands()` of the
  unit cell at every $\mathbf p$, for any window that holds the energies.
- With a window of the form $\varepsilon<E_\mathrm{max}$ the $i$-th lowest eigenvalue is
  never below the $i$-th lowest of the supercell (Cauchy interlacing), and it decreases as
  $E_\mathrm{max}$ grows. A wrong phase or a wrong block breaks this.
- A helical exchange field of period $n$ against `h.generate_spin_spiral(qspiral=1/n)`:
  the union of the spiral bands at the folded momenta shifted by $\mathbf Q/2$ equals the
  ultracell spectrum. Measured on the supercell of a spinful chain ($n=7$) at 1e-15, and
  0.4 off without the half step, so the shift is part of the test. This pins the algebra
  of $\mathbf B_{\mathbf Q}\cdot\boldsymbol\sigma$.
- A weak potential $V_{\mathbf Q}$ induces $\delta n_{\mathbf Q}=\chi_0(\mathbf Q)V_{\mathbf
  Q}$, against `chitk.densitychi.densitychi_RPA` at $q=Q$ and zero frequency, with `U=0`
  for the bare response, and in stage 2 with the same $U$ as the mean field.
- In stage 2, the all-band mean field equals the supercell mean field from the same
  initial guess, and a stable state does not depend on the random initial guess.

## Decisions for the maintainer

Each comes with a recommendation; they are handed over one at a time.

1. How the modulation is given: the projection of a supercell Hamiltonian, the harmonics of
   functions, or both. Decided by the maintainer on 14 November 2026: both, the projection
   in stage 1 and the harmonics in stage 3 as the fast path for onsite terms.
2. How the basis is cut: `nbands=` counted from the bottom, as Elk and defuqumat do;
   `emax=`, every state below an energy, which keeps the interlacing bound and needs no
   frozen density; or `window=(emin,emax)`, which pays off for a wide multi-orbital model
   but makes the states below `emin` a frozen, lattice-periodic density that does not
   respond to the modulation. Stage 1 took the recommended default, `emax=`, with the
   maintainer's go-ahead to use the recommendations and report them. Decided by the
   maintainer on 15 November 2026: `window=(emin,emax)` is added, with the states below
   `emin` frozen as a lattice-periodic density.
3. The normalization of the unfolded spectral function: per unit cell, so that zero
   modulation gives the unit cell's own `get_kdos_bands()`, or the convention of
   `operator="unfold"`, $N$ times larger. Stage 1 took per unit cell, the recommended
   default; it is a factor in `observables.kdos_bands` if the maintainer picks the other.
4. The public name: `h.get_ultracell(hs,emax=None)`, as in defuqumat, taken in stage 1 as
   the recommended default.
5. Whether the mean field rebuilds the basis with the $\mathbf Q=0$ part at every
   iteration, or keeps the basis of the clean `h` frozen as Elk does. Measured in stage 2
   (see "What stage 2 built") and decided by the maintainer on 14 November 2026: rebuilt by
   default, with the clean basis as an option.
6. The basis of a Nambu mean field with a cut. With $E\le E_\mathrm{max}$ on the BdG
   states every negative level is kept, at least half of all the states, so the cut never
   buys more than a factor 2, and the kept space is not closed under particle-hole
   symmetry. Only $|E|\le E_\mathrm{max}$ scales, and it needs a frozen density for the
   deep states, which is the `window=` question of decision 2. Without a cut the
   ultracell is the supercell in another basis and buys nothing over "ed". Recommendation
   of the Fable agent on 15 November 2026: leave Nambu until decision 2 is taken. The sparse
   engine already takes Nambu, so what would remain is the Fermi level of the paired state
   (`superscf.BdGFermiLevel`) and the band energy of a BdG Hamiltonian. Decided by the
   maintainer on 15 November 2026, together with `window=`: the ultracell mean field takes
   Nambu Hamiltonians, with the cut $|E|\le E_\mathrm{max}$.
7. The constraint $\int\mathbf m_{\mathbf Q=0}=0$ of the paper. It is a zero net moment
   and not specific to the ultracell. The recommended form is a named constraint of
   `scftk/mfconstrains.py`, for instance `"no_net_moment"` in `constrains=`, which removes
   the site average of the spin part of the onsite mean field; `_set_entries` takes csr,
   so it works on the sparse engine. It removes the uniform exchange field, not the
   uniform moment itself, and the two agree only for a uniform $U$ and in linear response.
   The exact form is a Lagrange field co-converged with the loop, one more unknown of the
   kind that caused trouble in the Kondo-lattice loop. No case is known yet where the loop
   falls to the ferromagnet without it, so there is no test that could fail. Decided by the
   maintainer on 15 November 2026: the named constraint, `constrains=["no_net_moment"]`.
