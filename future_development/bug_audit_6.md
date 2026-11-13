# Bug audit 6: the leftovers of the fifth sweep, 2026-11-13

The sixth sweep took the list that the fifth one left for the next sweep and
split it into three areas: the KPM mean field and the density-matrix routes,
the operators and the real-space observables, and the embedding with the
screening of the BSE package. One agent per area ran each item against an
oracle that shares no code with what it checks, and a second agent per area
tried to refute every finding, by re-running it and by reading the code,
before fixing the ones that are clear repairs. Every regression test was run
on a `git archive` export of `a108800`, where it fails, and on the fixed tree,
where it passes. Every run was on the CPU, on two local processes at most. The
reproduction scripts lived in a session scratchpad and are not kept here; each
entry says what was run and what it printed.

## Status

**Thirteen findings, all closed.** Eight were repairs, each with a
regression test that asserts an invariant (two routes agreeing, a
Hermiticity, a sum rule) and fails on the unfixed source. Five changed a
convention, a keyword or a default, so they went to the maintainer with the
options and what each buys, and were decided and fixed the same day (#9 to
#13), with tests of the same kind. The touched test folders pass on the
combined tree; the whole suite was not run.

## User-visible changes

- At a finite temperature, `get_fermi4filling_kpm` and the KPM mean fields
  (`Vinteraction_kpm`, `hubbard_kpm`, `VJinteraction` with
  `integration="kpm"`) now put the Fermi level at the requested filling, and
  it improves as `npol` grows. Before, the error grew with `npol`, about
  1e-3 electrons at `npol=1600` and $T=0.3$ on a small ring, and the loop
  converged to that wrong filling. At the default $T=10^{-7}$ nothing moves.
- The direct density-matrix routes (`full_dm_accumulate`,
  `full_dm_accumulate_sparse`, `full_dm_accumulate_sparse_with_fermi`,
  `full_dm_simultaneous`) with `delta=0` give the $T=0$ limit, as `full_dm`
  does, instead of NaN for a level exactly at the Fermi energy.
- `h.get_vev`, `h.get_single_vev` and `h.get_several_vev` of an operator
  without a matrix (valley, valley_x and valley_y on a periodic Hamiltonian,
  velocity, current, the unfolding projector, any `Operator` built from a
  function) are read at `h.temperature`, as the matrix route is. A
  Hamiltonian that is not a mean-field output gives the same numbers as
  before.
- `spectrum.ev2d` on a BdG Hamiltonian gives the electrons only, the same as
  the normal state at zero pairing, instead of $S(k)+S(-k)$; `ev2d` and
  `selected_bands2d` accept an operator by name and apply a k-dependent
  operator at the k-point of the state.
- `get_vev("current")` and `get_vev("velocity")` run on a dense 1D
  Hamiltonian (after `add_rashba`, for instance), with the numbers of its
  sparse copy; the 1D operator returns an array of the shape of the vector.
- `Embedding(h, selfenergy=...).get_density_matrix()` is Hermitian and equal
  to the real-axis integral (end occupations of a 6-site chain from 0.603,
  with an imaginary part of 0.27, to 0.508). A direct `get_gf` with a
  negative `delta` now returns the advanced Green's function.
- `embed_hamiltonian(h, selfenergy=matrix)` uses the Hermitian conjugate of
  the self-energy on the lower contour. A matrix $-i\Gamma$ with real
  $\Gamma$ is unchanged; a self-energy with a complex Hermitian part (a lead
  with spin-orbit coupling or a flux) now gives a Hermitian density matrix.
- `ScreenedInteraction.get_dict()` on an even mesh gains the vectors at
  $+n_k/2$, and the weight of the zone boundary is split in half between
  $-n_k/2$ and $+n_k/2$ (a quarter at a 2D corner), with `tol` acting on the
  halved value. On the mesh nothing moves; off the mesh $W(q)$ is now
  Hermitian. Odd meshes do not change.

- A `scale` given to the KPM density matrix, the KPM Fermi search, the KPM
  energy or the KPM Chern marker must exceed the spectral radius of $H(k)$
  by more than 1%, or a `ValueError` is raised; before, a scale in that
  margin lost the band edge without an error. The default `scale=None` is
  unchanged.
- `cores=` of the KPM density matrix, Fermi search and mean fields is the
  number of numba threads of the recursion during the call; it no longer
  forks a process pool every iteration or changes `parallel.cores`.
- The 1D velocity and current operators (and with them the spin current,
  the current-resolved bands and the ground-state current) are the Cartesian
  band speed along $a_1$, in units of $ta/\hbar$ as in 2D and 3D: every 1D
  number moves by $|a_1|$, which is 1 for a chain of lattice constant 1 and
  $\sqrt3$ for a zigzag ribbon of first-neighbor distance 1.
- `Embedding(h, m=..., selfenergy=...)` raises a `ValueError`; it used to
  drop the defect.
- A callable selfenergy of `embed_hamiltonian` is the retarded one, and the
  routine continues it below the real axis as $\Sigma(z^*)^\dagger$, as the
  boundary embedding does; a callable that used to read the sign of
  `delta` itself is now only called with a positive one. Both embeddings
  take `delta=0` as retarded (a matrix selfenergy used to take it as
  advanced). The semimetal selfenergy gives the right LDOS, and the density
  matrix of either embedding refuses it with a `NotImplementedError`, since
  it is not analytic off the real axis.

## Fixed

### 1. The finite-temperature Fermi count of the KPM engines aliases a large npol

**Cause.** `_filling_count` (`kpmtk/densitymatrix_kpm.py`) convolved the
cumulative KPM density of states with $-\partial f/\partial E$ on a fixed
grid of 801 points over $\pm 40T$, a spacing of $0.1T$. A large `npol`
resolves a discrete level into a step narrower than that, and the trapezoid
aliases it. `LaggedFermi` holds the loop to the same count, so the loop
converged to the wrong filling without a warning.

**Oracle.** The Fermi-Dirac count on the exact levels, and
`spectrum.get_fermi_energy_T`. On a 12-orbital ring at $T=0.3$ the Fermi
level was off by 9.6e-6, 2.4e-4, 6.3e-4 and -5.3e-4 at `npol` 200, 800, 1600
and 3200, growing instead of closing. `Vinteraction_kpm` on a six-site ring
at $T=0.3$, `npol=800`, converged to 5.9939 electrons instead of 6.

**Fix.** The count is now the closed-form convolution of the piecewise-linear
cumulative DOS with the Fermi function, through `logaddexp`, so there is no
grid. The ring then converges to 6.000001 electrons.
Test: `tests/scf/test_kpm_fermi_count_large_npol.py`.

### 2. The direct density-matrix routes return NaN at delta=0

**Cause.** The numba kernels evaluate $1/(1+e^{\epsilon/\delta})$, which is
0/0 for a level exactly at the Fermi energy when $\delta=0$, and the four
direct routes did not carry the $\delta=0\to 10^{-15}$ guard of `full_dm`.
Every caller inside the package applied the guard itself, so only a direct
call was exposed.

**Fix.** One helper, `densitymatrix.nonzero_smearing`, used by `full_dm` and
by the four routes. Test: `tests/densitymatrix/test_zero_smearing.py`.

### 3. get_vev of an operator with no matrix ignores the temperature of a mean-field Hamiltonian

**Cause.** `kresolved_orbital_vev` (`vev.py`) fell back to `delta_dm` when no
`T` or `delta` was given, while `full_dm` falls back to
`default_smearing(h)`, which reads `h.temperature`. The commit that moved
`full_dm` and `get_filling` to `h.temperature` left this route behind.

**Oracle.** A Fermi-Dirac sum over the eigenvectors by hand. On a honeycomb
metal with `h.temperature=0.3`, `get_single_vev("sy")` gave -0.25745 through
the matrix and -0.19059 through the same operator given by its action, the
second being the $T=10^{-6}$ value.

**Fix.** The default smearing is `default_smearing(h)`.
Test: `tests/vev/test_matrixless_vev_temperature.py`.

### 4. spectrum.ev2d counts twice on a Nambu Hamiltonian and refuses operator names

**Cause.** `ev2d` summed `op.braket` over every negative BdG state without
`operators.vev_operator`, the shape of finding #2 of the fifth sweep, which
repaired `get_single_vev` and its siblings but not `ev2d`. Separately,
`operator2list` wraps a string in a list without `h.get_operator`, so both
`ev2d` and `selected_bands2d` called `.braket` on a string, and a
k-dependent operator was applied at `k=None`.

**Oracle.** The occupied states of the normal $H(k)$ by hand: $s_z=-0.014295$
against -0.02859 on the BdG copy at zero pairing.

**Fix.** Both routines resolve the operator with `h.get_operator`, `ev2d`
passes it through `vev_operator`, and both apply it at the k-point of the
state. Test: `tests/spectrum/test_ev2d_nambu_and_names.py`.

### 5. get_vev("current") crashes on a dense 1D Hamiltonian

**Cause.** The 1D current operator inherits the type of `h.intra`, an
`np.matrix` after `add_rashba`, so it returned a (1,n) matrix, and the
elementwise product in `vev.py` became a matrix product.

**Fix.** The 1D operator returns `np.asarray(...)` of the shape of the
vector. `Operator.braket` goes through `braket_ww` with the same shapes, so
`get_bands` gives the same numbers as before.
Test: `tests/operators/test_current_dense_1d.py`.

### 6. The boundary embedding used the retarded self-energy on the lower contour

**Cause.** `boundary_embedding_gf` (`embeddingtk/boundaryembedding.py`) built
$\Sigma=-i\Gamma$ for every `delta`, and the density matrix evaluates the
lower contour with a negative `delta`, where the advanced function needs
$\Sigma^\dagger=+i\Gamma$. The poles of the lower function then sat inside
the contour.

**Oracle.** The real-axis integral of $i(G^r-G^{r\dagger})/2\pi$ on a fine
Gauss-Legendre grid, validated to 2e-7 on a real symmetric self-energy. On a
6-site chain the largest entry error was 0.28.

**Fix.** For a negative `delta` the routine calls itself at the conjugate
energy with $-\delta$ and returns the Hermitian conjugate.
Test: `tests/embedding/test_embedding_selfenergy_density_matrix.py`.

### 7. Embedded_Hamiltonian took the elementwise conjugate of a matrix self-energy

**Cause.** `object2selfenergy` (`embeddingtk/embedded.py`) returned
`np.conjugate(self)` for $\delta\le 0$; the advanced self-energy is
$\Sigma^\dagger$, and the two differ whenever the Hermitian part of $\Sigma$
is complex. With a complex Hermitian part of 0.4 on a 4-site island the
density matrix was off by 0.093 and non-Hermitian by 0.105.

**Fix.** `self.conj().T`. Same test file as #6.

### 8. ScreenedInteraction.get_dict on an even mesh is not Hermitian off the mesh

**Cause.** `_mesh_vectors` (`bsetk/screening.py`) runs over
`range(-(n//2), n-(n//2))`, which for even $n$ holds $-n/2$ and not $+n/2$,
so the weight of the zone boundary sat on one side only. At $n_k=4$ on a
two-site chain $W(q=0.137)$ was non-Hermitian by 0.137; at $n_k=5$ by 1e-16.

**Fix.** Each mesh vector is accumulated over its boundary images with equal
weights. Test: `tests/bse/test_bse_screening_dict_hermitian.py`.

### 9. A tight KPM scale clipped the band edge

**Cause.** `_kpm_pair_values` integrates the occupied window from
$-0.99\,s$, where the Chebyshev weight becomes singular, so the weight
between $-s$ and $-0.99\,s$ never entered a density-matrix entry, while the
scale guard accepted any $s$ at or above the spectral radius (#11 of the
fifth sweep). On a chain with spectrum $[-2,2]$ at `scale=2.0` the onsite
occupation was 0.4547 instead of 0.5 and the bond 0.273 instead of $1/\pi$,
with no error; the Fermi search renormalizes its own count, so the two
disagreed. The default `scale=None` keeps the spectrum inside $0.91\,s$ and
was never affected.

**Decision and fix.** Of the four options (integrate in $\theta$ over the
whole interval, pad a given scale, refuse it, document it) the maintainer
chose on 13 November 2026 to refuse it. `_check_scale_margin`
(`kpmtk/densitymatrix_kpm.py`) runs in the density matrix and in the trace
behind the Fermi search and the energy, only for a scale given by the
caller: the Gershgorin bound of $H(k)/s$ settles almost every case at no
cost, and only when it reaches $0.99$ is the spectral radius found exactly,
so a valid scale between the radius and the bound is still accepted. It
runs after the check on the moments, so a scale below the spectrum keeps its
own message. The window edge is one constant, `_WINDOW`. Tests:
`tests/scf/test_kpm_scale_guard.py` (2.0 and 2.01 refused in the density
matrix, 2.0 in the Fermi search, 2.05 equal to exact diagonalization); the
three refusals fail on the unfixed source.

### 10. cores= of the KPM engines re-forked a pool every iteration and never used it

**Cause.** `_kpm_pair_values` and `_kpm_trace` called
`parallel.set_cores(cores)` on every call, so a mean field with `cores=4`
created 121 pools in 118 iterations and mapped no task to them, since the
recursion runs on numba threads, with a bit-identical result, and left
`parallel.cores=4` after it returned, while the guide and the
`VJinteraction` docstring promised parallel workers.

**Decision and fix.** Of the three options (fork only when the count
changes, refuse the keyword in the KPM route, make it a thread count) the
maintainer chose on 13 November 2026 to make it the thread count.
`_numba_threads` (`kpmtk/densitymatrix_kpm.py`) sets the numba threads of
the recursion for the duration of the call and restores the previous count
on exit; it starts no pool, leaves `parallel.cores` alone, is clamped to 1
by `parallel.set_enabled(False)` and capped at the threads numba started
with. Test: `tests/scf/test_kpm_cores_threads.py`, which reads the thread
count inside the recursion (14 instead of 1 on the unfixed source), checks
that both counts are restored and that the result does not change.

### 11. The 1D velocity was per unit cell, not Cartesian

**Cause.** The 1D branch of `get_velocity` (and `get_current`, its alias)
was `current.current_operator`, the derivative per cell $dE/d\theta$, so the
same state of a chain gave a speed of 1.1756 with $|a_1|=1$ and 0.5878 with
$|a_1|=2$, while the 2D and 3D branches are the Cartesian band speed in
units of $ta/\hbar$. The sign and the $2\pi$ were right; only $|a_1|$ was
missing.

**Decision and fix.** Of the three options (both names Cartesian, only
`velocity` Cartesian, document the per-cell convention) the maintainer chose
on 13 November 2026 to make both Cartesian. `current.current_operator`
multiplies the derivative per cell by $|a_1|$ in 1D, so the velocity, the
current, the spin current, the current-resolved bands and the ground-state
current move together; `current.derivative`, which the effective mass uses,
is unchanged. Test: `tests/operators/test_velocity_1d_cartesian.py`,
$|dE/dk|=\sqrt{4-E^2}$ for every state of a chain in cells of 1, 2 and 3
sites, for both names; the supercells fail on the unfixed source.

### 12. Embedding(h, m=defect, selfenergy=...) ignored the defect

**Cause.** With `selfenergy=` the boundary embedding is built from `h`
alone, and a defect passed as `m` was dropped without a word.

**Decision and fix.** Of the three options (honour `m`, refuse the
combination, document it) the maintainer chose on 13 November 2026 to
refuse it: the `Embedding` constructor raises a `ValueError` when both are
given, saying to put the defect into `h`. Test: the last refusal test of
`tests/embedding/test_embedding_selfenergy_density_matrix.py`, which fails
on the unfixed source.

### 13. The semimetal self-energy ignored delta, and is not analytic

**Cause.** `get_selfenergy_from_potential(mode="semimetal")` returned the
retarded $-i\Gamma|E|$ for both signs of `delta`, so the spectral function
$i(G^R-G^A)/2\pi$ of `Embedded_Hamiltonian.get_ldos` came out zero. Behind
it, the callables of `Embedded_Hamiltonian` and of the boundary embedding
followed two contracts: the first was handed the sign of `delta` and had to
deal with it, the second was continued below the real axis by the routine.
And $|E|$ has no analytic continuation off the real axis, while the density
matrix of both embeddings is a contour integral over complex energies: on a
honeycomb flake with a Haldane coupling, an exchange field and Rashba
coupling the contour gave a trace of 17.29 against 18.34 from the real-axis
integral, with entries off by 0.30.

**Decisions and fix.** The maintainer chose on 13 November 2026 to fix the
helper, asking that it stay right without time-reversal symmetry, and then,
for consistency, one contract for both embeddings. A matrix or a callable
selfenergy is the retarded one, and for a negative `delta` the advanced one
is its continuation $\Sigma(z)=\Sigma(z^*)^\dagger$, the Hermitian
conjugate, which holds whatever the symmetries of the Hamiltonian
(`object2selfenergy` in `embeddingtk/embedded.py` and
`boundary_embedding_gf`, now with the same convention at `delta=0`,
retarded). A callable has to be analytic in the upper half plane, as a
causal selfenergy is; the semimetal helper is marked `analytic=False`, and
the density matrix of both embeddings refuses such a selfenergy with a
`NotImplementedError`, while its LDOS and Green's function at real energies
are right. Tests in the same file: on the flake without time reversal
$\mathrm{Tr}\,A=-\mathrm{Im}\,\mathrm{Tr}\,G^R/\pi$ exactly and $A$
Hermitian; an analytic callable $\Gamma\,0.8/(z-0.5+0.6i)$ in
`embed_hamiltonian` gives the real-axis density matrix; the semimetal and a
non-analytic callable of the boundary embedding are refused. All of them
fail on the unfixed source.

Closed with the record: the error messages of `bsetk/factorize.py`,
`bsetk/qtt.py` and `bsetk/solve.py` told the user to call
`get_dict(cutoff=...)`, which raises a `TypeError`, since the keyword is
`tol`; they now say `tol`.

## Chased and cleared

- The anti-Hermitian growth of #5 of the fifth sweep with a Hubbard $U$ alone
  on a 2D lattice.
- `hubbard_kpm` called directly, dense and sparse, and
  `constrains=["no_charge"]` through `hubbard_kpm` and `Vinteraction_kpm`.
- A bond pairing through the KPM density matrix, one map step.
- k-meshes with `nsuper` other than 1.
- The sparse density-matrix routes with a finite smearing and a level at the
  Fermi energy.
- The finite-temperature total energy of the KPM engines adds a $T=0$ band
  energy to a finite-temperature density matrix; no docstring or guide page
  claims a free energy, so it is a convention, not a finding.
- `real_space_chern` on spinful and Nambu flakes and with `operator=`.
- valley, valley_x and valley_y through `get_vev`.
- `kresolved_orbital_vev` against `full_dm` at an explicit temperature.
- The ring entanglement entropies of periodic 1D and 2D Hamiltonians and the
  Li-Haldane spectrum.
- `selected_bands2d` on a Nambu Hamiltonian, and the sign of the 1D current
  with its equilibrium sum.
- `ScreenedInteraction.get_dict` on the mesh and on an odd mesh, and the
  embedded density matrix with a real symmetric self-energy.

## Left for the next sweep

- `real_space_chern` on a Chern superconductor (Rashba 0.3, exchange 0.6,
  onsite 0.3, pairing 0.3): the bulk marker of a small island was 0.247
  against $-2$ in k-space. The bulk gap was not computed and no larger island
  was run, so it is unexplained; a second parameter set (gap 0.25) gave $-2$.
- `real_space_chern(operator=)` on a Nambu flake, and with an operator that
  does not commute with $H$, against `topology.chern(operator=)`.
- The current and velocity through `get_vev` in 2D and 3D, and the Nambu ring
  entropies.
- The consumers of `get_dict` on an even mesh after #8,
  `get_mean_field_hamiltonian(V=W.get_dict())` and the exciton bands at $Q$
  off the mesh.
- The fine grid of #1 on a k-mesh and through `VJinteraction`,
  `constrains=["no_magnetism"]` and its in-plane variants (the test case
  did not converge in any engine), a converged paired KPM loop, and the jax
  density matrix in single precision.
- `examples/1d/current_haldane`, `current_antihaldane` and `ribbon_velocity`
  print 1D currents and velocities, now larger by $|a_1|$ after #11; their
  comments were not checked against the new numbers. The folders
  `tests/parallel`, `tests/island`, `tests/superconductivity`,
  `tests/spinon`, `tests/kondolattice` and `tests/magnon`, the most exposed
  to #3, #10 and the new `mode=` of `real_space_chern`, were not run on the
  final tree.
- Everything of the embedding and BSE area beyond #6 to #8: the embedding
  with `nsuper>1`, in 2D or with Nambu, cRPA against an oracle, `nkW>nk`,
  the exciton amplitudes, `exciton_bands` with `cores>1` and in 3D,
  `metal=True` magnons, the transverse rung of `Vchannels` against the ring
  oracle, the quantics solver on a spinful 2D model, the iterative solver near
  a degenerate cluster, `gauge="projection"` with screening, and the GPU
  screening path. The workstation for the heavy runs was not reachable, so
  these were not run.
