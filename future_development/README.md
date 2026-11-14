# future_development

Maintainer-facing roadmaps for work that is planned, partially done, or
scoped-but-not-started. These are notes to a future implementer (human or
Claude), not user documentation -- user-facing behaviour belongs in the
user guide (the website pages under `documentation/guide/`) and `README.md`.

A document here should say what exists today, what is missing, what was
already measured or ruled out, and what the next decision point is, so that
picking the work up again does not mean re-deriving conclusions that were
already reached once.

- [`bug_audit.md`](bug_audit.md) -- the standing list from the four-lens
  audit sweep: every finding with its reproduction, what has been fixed and
  in which commit, why the two items that were decisions rather than repairs
  were decided the way they were (nothing in it is open now), the areas the
  sweep did not cover, and one candidate it chased and cleared.

- [`bug_audit_2.md`](bug_audit_2.md) -- the eight-lens second sweep, aimed at
  the areas the first one listed as not covered: 80 distinct findings (bugs,
  optimization candidates, and coverage/documentation holes), each with its
  reproduction, its oracle, and the structural cause. **76 are fixed**, each
  with a regression test asserting an invariant; the four that remain are
  decisions rather than repairs and the file says why each one is. Read its
  "Status" section before re-opening anything, and its list of user-visible
  behaviour changes before upgrading -- a great many of these fixes turn a
  silent wrong number into a right one or into a raised exception.

- [`bug_audit_2_reproductions.md`](bug_audit_2_reproductions.md) -- the 211
  scripts written while producing that audit and then fixing it, preserved
  verbatim (with absolute paths replaced by placeholders) and annotated with the
  finding each one backs. Evidence, not a test suite: they print numbers a human
  reads and none of them asserts.

- [`bug_audit_3.md`](bug_audit_3.md) -- the third sweep, aimed at the five
  areas the second one left unexercised (the jax SCF solvers, the 3D superfluid
  weight, the qtci SCF backend, Broyden mixing and the per-site filling): 18
  findings, **16 fixed**, each with a regression test that fails on the unfixed
  source, and two more fixed after it (the unbounded `maxite` default of the
  numpy mean-field loops, now 1000, and `fsolve` stalling at a soft mode), with
  the qtci resolution limit in high-symmetry metals closed too. Read its list
  of user-visible changes before upgrading -- two of them are breaking. It also records why the qtci backend was kept although its cross
  interpolation compresses nothing in 2D, and where a fourth sweep should start.

- [`bug_audit_4.md`](bug_audit_4.md) -- the fourth sweep, on the
  superconductivity routines: seven findings, all closed, among them four
  pairing modes that broke Fermi antisymmetry (removed) and the spinless Nambu
  Hilbert space (removed, so a Nambu Hamiltonian is always spinful). Read its
  user-visible changes before upgrading, since three are breaking. It also
  records the closed-form spectra, the Beenakker check and the exact
  diagonalization the Nambu construction passed, and why a spectrum-level
  particle-hole check cannot see a broken Fermi antisymmetry. The small items
  it left open (an unchecked pairing callable, `get_dagger` at $-R$, the BKT
  bracket, two dead functions) were closed afterwards.

- [`bug_audit_5.md`](bug_audit_5.md) -- the fifth sweep, on four places no
  sweep had run against an independent oracle: the KPM mean field, the
  routines that consume a density matrix, the backends that compute one, and
  the BSE package. Sixteen findings, **all closed**, ten as repairs and six
  as the maintainer's decisions: among them a projection gauge that moved
  the exciton energies of the quantics solver, expectation values that
  counted twice or returned zero on a Nambu Hamiltonian, and a KPM
  mean-field loop that amplified its own roundoff and found its Fermi level
  at the wrong temperature. Read its user-visible changes before upgrading,
  since five are breaking, one of them from after the sweep, when the things
  its fix pass left were closed (a numpy-array `nk`, the KPM benchmark
  examples, a spinless $U$ in the Hubbard wrappers, the `scale` of the KPM
  density of states). It also records what was checked and found right, and
  where a sixth sweep should start.

- [`bug_audit_6.md`](bug_audit_6.md) -- the sixth sweep, on the list the
  fifth one left for the next sweep: thirteen findings, **all closed**, each with
  a regression test that fails on the unfixed source (a KPM Fermi count that
  aliased a large `npol` at finite temperature, NaN at `delta=0` in the
  direct density-matrix routes, `get_vev` of an operator without a matrix at
  the wrong temperature, `spectrum.ev2d` counting twice on Nambu, the 1D
  current on a dense Hamiltonian, two embedding self-energies conjugated the
  wrong way, a screened interaction that was not Hermitian off an even
  mesh, and a given KPM scale within 1% of the spectral radius, now
  refused, `cores=` of the KPM engines, now a thread count, the 1D
  velocity, now Cartesian, a defect ignored with `selfenergy=`, now
  refused, and one contract for the callable self-energies of both
  embeddings, under which the non-analytic semimetal one gives the right
  LDOS and is refused by the density matrix); five of them were the
  maintainer's decisions. Read its user-visible changes before upgrading. Most of the
  embedding and BSE items were not reached and are listed for the next sweep.

- [`bug_audit_7.md`](bug_audit_7.md) -- a second reading of the 178 commits
  of 4 to 13 November 2026, done by reading the code only: nine potential
  bugs and seventeen optimization opportunities that a refuting reader
  confirmed at medium or high severity, and thirty-one low ones in one line
  each. Among the bugs: an LDOS with an operator that depends on the basis
  inside a degenerate multiplet, a spinon exchange that is not reciprocal in
  the 36-site example, a cotunneling operator off by a factor 2 with `J=`,
  and a nonlinear Drude term that covers only $x$ and $y$ in 3D. Among the
  optimizations: memory that grows with the k-mesh in the spin
  susceptibility and in the grid Kubo conductivity, and a local moment
  susceptibility that diagonalizes $2n_k^{2d}$ times. All of them were
  repaired on 13 and 14 November 2026 except the default SHG source kernel
  on nearly flat bands, and two optimizations in part; the record lists what
  the repairs found, the decisions of 14 November 2026 on what they left
  open (the generic `Vr` and `Jr` raise when they are not reciprocal, the
  Kondo-lattice loop reads its density matrix the right way, a 3D
  embedding has a density matrix, the qtci DOS and LDOS do not depend on
  the zero or the unit of energy), and what is still open.

- [`audit_open_decisions.md`](audit_open_decisions.md) -- everything the second
  sweep did NOT fix, and the calls that were made one way and could reasonably
  be made the other: three findings that are decisions rather than repairs (the
  QGT batching, since decided by returning the non-Abelian tensor in the
  gauge-independent orbital basis, together with the QGT's orbital-position
  gauge, atomic by default since 25 September 2026, what the AAA's `converged` flag
  should mean, since decided as a local relative error with the broadening
  as its floor, and GPU Tier 2, since signed off and landed), three judgement calls (the one
  normalization deliberately left, since divided out, the aliasing sibling
  since resolved by making the public `get_multicell` copy, and the one
  assertion deliberately weakened, since restored by pinning the gauge),
  and where a third sweep should start (the bond pairing prefactor since
  pinned by a stationarity test). Its last section holds two calls raised by
  guiqula on 27 September 2026 and decided the same day: jax's double
  precision is now set once, in `gpu.py`, rather than as an import side
  effect of eleven modules, and the topology and QPI routines take `write=`,
  with a package-wide `filewrite.set_write()` over every routine that has
  the keyword, which since the same day is every routine that computes
  something and writes it (the mean-field `MF.pkl` included); it lists the
  explicit writers the switch leaves alone, and the routines that now
  return what they used to only write.

- [`bse_excitons.md`](bse_excitons.md) -- Bethe-Salpeter/exciton roadmap:
  observables, iterative solvers, and a measured feasibility study of a
  quantics tensor-train route to large k-meshes. Its last section is the
  GPU port of the static polarizability behind the screened interaction,
  and why that kernel masks where the other device kernels gather: it is
  the one path in the package that loses in double precision on a
  consumer card (7-8x in single from 36 orbitals up, 0.1-0.6x in double).
  After it, the review of 30 October 2026: what profiling found (a
  56-wide LOBPCG block, denormal Chebyshev coefficients, the convolution
  apply for long-range tails) and two magnon bugs, a Casida
  metric split in the wrong place and a ferrimagnet polarized down that
  lost its acoustic branch.
- [`gpu_rpa_spin_response.md`](gpu_rpa_spin_response.md) -- the jax port
  of the Lindhard kernel behind the site-basis spin response
  (now reached with `gpu.set_gpu(True)`): what the switch moves to the device and what stays
  host numpy (the RPA dressing), the measured 833x at N=64 on a V100 and
  the crossover near N=7 (21.7x and N~17 on a consumer GTX 1060, whose 21x
  FP64 penalty is why single precision, `chi_prec`, is the GPU default, at
  ~220x), and what is still open: a run at N=100 itself, a
  newer card, re-profiling the deferred host-side items now that the kernel
  is fast. Its section 12 settles the question of whether the pair-basis
  response wanted a port of its own: it did, since the public entry points
  now take that route for any interaction between different sites and the
  device was refused there outright, and the kernel turned out to be the
  same contraction with the site index replaced by the pair index (29x
  double, 206x single at 128 pairs on a GTX 1060, and slower than the CPU at
  32, so the crossover is in the size of the pair basis).
- [`gpu_kpm_mean_field.md`](gpu_kpm_mean_field.md) -- the KPM mean field
  on the device: why it did not use it (a per-pair numba density matrix
  the switch never reached, and a batched jax kernel no faster than numba
  on a consumer card), the prototype that settled the shape (one block
  recursion over every starting column and k-point, the sparse product as
  ELL row gathers), and the port built from it: 5 to 7x per iteration on
  the CPU and 20 to 75x on the card over the old engine, the card faster
  than exact diagonalization from 768 orbitals up, with single precision
  agreeing with double to 1e-8 along a whole SCF. What it leaves open is
  listed at its end.
- [`sparse_kpm_mean_field.md`](sparse_kpm_mean_field.md) -- the plan for a
  KPM mean field whose memory is linear in the number of orbitals, for
  $10^5$ sites with non-collinear magnetism or Nambu: why the recursion
  already scales and the dense or $n^2$ pieces around it do not (with a
  quadratic `bmat` on the Hamiltonian construction path, measured at 10,000
  sites), why the exact route is $N^2$ in time and practical to about $10^4$
  sites, the light-cone truncation that makes it linear in gapped or
  finite-temperature states, and the four decisions taken on 6 October 2026
  (exact first then truncation, the Fermi level from the lagged exact trace,
  numba on the CPU and jax on the card, and the dense engine kept with the
  sparse one as the default for a sparse Hamiltonian), and the five taken
  while the kernel and the truncation were built. All four steps are built:
  the Hamiltonian construction, guesses and constraints are linear in the
  number of sites, measured to $10^5$, and so is the memory of the KPM mean
  field of a sparse Hamiltonian through both entry points, Nambu included,
  with the dense engine's mean field to roundoff; the CPU recursion is a
  numba kernel with moment doubling (doubled on the card too, in double
  precision only, where it pays), the Fermi level comes from the previous
  iteration's trace (with the measured change of path and of converged
  state), the convergence check of a sparse mean field no longer shrinks
  with the size, and `kpm_radius` truncates the recursion of every orbital
  to a ball of that many hops, with the error measured to fall exponentially
  in a gapped state, so that an iteration at $10^5$ sites takes half a
  minute on six desktop cores in 1.2 GB (two minutes in 2.3 GB with Nambu).
  Left open: an a-posteriori estimate of the truncation error and the
  doubling on the card in single precision (the per-site filling with
  `integration="kpm"` was closed on 16 October 2026). Its last section is what else the recursion makes linear in
  the number of sites: the LDOS map of every site (`mode="KPM"` of
  `h.get_ldos()` and `h.get_multildos()`), built and exact on the light
  cone of the expansion, with the error measured below it; the local KPM
  Chern marker (`h.get_kpm_chern_marker()`, 13 November 2026), held to the
  dense marker in the bulk and close to linear in the number of sites up to
  $4\times10^4$; and the two surveyed and not built, the Kubo-Bastin
  conductivity and the Chebyshev time evolution, with what each needs.
- [`magnons_screening.md`](magnons_screening.md) -- why the screened
  interaction must NOT be used in the magnon RPA kernel on its own, with
  the Goldstone/Ward-identity measurements that settle it.
- [`magnons_tdhf.md`](magnons_tdhf.md) -- the three magnon routes (site
  basis, the interaction's pair basis, and time-dependent Hartree-Fock in
  the electron-hole pair basis), what each covers and why, the Goldstone
  and exact-reference measurements validating all three, how the
  transverse rung of an exchange interaction was carried into the two
  pair-basis kernels (checked against a brute-force TDHF reference), why
  any interaction that couples different sites is now summed in the pair
  basis even through the site-basis entry points, how SzSz/SxSx/SySy
  record their channel and why a global spin rotation refuses an
  anisotropic exchange, what was closed on 5 October 2026 (the `Jr`
  onsite term, kept as the same convention as `Vr`, and the local
  `q=None` response of the site basis, now the average of the dressed
  response as in the pair basis), and what is still open (Nambu and the
  local rotations).

- [`kpm_excitonic_optics.md`](kpm_excitonic_optics.md) -- the Chebyshev resolvent behind
  `method="kpm"` of the excitonic absorption and second harmonic: why no kernel is needed, the
  equality with the dense route, the BSE block applied as a convolution on the mesh (exact,
  independent of the range of the interaction, taking a screened W), the device path and why it
  needs `Precision.HIGHEST`, the timings of every step on the MoS2 monolayer and a TMD
  tetralayer, the all-band comparison with the velocity gauge, and what is open
  (non-Tamm-Dancoff absorption without a Lanczos metric, the convolution for the eigensolvers,
  the position operator on the device). Also the exciton spectral functions at finite Q built
  on the same engine, the projected density of states and the structure factor, with the
  supercell check that pins the phase of the density vertex.
- [`second_harmonic_generation.md`](second_harmonic_generation.md) -- the second-harmonic
  conductivity and susceptibility: why the velocity gauge with the third-derivative vertex is the
  default and the length gauge only a check, the two-gauge and finite-field measurements that pin
  it, the order-of-magnitude check of monolayer MoS2 against Trolle et al. with the three-band TMD
  model, the excitonic SHG from the Tamm-Dancoff BSE and how it was checked, and what is open
  (fitted interlayer couplings for TMD multilayers, full non-Tamm-Dancoff excitons, the triangle
  and qtci integrations in 3D). The grid SHG is 3D since 7 November 2026, and the sum and
  difference frequencies and the photocurrents moved to `nonlinear_response.md`.

- [`nonlinear_response.md`](nonlinear_response.md) -- the nonlinear electrical response, built on
  7 and 8 November 2026: the general velocity-gauge recursion at any order and any tuple of
  frequencies (CPU and GPU), the second- and third-order dc conductivity of a metal in powers of
  tau with the three quantum-metric conventions and why `shibata` is the default, the Berry
  curvature dipole, the shift and injection currents and why the shift current is the total minus
  eta/delta, the Laurent structure of the dc limit with the measured A_0 that matches no
  convention, the external benchmarks (WannierBerri, the quantized CPGE of a Weyl model), the GPU
  crossover, the debug pass (the finite-frequency prefactor of the nonlinear Drude conductivity
  was wrong and is fixed, the tests that could not fail now can) and the optimization pass (the
  dc loop on one BLAS thread, 2x; the device memory budget; the timings before and after on the
  CPU and the card), and what is open (the BLAS threads of the CPU kernel, the row pool and the
  matmul Drude prototypes, the third-order tau^2 and tau^0 terms, spin photocurrents, pinned
  timings on an idle machine, the user notebook and figure). In the seventh bug audit the sign
  of the unshifted dispersion current became -1/2 qmd_disp, the dc batch follows its memory
  budget down to one k-point, and each dc term and the injection current build only what they
  read (2.3x to 3.4x for the dipole, 2x to 2.7x for the injection current).

- [`qtci_optical_response.md`](qtci_optical_response.md) -- where QTCI helps
  the k-integral of the optical conductivity and where it does not: QTCI of
  sigma itself loses or ties against the uniform mesh, the joint (k, omega)
  interpolation loses, the triangle integration gives about a factor 1000 in
  diagonalizations at delta = 1e-3 t, and QTCI of the smooth transition
  energies and weights adds 2 to 15 on top. The triangle sum is now built
  once for all frequencies, 24 to 50 times faster, and the gain holds in
  supercells up to 32 bands. A band window (`ewindow=`, or `nv`/`nc`) keeps
  only the bands near E_F, with exact Re sigma below the window and a
  Kramers-Kronig bias of Im sigma, measured; it brings a 1084-orbital
  twisted bilayer from 16 GB at nk = 6 to 2.9 GB at nk = 36. The qtci path
  now runs one interpolation of all the fields at once, with 3.4 times fewer
  diagonalizations than one per field on the gapped honeycomb lattice at
  nk = 1024 and the same error; on a disordered supercell both forms
  diagonalize the whole mesh and the plain triangle is faster. In 3D the
  Kubo velocities and the tetrahedra are built: at delta = 1e-2 t on a
  stack of Chern insulators the tetrahedra need about 100 times fewer
  k-points than the grid at the absorption edge (8e-2 at nk = 24 against
  nk = 160), and reach 1e-2 at nk = 64 to 96, out of reach of the grid.
  The source kernel now takes tetrahedra too, at 1e-11 of the peak of each
  one: the vertex form cancels like (|x|/s)^3 in 3D, so the tetrahedra
  where that matters are kept whole and summed per frequency, and those
  far from the frequency window go to energy bins. It is 4 to 14 times
  faster than the direct sum (7 at nk = 64 and 400 frequencies), and 130
  times faster than the grid at the same largest error. The 2D cost of the
  tetrahedron change is measured and given back: the moments of the bins
  stop at the order each simplex needs, and a simplex kept whole goes to
  the moments of its bin at the frequencies far from it (the spinful
  Haldane model at 512^2 back to 2.8 s from 3.7 s, at the same error
  against a longdouble reference). A kept tetrahedron is now split along
  its Newton table, so that only its close edges and faces are evaluated
  per frequency, and every point source is summed with the others of its
  energy bin far from it: the 3D kernel at nk = 24 is 2 times faster at 400
  frequencies and 15% at 40, and the 2D one 1.6 times, at the same
  precision against mpmath and longdouble references. Open: many bands,
  metals, SHG, the build at few frequencies, and qtci in 3D.

- [`tetrahedron_density_matrix.md`](tetrahedron_density_matrix.md) -- the
  triangle and tetrahedron density matrix of the mean field,
  `integration="triangle"`: exact filling, the right magnetic state on coarse
  meshes at low T, about 4 times fewer k-points for the magnetization and
  about 64 for the total energy and the bond densities with the Bloechl
  term. Exact diagonalization wins at T = 1e-2 t on dense meshes, and an
  insulator gains nothing. The qtci mean field is rebuilt on the same
  weights: it now reaches the right state on metals, with 4% of the mesh
  on the square ferromagnet at nk = 256 but nearly the whole mesh on the
  honeycomb one up to nk = 128. qtci runs in 3D too, and reaches the
  tetrahedron state of a cubic Hubbard ferromagnet to 1e-14; at 128^3 it
  diagonalizes 0.3% of the mesh on the cubic band and about 60% on layered
  four-band metals, but it is only 10% faster on the first and 2.2 times
  slower on the second, since the tetrahedron weights on the full mesh
  dominate; the X fields reached the rank cap of 400 there, so the cap is
  800 now. The total energy of a BdG mean field is
  settled: every integration adds mu times the requested charge, as "ed"
  did, which is the grand potential plus a constant, stationary at fixed mu
  (qtci added mu times the paired charge, 1.03e-2 lower on the converged
  mesh); the triangle accepts BdG now. The Fermi level of a BdG mean
  field is now found on the charge of the paired state (one step per
  iteration, converged with the mean field, in Vinteraction and
  VJinteraction), so the paired state holds the filling and the energy is
  <H> (U = -2 at filling 0.3 held 0.606 before; U = -6 at filling 0.1 held
  0.334 instead of 0.2). The BdG triangle is the step of exact
  diagonalization on the same mesh, so its Fermi search counts that step
  now (the old triangle count lost the pairing of U = -1 and made a Nambu
  ferromagnet). A qtci BdG mean field from a random guess stopped at the
  Hermiticity check, from an imaginary onsite diagonal of the size of the
  interpolation error; that diagonal is kept real now, and the loop
  reaches the state of "ed". A gapless BdG state whose pairing has died
  out (a Zeeman field above the gap) now holds the filling exactly: "ed"
  and the triangle fill the shell at zero energy in part inside the
  density matrix, and then give the same state (they differed by 8.1e-7
  in the energy), with the step band energy plus the change the fill
  makes, continuous across the fill; a gapless state with the pairing
  alive is not touched. Open: the target of the finite-T Fermi search,
  rounded to a whole number of states, which an exact one would make the
  fill unneeded at T > 0 for "ed", VJinteraction with the triangle, a slower loop when the pairing
  dies, the shell of a gapless state with the pairing alive, and a gapless
  state whose Fermi level falls between two levels instead of on one. RPA chi(q) is built, see
  `tetrahedron_susceptibility.md`. The anomalous Hall of metals with the
  same occupations is built, see the next entry. The seventh bug audit wrote
  the finite-temperature weights in closed form (16 to 19 times faster at
  32^3), contracts every hopping direction with one product, sizes the
  batches by the matrix and repaired the charge of the BdG shell fill; the
  decisions of 14 November 2026: a simplex with no vertex within 40 T of mu
  takes three nodes of the Gauss rule of -f' instead of 208, exact to
  rounding, 6 to 34 times faster in the Fermi search and 3 to 10 in the
  weights at T = 1e-7; the shell vectors keep no cap, and `get_dm` keeps its
  single product for one direction.

- [`simplex_fermi_search.md`](simplex_fermi_search.md) -- the Fermi search
  of the triangle and tetrahedron integration, `fermi_from_energies`, now a
  safeguarded Newton iteration with the simplex DOS from a kernel that
  returns charge and DOS together. It takes 3 or 4 evaluations in a metal
  instead of about 11, and 1 in an insulator, and agrees with the old
  brentq level to 1e-14 in metals. It is 17 to 29 times faster in 2D at
  nk = 256 and 102 times faster at 32^3 and T = 1e-2 (20.8 s to 0.2 s). The
  triangle mean field of the square lattice goes from 0.22 to 0.07 s per
  iteration at nk = 128. In a gap the level is the middle between the
  vertex energies at every T, because the cut Fermi function makes the
  charge exactly flat there too; at T>0 the old search put it up to 2.4e-3
  off the middle, with no effect on the density matrix. Since the seventh
  bug audit the search and the density matrix of one mean-field iteration
  share their eigenpairs and simplices, so an iteration diagonalizes the mesh
  once.

- [`triangle_anomalous_hall.md`](triangle_anomalous_hall.md) -- the
  Fermi-sea anomalous Hall conductivity of a 2D metal,
  `h.get_anomalous_hall_conductivity()`, the Kubo Berry curvature in the
  atomic gauge integrated with the triangle weights of the density matrix
  (`integration="triangle"`, default) or the uniform mesh (`"grid"`). In a
  gap it is the Chern number and both are the same number. At T = 1e-3 t a
  1e-3 accuracy takes nk = 64 with the triangle against 192 to 256 uniform,
  8 to 12 times less wall time, with 1/nk^2 convergence; the atomic gauge
  matters once C3 is broken. In 3D it is a conductivity per unit length,
  C/c for Chern layers of spacing c, with `direction=` choosing the plane;
  there, in a metal at T = 0, the grid error averages out over kz and the
  grid beats the tetrahedra (1e-4 at nk = 64 against 7e-4), while the
  tetrahedra converge smoothly at second order and extrapolate to 2e-5.
  Open: band crossings, qtci, the README bullet and its notebook.

- [`tetrahedron_susceptibility.md`](tetrahedron_susceptibility.md) -- the
  triangle and tetrahedron bare susceptibility, `integration="triangle"` of
  `chiAB_q` and of everything built on it (`h.get_chi()`, the RPA spin and
  density response, the site-basis magnons). It gives U_c and the static
  chi_0 of metals to 1e-5 with 4 to 256 times fewer k-points than the grid
  at T <= 1e-3, and the dynamic response with an error that does not depend
  on delta (64 times fewer k-points at delta = 1e-3). The dynamic response
  goes through a source kernel (bins of transition energy, summed by their
  moments far from each frequency), 11 to 94 times faster at T = 0 and 15 to
  37 at T = 1e-3 than the first kernel, the same numbers to 1e-13; the T = 0
  cost had grown like nk^4 through a bincount. At delta = 1e-3 it now also
  wins in wall time, 7 times at equal accuracy. At high T, and at delta =
  1e-2 for the dynamic response, the grid converges exponentially and wins.
  An odd nk is needed at a commensurate nesting vector. The delta/100
  pair cutoff of the library grid, which biased it by up to 2e-3 at q = 0,
  is replaced by a bound on what the pair contributes. The pair basis of
  `chitk/pairchi.py` had no degenerate pair term, so at q = 0 a metal lost
  its whole Fermi-surface term there (0.205 on a square metal, which is all
  of its static chi_0 at q = 0); it now uses the site-basis rule, the two routes agree
  to 2e-15, and the Goldstone residual of ferromagnetic and spiral metals
  does not move. At finite T a piece of x where the tiling does not move
  is evaluated once, a one-band region is the simplex minus a corner, and
  delta < T goes through the source kernel too, with a correction for
  each frequency whose resonance crosses a simplex: at T = 1e-3 finite T
  now costs 3.5 to 6 times T = 0 (10 to 16 before), at T = 1e-2 11 to 18
  times (65 to 75 before), delta < T is 97 times faster at nk = 256, the
  same numbers to 3e-13. The larger cost at delta = 1e-2 was the load of
  the machine. The seventh bug audit put the trace before the kernel and took
  the matrix elements from the nonzero rows of the operators, 11 times less
  per q on a 200-band cell. Open: the 12 nodes per moving piece of x, a high-T hybrid and the
  triangle on the pair-basis route.

- [`qtci_shg.md`](qtci_shg.md) -- the triangle and qtci k-integrations of
  the second-harmonic conductivity. Single resonances and same-pair products
  are exact; three-band products go through a prism triangulation averaged
  over the vertex orderings, and all of it is second order in 1/nk. At
  delta = 1e-3 t the triangle needs 64 times fewer diagonalizations than the
  grid. The sum is now built once for all frequencies (15 times faster on
  MoS2), so the triangle beats the grid in wall time on both models, by 5 to
  20 times on MoS2. QTCI saves diagonalizations only for two bands and never
  wall time; it calls the shared per-field helper of `qtcitk/meshfields.py`
  now, and the stacked form of the linear response and the LDOS was not tried
  for it. Open: QTCI for many bands, metals, an exact three-band product,
  and 3D. The seventh bug audit made the Taylor rule of the direct sum per
  frequency again (9e-15 against mpmath, it was 5.5e-4 far from resonance),
  kept the first-order term of the series in both kernels, also for an
  exactly flat band, and built only the derivative vertices the weights read
  (2.3x on the fields for xxx). The source kernel keeps its rule on delta and
  loses up to 5e-9 of the peak at delta = 1e-6 on nearly flat bands; the open
  item is the port of the energy bins and the Newton-table split of
  `kubosimplex.py`.

- [`kpm_optical_conductivity.md`](kpm_optical_conductivity.md) -- the
  Chebyshev (KPM) Kubo conductivity of `integration="kpm"` and
  `h.get_dc_conductivity()`: the double-moment formulas, the sign of the
  Kubo-Bastin expression, delta as the line width, and the checks against
  the bands, the f-sum rule and the Chern number up to 10^6 orbitals, with
  where a k-mesh still wins. Open: the GPU, the Hall response at finite
  frequency and the SHG.

- [`qtci_dos_green.md`](qtci_dos_green.md) -- the triangle and tetrahedron
  integration of the DOS (`h.get_dos(integration="triangle")`, `"qtci"`)
  needs fewer diagonalizations than the uniform mesh, by a factor of 250 to
  400 on graphene at delta = 1e-3 t, about 40 on a gapped four-band model
  and 16 on its sz-projected DOS, and about 1000 on a 3D cubic band at
  delta = 1e-2 t and an error of 1e-3; it is exact at delta = 0. qtci adds
  6 to 65 on graphene and 1.7 to 9 on the four-band model, the larger
  factors at the smaller errors.
  One interpolation of all the fields at once needs 1.3 to 3.7 times fewer
  diagonalizations than one per field at the same error, and is twice as
  slow on a 32-band supercell, so the qtci LDOS uses it and the qtci DOS
  keeps one per field. On disordered supercells qtci saves nothing in
  either form. In 3D (2^R x 2^R x 2^R, interleaved bits, which need 1.6 to
  3.8 times fewer points than fused ones at the same rank) qtci diagonalizes
  0.19% of the 128^3 mesh on the cubic band and 35 to 42% on layered
  two-band models, but it does not win in wall time: the tetrahedron sum on
  the full mesh dominates both, and qtci is 15% faster on the cubic band
  and 1.5 to 2.2 times slower on the layered ones at 128^3.
  The triangle bulk Green's function loses to
  `bloch_selfenergy(mode="adaptive")` at equal accuracy in 1D and 2D, so it
  is wired in only for 3D crystals, as `bloch_selfenergy(mode="triangle")`.
  Open: the cost per energy at finite delta, band crossings, a tetrahedron
  sum on the tensor train for 3D qtci, the qtci LDOS in 3D. Its section on
  the seventh bug audit (13 and 14 November 2026) records the repairs: the
  LDOS with an operator as the local matrix element in every mode, the
  simplex sum at finite delta in vertex form, the 3D Green's function from
  the eigenvectors instead of the band projectors, the 3D self-energy of a
  whole energy sweep from one diagonalization of the mesh, a batch of
  diagonalizations that shrinks as one over the square of the cell size,
  and the tetrahedra built in place. Since the decisions of 14 November
  2026 the tetrahedra take a complex energy as a real energy with a
  broadening, so a 3D embedding has a density matrix (one diagonalization
  per contour point, 60 to 76 points), `get_dos(mode="Green",
  gmode="triangle")` sweeps from one diagonalization, and the qtci DOS and
  LDOS measure each band from the middle of its own range, so they do not
  depend on the zero or the unit of energy.

- [`nonlinear_spin_transport.md`](nonlinear_spin_transport.md) -- what the
  X-wave nonlinear Drude spin conductivity covers, the measured performance
  and its nb^3 scaling, why the thermal (spin-Nernst) channel is
  deliberately unbuilt (the paper's linear i-wave result does not
  reproduce, with the Brillouin-zone-domain trap that produced a false
  positive), and why a gapped system returns an exact zero at every order.
  Ezawa's second-order charge response is built in `nonlineardc` (see
  `nonlinear_response.md`); this module is 3D since `5b55594`, and its
  component sweep covers $z$ since the seventh bug audit.

- [`notebook_physics_review.md`](notebook_physics_review.md) -- a physics
  review of all 93 executed notebooks against their own figures and printed
  numbers, which added a literature cell citing the matching figure or
  equation to 56 of them, and the fix pass it led to: the library errors it
  found (Kane-Mele couplings missing half the bonds on diamond, a
  non-Hermitian mean field that dropped the non-Hermitian part, the SCF's Vr
  twice the RPA's, half-strength shells in the charge RPA, a non-Hermitian
  Wannier Hamiltonian, a local-probe S-matrix floor, a p-wave spin current
  that was mesh noise, among others) and the notebooks re-run on converged,
  reference-comparable cases. What it leaves open: the RKKY routine's
  broadening and a few conventions; the p-wave guess, the default mixing
  (now 0.3), the decimation that looped on NaN, the Lorentz kernel's fixed
  $\lambda$ and the Nambu random starts were closed on 16 October 2026.

- [`orbital_field_in_a_superconductor.md`](orbital_field_in_a_superconductor.md)
  -- why `add_peierls` refuses a Hamiltonian that already carries pairing
  (the anomalous term has no single Peierls phase; a real orbital field
  means self-consistent vortices), what the supported field-then-Nambu
  workflow is and the test that pins it, and what a vortex implementation
  would actually need.

- [`topological_invariants.md`](topological_invariants.md) -- which
  topological invariants the package computes, the ones it does not, found
  in a survey of arXiv in four families (one dimension and superconductors,
  crystalline and higher-order topology in two dimensions, real space and
  disorder, three dimensions with the non-Hermitian, driven, interacting and
  bosonic cases), each with its reference, its algorithm, the existing piece
  it builds on and a test with its expected value, most of them measured in
  a prototype. It records the three repairs made while the survey ran (the
  time reversal of a BdG Hamiltonian, so class DIII superconductors get
  their $Z_2$; the Chern number from the Wannier-center winding; a $Z_2$
  count that changed with the resolution on supercells), what was ruled
  out and why (the quadrupole operator among them), nine loose ends found
  while prototyping, and the recommended order. The first package, the general
  Wilson loop with the 3D strong and weak $Z_2$ and the Chern vector, is built,
  with the Fu-Kane parities as its oracle, and so are the second, the winding
  number of chiral chains and the $Z_2$ of helical superconducting wires,
  and the third, the spin Chern number that survives Rashba coupling and the
  mirror Chern number, and then the nested Wilson loop with the quadrupole
  moment, benchmarked against an independent GPL implementation. The nine
  loose ends are closed, four repaired, two left by decision, two documented
  and the scattering-matrix one not reproduced. The Bott and spin Bott
  indices of a disordered periodic supercell, the last of the first tier,
  are built too (4 November 2026), with their sign following the order of
  the lattice vectors as `h.get_chern()` does; left open there are the run
  against PyBott, a sparse route for large samples and a guard on a
  non-integer result. The symmetry eigenvalues of the occupied states, with
  the Chern number modulo $n$ and the parity $Z_2$, are built too
  (5 November 2026), and so is the corner charge for $C_2$, $C_4$ and $C_6$,
  checked against flakes, and the $Z_4$ indicator in three dimensions, the
  symmetry branch of the section on higher-order topology. The multipole
  chiral number of a chiral higher-order insulator is built too
  (13 November 2026), on the torus and on the open flake, held to the
  sublattice-resolved zero modes at the corners of a flake; what is left of
  that section is the octupole moment, the $C_3$ corner charge and a
  real-space check of the hinge modes. The section on collective modes
  (13 November 2026) records the Chern number of an exciton, a magnon and
  the spin exciton of a Kondo lattice, built the same day in three routes:
  the eigenvectors of the BSE and of the magnon problem lifted to the orbital
  pair basis (`h.get_magnon_chern()`, `h.get_exciton_chern()`), the null
  vectors of the RPA kernel (`get_moment_magnon_chern()` on the Kondo
  lattice), and the response function wound as a Green's function, a
  module-level check. It records the measurements, the seven design
  decisions taken that evening and what they leave unbuilt: route 2 as a
  `method=` of `get_magnon_chern`, a spinon Chern method, route 3 as a
  method, a comparison of two meshes with no solve, a gap diagnostic between
  the points of the mesh, the thermal Hall convention and an external check
  of the sign.

- [`unfolding.md`](unfolding.md) -- a check of `operator="unfold"` through
  every mode of the routines that take it (bands, kdos, DOS, the Fermi
  surfaces, QPI) against a brute-force oracle, and what came out of it. The
  repairs: the Fermi surface changed scale by $1/\pi$ or $2\pi$ with the
  mode, the KPM kdos was N times smaller than the other modes, and the QPI
  dropped or misread its unfolding arguments. Three pieces built on it: ARPACK
  eigenvectors made orthonormal inside degenerate levels (scipy routes a
  complex Hermitian matrix to the non-Hermitian driver, so every sum over a
  level double counted), an exact KPM route to unfolding from the primitive
  Bloch states as start vectors (the KITE construction), and a Fermi-surface
  and QPI mesh drawn in the primitive zone and mapped with $M$, for any
  supercell. It also records the two spectral functions of a non-Hermitian
  Hamiltonian (right-eigenvector weights, the default, and the biorthogonal
  Green's-function one of Kozii and Fu, now both available and consistent
  across modes), and lists the user-visible changes.

- [`unreferenced_modules.md`](unreferenced_modules.md) -- the dead-module
  cleanup: how "unreferenced" was actually established (an AST walk of every
  import in the repo, since grep both over- and under-reports on module names
  that are ordinary English words), the 15 modules that were removed and why
  each was safe, and the seven that are unused but import cleanly and were
  deliberately kept because deleting them is a public-API decision rather
  than a repair.

- Computed figures in the user guide -- built on 2026-10-15, in
  `documentation/figures/` (its `README.md` has the conventions). Unlike
  dmrgpy's `compute.py` plus `plot.py` plus `data.npz`, each figure is one
  short script that computes and plots, because the maintainer asked for
  code that is short and easy to read, and every pyqula figure runs in
  seconds to minutes. The images are committed; nothing in CI recomputes
  them, and no test runs the scripts (the static snippet test parses them),
  so a change in the physics shows up only when `run_all.sh` is re-run.

- [`guide_figures_followups.md`](guide_figures_followups.md) -- what running
  every guide snippet for the figures turned up: the library bugs fixed with
  it (the square-lattice M point, the atomic LDOS normalization, the pair
  magnons at Gamma and their infinity crossings, the mean-field convergence
  error, the scipy solvers, the whole-zone Berry curvature window and the
  path method, the kappa sign guard, the temperature at which a mean-field
  Hamiltonian is read, the dk of the Berry density, the q-paths filled
  with nq points, the Keldysh panel confirmation, the slow TDHF mean
  field), the S^zS^z chain that needs a small T, and the broadened Chern
  density, now closed on the real axis with exact_at; nothing is open.

- [`spinon_response.md`](spinon_response.md) -- the dynamical spin response
  of the spinon mean field: the bond decoupling, the held ansatz (phases
  or the whole pattern), the Weiss field for a Zeeman field and the
  spin-channel RPA that were built for cells of about 200 sites, checked
  against arXiv:2503.13831 (hopping, static bubble at K and M, both critical
  couplings); what the Hartree-Fock spinon saddle and its pair ladder do
  instead (unstable, measured); the cost at 196 sites; and everything left
  unbuilt with what is known about it: the longitudinal response in a
  field, the constraint projection, Popov-Fedotov (not a free addition),
  the structure factor at a lattice momentum, the Z2 ansatz,
  exact projection on small clusters, variational Monte Carlo, Schwinger
  bosons. Decisions of 14 November 2026: `Vr` and `Jr` raise in every
  builder when they are not reciprocal, a field in any direction takes the
  collinear route of the bubble (27.6 s to 4.3 s at 196 sites), the binned
  local response of `q=None` is left unbuilt with its reason, and the
  36-site example keeps its origin. Open: the default `nk=60` with `q=None`
  in 2D.

- [`kondo_tunneling.md`](kondo_tunneling.md) -- tunneling spectra with a
  Kondo peak for the two pseudofermion mean fields: the second and third
  order of arXiv:1505.04430 on the spinon state, reduced with Wick's
  theorem to the local spectral matrices of the site (our derivation,
  checked against sums over the Fock space, against dmrgpy and against the
  spin 1/2 of the paper), and the cotunneling spectrum of the Kondo lattice
  (arXiv:0910.1138, arXiv:1001.4530); what they give (no zero-bias peak on a
  spinon continuum, a peak on a site with a level at the Fermi energy), the
  cost at 196 sites, and what is left: the README bullet and notebooks, the
  spinon Kondo-Heisenberg loop, the leading logarithms, the potential term.
  Decisions of 14 November 2026: the Kondo-lattice loop reads the density
  matrix the right way, which matters with a complex hopping between the
  moments, and the two k-loops diagonalize one spin block (3.1 s to 0.6 s at
  196 pairs).

- [`kondo_composite_fermion.md`](kondo_composite_fermion.md) -- the two
  comparisons of arXiv:2302.09701 between matrix product states and the
  mean field of the Kondo chain: the matrix spectral function of the
  conduction electron and the composite fermion, and the spin
  susceptibility of the moments with a residual interaction in the RPA.
  The conventions that had to be settled ($J=2J'$, the amplitude of $F$ in
  $f$, the 3/2 rescale of $V$, two places where the printed mean field of
  the paper was not followed), the exact 4-site ring they were checked
  against, the instability of the paper's $U'$ at weak coupling without
  the rescale, the cost at 200 Kondo pairs, and what is left: the matrix
  product side, the three-fermion continuum of $F$, unfolding, the README
  bullet and notebook.

- [`inplane_valley.md`](inplane_valley.md) -- the check of `valley_x`,
  `valley_y` and `add_valley_exchange`, and their replacement by a closed
  form, a first-neighbor hopping modulated by a plane wave of wavevector
  $K-K'$. It records the two conventions of the closed form (which $K$,
  and the lattice site the phase is measured from, with the rule for
  equally close sites and the failure that made it tolerant), the three
  defects measured in the construction it replaced (a calibration fixed by
  eigenvector phases, wrong bonds at the edges of a finite system, a
  handedness that followed the atom order), the textures that were
  verified (vacancy vortex, zigzag and armchair edges), and what is open:
  a sublattice-diagonal operator and a texture averaged over the Kekule
  cell.

Related, living elsewhere for historical reasons:
`documentation/gpu_porting_plan.md` (jax/GPU roadmap).
