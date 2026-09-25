# future_development

Maintainer-facing roadmaps for work that is planned, partially done, or
scoped-but-not-started. These are notes to a future implementer (human or
Claude), not user documentation -- user-facing behaviour belongs in
`documentation/user_guide.md` and `README.md`.

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

- [`audit_open_decisions.md`](audit_open_decisions.md) -- everything the second
  sweep did NOT fix, and the calls that were made one way and could reasonably
  be made the other: three findings that are decisions rather than repairs (the
  QGT batching, since decided by returning the non-Abelian tensor in the
  gauge-independent orbital basis, together with the QGT's orbital-position
  gauge, atomic by default since 25 September 2026, what the AAA's `converged` flag
  should mean, since decided as a local relative error with the broadening
  as its floor, and GPU Tier 2, still open), three judgement calls (the one
  normalization deliberately left, since divided out, the aliasing sibling
  since resolved by making the public `get_multicell` copy, and the one
  assertion deliberately weakened, since restored by pinning the gauge),
  and where a third sweep should start (the bond pairing prefactor since
  pinned by a stationarity test).

- [`bse_excitons.md`](bse_excitons.md) -- Bethe-Salpeter/exciton roadmap:
  observables, iterative solvers, and a measured feasibility study of a
  quantics tensor-train route to large k-meshes. Its last section is the
  GPU port of the static polarizability behind the screened interaction,
  and why that kernel masks where the other device kernels gather: it is
  the one path in the package that loses in double precision on a
  consumer card (7-8x in single from 36 orbitals up, 0.1-0.6x in double).
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
  anisotropic exchange, and what is still open (the Jr onsite terms,
  Nambu, the local rotations, and the q-averaged site-basis response).

- [`nonlinear_spin_transport.md`](nonlinear_spin_transport.md) -- what the
  X-wave nonlinear Drude spin conductivity covers, the measured performance
  and its nb^3 scaling, why the thermal (spin-Nernst) channel is
  deliberately unbuilt (the paper's linear i-wave result does not
  reproduce, with the Brillouin-zone-domain trap that produced a false
  positive), and why a gapped system returns an exact zero at every order.

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
  mirror Chern number. Still open: the Bott and spin Bott index, the last of
  the first tier.

- [`unreferenced_modules.md`](unreferenced_modules.md) -- the dead-module
  cleanup: how "unreferenced" was actually established (an AST walk of every
  import in the repo, since grep both over- and under-reports on module names
  that are ordinary English words), the 15 modules that were removed and why
  each was safe, and the seven that are unused but import cleanly and were
  deliberately kept because deleting them is a public-API decision rather
  than a repair.

Related, living elsewhere for historical reasons:
`documentation/gpu_porting_plan.md` (jax/GPU roadmap).
