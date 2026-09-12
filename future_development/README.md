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

- [`audit_open_decisions.md`](audit_open_decisions.md) -- everything the second
  sweep did NOT fix, and the calls that were made one way and could reasonably
  be made the other: three findings that are decisions rather than repairs (the
  QGT batching that is not output-equivalent, what the AAA's `converged` flag
  should mean, GPU Tier 2), three judgement calls (the one normalization
  deliberately left, the one aliasing sibling deliberately left, and the one
  assertion deliberately weakened), and where a third sweep should start.

- [`bse_excitons.md`](bse_excitons.md) -- Bethe-Salpeter/exciton roadmap:
  observables, iterative solvers, and a measured feasibility study of a
  quantics tensor-train route to large k-meshes.
- [`magnons_screening.md`](magnons_screening.md) -- why the screened
  interaction must NOT be used in the magnon RPA kernel on its own, with
  the Goldstone/Ward-identity measurements that settle it.
- [`magnons_tdhf.md`](magnons_tdhf.md) -- the three magnon routes (site
  basis, the interaction's pair basis, and time-dependent Hartree-Fock in
  the electron-hole pair basis), what each covers and why, the Goldstone
  and exact-reference measurements validating all three, and the one thing
  still open (the transverse exchange rung in the pair-basis kernels).

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

- [`unreferenced_modules.md`](unreferenced_modules.md) -- the dead-module
  cleanup: how "unreferenced" was actually established (an AST walk of every
  import in the repo, since grep both over- and under-reports on module names
  that are ordinary English words), the 15 modules that were removed and why
  each was safe, and the seven that are unused but import cleanly and were
  deliberately kept because deleting them is a public-API decision rather
  than a repair.

Related, living elsewhere for historical reasons:
`documentation/gpu_porting_plan.md` (jax/GPU roadmap).
