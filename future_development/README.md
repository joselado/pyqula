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

Related, living elsewhere for historical reasons:
`documentation/gpu_porting_plan.md` (jax/GPU roadmap).
