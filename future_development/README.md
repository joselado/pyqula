# future_development

Maintainer-facing roadmaps for work that is planned, partially done, or
scoped-but-not-started. These are notes to a future implementer (human or
Claude), not user documentation -- user-facing behaviour belongs in
`documentation/user_guide.md` and `README.md`.

A document here should say what exists today, what is missing, what was
already measured or ruled out, and what the next decision point is, so that
picking the work up again does not mean re-deriving conclusions that were
already reached once.

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

Related, living elsewhere for historical reasons:
`documentation/gpu_porting_plan.md` (jax/GPU roadmap).
