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
- [`magnons_tdhf.md`](magnons_tdhf.md) -- the two magnon routes (site-basis
  RPA vs time-dependent Hartree-Fock in the spin-flip pair basis), what
  each covers, the Goldstone measurements that validate both, why a
  neighbor-shell exchange interaction works in the RPA once the SCF records
  its three spin channels, and why a neighbor-shell density-density one
  structurally cannot.

Related, living elsewhere for historical reasons:
`documentation/gpu_porting_plan.md` (jax/GPU roadmap).
