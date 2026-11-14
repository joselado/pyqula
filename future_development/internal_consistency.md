# Internal consistency of the public keywords and defaults

The bug audits made the errors, the `write=` switch and the copy contract of
`get_*` uniform across the package. This file records what is still not
uniform: the same keyword that gives a different number in different routines,
the same choice that is spelled in different ways, and the spellings that only
cost time. Nothing here was changed. Each item is a decision for the
maintainer, and the last section says where we would start.

The survey was made on 13 November 2026, at `9bcf52d`. It has three parts.
An `ast` walk collected every parameter name and default of every function in
`src/pyqula` (12495 parameters). Each public `Hamiltonian` method was then
resolved to the function it delegates to, and only those signatures are quoted
below, since the raw count includes private helpers. About thirty public calls
were run on the gapped honeycomb lattice to read what they return.

## What is already uniform

Every mutator returns None, and modifies the Hamiltonian in place. This was run
for `add_exchange`, `add_onsite`, `add_sublattice_imbalance`, `add_haldane`,
`global_spin_rotation`, `turn_dense`, `setup_nambu_spinor` and `g.shift`.
`get_supercell` returns a new object and never its receiver, as `ccdee4a`
requires. The density of states has the same normalization in its three
modes: on the spinless honeycomb lattice with a sublattice mass 0.2 it
integrates to 1.98, 2.00 and 1.98 states per cell in `"ED"`, `"KPM"` and
`"Green"`. The missing 1% of the first and last is the Lorentzian tail outside
the window from -4 to 4.

## The same keyword gives a different number

- **The default temperature.** `T=None` has five meanings. It is zero in
  `get_anomalous_hall_conductivity` (`topologytk/anomaloushall.py:257`), in the
  simplex density matrix (`dmtk/simplexdm.py:514`) and in `pair_chi0`
  (`chitk/pairchi.py:213`). It is `delta` in `get_optical_conductivity`
  (`conductivitytk/kubo.py:492`), the SHG (`conductivitytk/shg.py:351`), the
  photocurrents (`conductivitytk/photocurrent.py:151`) and the spin
  susceptibility (`chitk/spinchi.py:109`, `chitk/localspinchi.py:199`). It is
  `h.temperature` in `get_density_matrix` (`densitymatrix.py:76`),
  `get_filling` (`spectrum.py:543`), `get_vev` and `get_magnetization`. The
  nonlinear dc conductivity refuses it (`conductivitytk/nonlineardc.py:371`).
  The explicit defaults are 0.05 in `get_drude_weight`, `get_sum_rule_weight`,
  `get_berry_curvature_dipole` and `get_nonlinear_dc_conductivity`, 0.01 in
  `get_nonlinear_drude_conductivity`, 0 in `get_fermi4filling` and 1e-7 in
  `get_exchange_mean_field_hamiltonian`. A converged mean field stores the
  temperature of its loop as `h.temperature`, but only the routines that build
  a density matrix read it. So for the same mean-field `h2`,
  `h2.get_magnetization()` reads the state at its own temperature, while
  `h2.get_optical_conductivity()` reads it at T=0.1. The T=delta of the optics
  is deliberate: the Drude channel is weighted by -df/dE, which vanishes on a
  mesh at T=0 (the docstring of `conductivity.optical_conductivity` says so).
  `tetrahedron_susceptibility.md` already records that `chiAB_q` and
  `pair_chi0`, two routes to the same response, default to T=delta and to a
  step.

- **The default broadening.** `delta=None` gives 5/nk in the DOS
  (`dos.py:210`), 0.001 in the LDOS (`ldos.py:453`), zero in the exact-limit
  routes (`ldos.py:470` and `624`), the energy spacing in `get_kdos_bands`
  (`kdos.py:81`), 1/nk in the magnetic response
  (`chitk/magneticresponse.py:20`) and 6/nk in `get_fermi_surface`
  (`fermisurfacetk/singlefs.py:69`). The explicit defaults run from 1e-5 in
  `get_gf` to 0.1 in the optics and the QPI. Different physics wants different
  broadenings, so what is open is the rule for None rather than one number.
  The resolution of a KPM expansion is set by `delta` in `get_dos`, by `npol` in
  `get_kpm_chern_marker`, and by both in `get_dc_conductivity`.

- **`ewindow` has four meanings.** It is a half width in `get_kdos_bands`
  (default 4.0) and in the KPM DOS, a callable predicate on the energy in
  `get_bands` (`bandstructure.py:159`) and in `operator_berry`, a band window
  given as W or as a pair in `get_optical_conductivity`
  (`conductivity.py:202`), and a pair (emin,emax) in the state writers of
  `states.py`.

- **The channel.** `get_nonlinear_drude_conductivity` defaults to
  `channel="spin"` (`conductivity.py:569`), while its neighbours
  `get_nonlinear_dc_conductivity` and `get_berry_curvature_dipole` default to
  `"charge"`. The spin channel itself is normalized the same way, (up-dn)/2,
  in the nonlinear Drude and dc conductivities, which share
  `nonlineardrude._combine`.

- **The chemical potential.** It is `fermi=` in
  `get_anomalous_hall_conductivity`, `get_dc_conductivity` and
  `get_total_energy`, and `mu=` in `get_nonlinear_drude_conductivity` and in
  the mean field. The optics, the SHG and the nonlinear conductivities take
  neither, and their docstrings ask for `h.shift_fermi()`.

## The same choice has a different call shape

- **Choosing the formalism.** The DOS, the LDOS and the k-resolved DOS use
  `mode=` (`"ED"`, `"KPM"`, `"Green"`, `"adaptive"`, registry at `dos.py:536`).
  The mean field uses `integration=` (`"ed"`, `"kpm"`, `"qtci"`,
  `"triangle"`), and so do the topology and the optics (`"grid"`,
  `"triangle"`, `"qtci"`). The DOS takes both levels, so the same choice is
  `h.get_dos(mode="ED",integration="triangle")` against
  `h.get_mean_field_hamiltonian(integration="triangle")`. `get_fermi_surface`
  takes `mode="eigen"` and `backend="grid"` side by side. The flag
  `use_kpm=True` repeats `mode="KPM"` in `get_dos`, `get_kdos_bands` and
  `get_total_energy`. No routine lowercases these names, so `"KPM"` and
  `"kpm"`, `"ED"` and `"ed"`, and `"Green"` in the topology
  (`topology.py:37`) against `"green"` in the LDOS (`ldos.py:455`) are all
  required spellings. The registries reject a wrong name with the list of the
  accepted ones, so this costs time, not correctness.

- **Return containers.** `get_bands` returns one stacked array of shape
  (2,N), and `get_kdos_bands` one of shape (3,N). `get_dos` returns a tuple
  (e,d), `get_ldos` a tuple (x,y,d), `get_fermi_surface` a tuple (kx,ky,d),
  and `get_berry_curvature` a list of three arrays. The response tensors come
  as an (ne,3,3) array from `get_optical_conductivity` and a (3,3) array from
  `get_drude_weight`, but as a dict keyed by component strings such as
  `"xxx"` from `get_shg_conductivity`. The keyword that selects components is
  `component=` in the optics and `get_dc_conductivity`, `components=` in the
  SHG and the nonlinear routines, and `direction=` in the anomalous Hall
  conductivity.

- **Mutators in the guidance.** The code is uniform, every mutator returns
  None, but the project `CLAUDE.md` says the opposite twice: at line 137
  ("modified in place *and* returned") and in the typical call pattern at line
  179 (`h.add_exchange(...)`, "mutates and returns"). Either the two lines are
  corrected, or the mutators are made to return `self` for chaining, which is a
  package-wide change.

- **Interactions.** The same anisotropic exchange is `J1x`, `J1y`, `J1z` in
  `get_combined_mean_field_hamiltonian` (`scftk/spinspin.py:393`) and `Jx1`,
  `Jy1`, `Jz1` in `get_exchange_mean_field_hamiltonian`
  (`scftk/spinspin.py:284`). The mean field and `get_densitychi_RPA` take
  `U`, `V1`, `V2`, `V3`, `Vr`, while the BSE, the RPA poles and the magnons
  take a `V=` dictionary W, or read `h.V` from the converged mean field. That
  `h.V` stores W/2, a factor of two that `bsetk/interaction.py` explains at
  length because it is easy to miss.

## Spellings

These only cost time, and could go in one sweep with the old names kept as
deprecated aliases.

- `constrains=`, a public keyword of the mean field, in 76 places.
- `verbose=`, `info=`, `silent=` and `verbosity=` for the same switch, and
  `get_multildos` and `get_qpi` print their progress with none of them set.
- `tol=`, `tolerance=` and `maxerror=` for a tolerance, `maxite=` and
  `maxiter=` for an iteration limit.
- `nsuper=`, `nrep=` and `n=` for a number of cells, `ntries=` and `nrand=`
  for a number of random vectors.
- A single energy is `e=` in `get_density`, `get_fermi_surface` and
  `get_iets_ldos`, and `energy=` in `get_gf` and `get_eigenvectors`. Arrays are
  `energies=`, with `frequencies=` next to it in `get_nonlinear_conductivity`
  and `omega=` in `get_nonlinear_drude_conductivity`.
- `get_multihopping` is defined twice in the `Hamiltonian` class
  (`hamiltonians.py:551` and `913`), and the second definition wins.
- `get_topological_invariant` returns a Berry phase (`topology.berry_phase`).

## Decided, with a spread left

`get_chern` and the Bott index follow the orientation of a1 and a2, while the
anomalous Hall conductivity follows the Cartesian z axis, so the two differ in
sign on a left-handed basis (the comment at `topologytk/anomaloushall.py:68`).
The quantum geometric tensor stays in reduced coordinates, while the velocities
and the optics are Cartesian. Both were chosen on purpose; what is left is that
a user has to know which observable uses which frame.

## The KPM density of states, repaired

The survey found two bugs in `dos.dos_kpm`, outside the consistency question,
and both were repaired in the working tree (not committed when this was
written).

The first is that the value at one energy depended on which other energies
were asked. The routine evaluated the Chebyshev series on `ne=len(energies)`
points spread over max|E| and interpolated from them. On the honeycomb lattice
with a sublattice mass 0.2 and `delta=0.05`, the DOS at E=0.5 was 0.15 to 0.24
on an 801-point grid, 0.05 to 0.07 when three energies were asked, and exactly
zero for one energy. ED gives 0.21. The spread of the dense value comes from
the random k-points of the default `random=True`. Now `kpm.moments_to_dos`
evaluates the series at exactly the energies asked, with no grid and no
interpolation, which also removes the interpolation error of the KPM
k-resolved DOS, the other caller that passes energies.

The second is that `h.get_dos(mode="KPM",npol=400)` raised a `TypeError`,
since `npol` was computed from `delta` and then passed a second time. Now
`npol` is accepted and wins over `delta`.

The test is `tests/kpm/test_kpm_dos_energies.py`. It fails on the unfixed
source, where three energies gave 0.042 against 0.327 on the dense grid of the
same fixed mesh. With the repair, `tests/kpm`, `tests/dos`, `tests/kdos`,
`tests/ldos`, `tests/nonhermitian`, `tests/unfolding`, `tests/operators` and
`tests/filewrite` pass, 362 tests.

## What was not checked

We did not check whether every built-in lattice has a right-handed a1, a2, so
we do not know whether the sign split of the last section ever appears in
practice. The return shapes were read on about thirty methods, not all of
them. The case of the mode strings was compared by a grep of the `==`
comparisons, not by calling every routine.

## Where to start

The default temperature is the first decision, because it changes a number
silently on the most common chain, a mean field followed by an observable, and
half of a rule already exists in `h.temperature`. Three rules are possible.

- Read `h.temperature` when h comes from a mean field, and the routine's own
  default otherwise. One state is then read at one temperature, but a mean
  field converged at the default T=1e-7 loses the Drude channel of its optics.
- Keep a default per family and align each family: the Kubo routines (optics,
  SHG, photocurrents, Drude weight, sum rule) at T=delta, the Fermi-surface
  routines (anomalous Hall, Berry curvature dipole, nonlinear dc and Drude) at
  one value, and the density-matrix routines at `h.temperature`, all in one
  table of the user guide. This changes the fewest numbers.
- Make T explicit in every response routine, so that `T=None` raises as it
  already does in the nonlinear dc conductivity. Nothing is hidden, but every
  call without T breaks.

After it, the formalism keyword is the largest surface a user types, and the
return containers are the item that most affects scripts written against the
package.
