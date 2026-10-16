# What the guide figures turned up and left open

The figures of the user guide (`documentation/figures/`, 2026-10-15) were
computed by running every snippet of the guide, and reading the output
against the text found wrong sentences and a few library bugs. The wrong
sentences and the bugs listed under "Fixed" were repaired in the same piece of
work. This file keeps what was fixed, what was measured, and what is still open.

## Fixed

- The square-lattice "M" label resolved to (1/2,0), the same point as "X"
  (`kpointstk/labels.py`); it is now the zone corner (1/2,1/2) when the
  reciprocal vectors are orthogonal. Test:
  `tests/bandstructure/test_square_lattice_m_point.py`.
- `projection="atomic"` LDOS maps were unnormalized, so on a bipartite ribbon
  the summed map was 46.7 at -2t against 2260 at +2t at equal DOS. Each state
  is now normalized on the grid to its TB weight
  (`ldostk/atomicmultildos.py`). Test: `tests/ldos/test_atomic_ldos_per_state.py`.
- `method="pair"` magnons listed the Sz-conserving excitons of the full pair
  basis (a fourfold 0.96 at Gamma on the U=3, V1=0.5 honeycomb antiferromagnet)
  as magnons, and missed the Goldstone mode, which the broadening moves to
  omega=delta, below the first grid point. `channel="auto"` now keeps only the
  spin-flip sectors of a collinear state, and omega=0 is added to the scan.
  Test: `tests/chi/test_pair_magnon_sectors.py`.
- A mean field that does not converge raises `MeanFieldConvergenceError`
  (`scftk/convergence.py`, the maintainer's decision) instead of returning
  None; the error keeps the last SCF object as `err.scf`.
- The scipy solvers of the density-density mean field (`krylov`, `anderson`,
  `broyden1`, `linear`) never set `scf.converged`, so the call failed with an
  AttributeError. Test: `tests/scf/test_scipy_solvers.py`.
- `h.get_qpi()` printed "Done QPI" once per energy.
- A second round fixed what the first one left: `solver="linear"` stepped
  away from the fixed point (scipy's `linearmixing` assumes F=g(x)-x, the code
  passed x-g(x)), now `alpha=-mix`, the plain loop's step; the pair-basis pole
  search drops crossings of Re lambda through infinity next to a pole of chi0,
  with a sign test (Im lambda times the slope of Re lambda is negative at a
  causal zero and positive at a pole), and `method="rpa"` takes `channel=`;
  `h.get_berry_curvature()` samples a Cartesian window of whole zones (two for
  the hexagonal lattices, one for the square one, the parallelogram of b1 and
  b2 for an oblique one), so `np.mean(b)/(2*np.pi)` is the Chern number, and
  `topology.dOmega_dE_kmap` uses the same window and now honours its `delta`;
  `h.get_berry_curvature_path()` is the curvature along a k-path again, as it
  was before 2022; `get_kappa` raises a clear error when the conductance changes
  sign between the two couplings, and fits |G| when it is negative at both.
  The site-basis `"rpa"` pole search (`chitk/rpa.py`) got the same sign test,
  now the only copy, and adds omega=0 to the scan, so it finds the Goldstone
  mode; tests `tests/chi/test_rpa_infinity_crossings.py`. With it the plasmon
  snippet of spin-response.md returns no pole (its old 0.60t "pole" had a flat
  |lambda| near 0.6, never close to zero), and the charge-channel snippet three
  instead of seven.
- The S^zS^z polarized chain that stopped converging at nk=60 and 100 was not
  a bug: with an even number of electrons on a mesh holding k=pi, a pair of
  states at +k_F and -k_F sits exactly at the Fermi level, and at the default
  T=1e-7 a tiny imaginary hopping fills one and empties the other, which the
  next iteration reverses. `T=0.01` converges it; `exchange.md` says so. Test:
  `tests/scf/test_szsz_polarized_chain.py`.
- Last round: observables read a mean-field Hamiltonian at the temperature
  its loop converged at, stored as `h.temperature` next to `h.nk`
  (`densitymatrix.default_smearing`, and `get_filling` with `T=None`). The
  polarized S^zS^z chain at T=0.01 read a moment of -0.4024 (nk=60) and
  -0.4100 (nk=100) instead of -0.4, because its pair of states at the Fermi
  level sits about 1e-6 below zero and the default smearing of 1e-6 counted
  both as full. A Hamiltonian that is not the result of a mean field has
  temperature 0 and reads as before. The MAR cost sentence of transport.md
  was rewritten from timings (every voltage costs under a second; small
  voltages need more sidebands but integrate over a shorter window), the
  outdated omega=0 comment of notebook 10 was fixed and the notebook
  re-executed (same figure), and the two README images of the Berry
  curvature were redrawn with the whole-zone window.

- The finite-difference step of `topology.chern_density` and
  `dOmega_dE_kmap` is now `dk=delta/25` by default instead of a fixed 0.02
  (and 0.01), which was too large at every delta of 0.05 and below
  (`topologytk/green.py`, `default_dk`). Test:
  `tests/topology/test_chern_density_dk.py`, a step four times smaller gives
  the same density, the old 0.02 does not.
- The routines that scan a collective mode along a q-path
  (`get_magnon_bands`, the pair and TDHF magnons, `get_plasmon_bands`,
  `get_exciton_bands`, `get_qdos_iets`) took a list of vectors as the literal
  list of q-points and ignored `nq`, so `qpath=[[0,0,0],[0.5,0,0]], nq=26`
  was two points with no pole. They now go through `klist.get_qpath`: with
  `nq` given, the vectors are the vertices of a path of `nq` points; without
  it, or with `nq` equal to their number, they are the points, as before.
  The k-path routines (`get_bands` and the others with a numeric `nk`
  default) keep the literal list. Test: `tests/chi/test_qpath_nq.py`.
- The second TDHF mode near K (1.09t, 1.01t, 0.97t at nk=18, 24, 30) is not a
  defect: it is the lowest pair state of the spin-flip continuum, a discrete
  level on a finite mesh, coming down to the bottom of the continuum at K,
  which is the mean-field gap 0.9106t (the smallest E_c(k+K)-E_v(k), the
  same at nk=60, 120 and 240 and for SCF meshes 24 and 30). spin-response.md
  now says so.
- The mean field of the U=3, V1=0.5 honeycomb antiferromagnet at `nk=24`
  (spin-response.md, `spin_pair.py`, `spin_tdhf.py`) did not stall: with
  `maxerror=1e-10` and the defaults it stopped at maxite=1000 with an error of
  1.06e-9 (and `VJinteraction` returns `converged=False` there, it does not
  raise; only the Hamiltonian methods raise). The error of the plain-mixing
  loop falls geometrically with no floor, down to 1e-13 at iteration 1522.
  The rate per iteration at `mix=0.1` is 0.974, 0.982, 0.982, 0.982 at nk=6,
  12, 18, 24, and 0.947, 0.91, 0.88, 0.84 at mix=0.3, 0.5, 0.7, 0.9 at nk=24,
  which is 1-mix*(1-lambda) with one slow eigenvalue lambda of about 0.82 of
  the fixed-point map. So the error measure is not the cause: `diff_mf` is a
  per-entry mean of mean-field matrices whose size does not depend on nk,
  and the Fermi level comes from exact diagonalization. The guide and the two
  scripts now use `mix=0.5, maxerror=1e-10`, converged in 220 iterations,
  with a Goldstone residual of 2.5e-10 (2.6e-8 at the old 1e-8). The figures
  were not regenerated, since a 1e-8 change in the mean field does not move
  them. Test: `tests/scf/test_vj_antiferro_tight_tolerance.py`. Left open: a
  Broyden or Anderson step in `_run_anisotropic_scf` would remove the need
  to tune `mix` by hand, an architectural change to propose, not made here;
  what the slow mode is physically was not traced.
- The Keldysh STM-tip conductance (`transport_stmtip`) at `tol=2e-3`,
  `nmax_max=40` returned one point, eV=0.2725, at 0.173 instead of 0.319,
  with no warning. The sidebands were not the cause: both settings stop at
  nmax=32, and with direct self-energies both give 0.3191. What the two
  settings change is the window of the AAA fit of the self-energies that
  `didv(energies=...)` builds for the whole sweep, and the fit is within its
  1e-3 contract; the error is in the quasienergy integral at voltage+dv,
  where the adaptive Gauss-Kronrod rule accepted the panel [0.069,0.138],
  which holds the gap-edge peak at eV=Delta, with an estimated error of 9e-6
  and an actual one of 8e-4, because its 10-point and 21-point sums agreed
  by coincidence. So the current came out 1.2% low, and the finite
  difference of `keldysh_didv`, which divides by Ip-Im, about 1/40 of the
  current here, turned that into a factor 1.85 in dI/dV. scipy's `quad`
  (`quadrature="adaptive_scipy"`) is fooled identically, and the tight
  setting was right by luck, not by design. `adaptive_quad_batch`
  (`keldyshtk/quadrature.py`) now takes as the error of each panel the
  larger of its Gauss-Kronrod estimate and the disagreement between its
  parent's estimate and the sum of the parent's two children, so a panel is
  accepted only when two independent node sets agree, and the loose setting
  gives 0.3191 at that point. `confirm=False` restores the old rule. The
  price is one more level of bisection wherever the integral is refined:
  an easy integrand takes 63 nodes instead of 21, and a whole `dc_current`
  call 1428 instead of 630 nodes on the deep-subgap benchmark, 5418 instead
  of 3906 on the worst-accuracy one and 252 instead of 84 on the
  normal-normal junction. Tests: `tests/keldysh/test_quadrature_gap_edge.py`
  and `test_one_coincidental_estimate_does_not_end_the_refinement` in
  `tests/keldysh/test_adaptive_quad_batch.py`. Left open: two levels can
  still agree by coincidence on a narrow enough peak, so this makes the
  failure much rarer, not impossible; the epsrel=1e-3 of the quasienergy
  integral is fixed inside `dc_current` and does not follow `tol`; and the
  finite difference still amplifies a 1e-3 error of the current by Ip/(Ip-Im):
  at this point a 1e-3 error of I=0.067 over 2dv=5.5e-3 is 1.2e-2 against
  G=0.32, about 4%, so the tolerance bounds dI/dV to that, not to `tol`
  (the measured gap between the loose and the tight setting is now 2e-4). The figure was not regenerated, since it already uses the tight
  setting.

- The Berry curvature density in energy (quantum-geometry page) reaches the
  Chern number only slowly as delta goes down. The first numbers (plateau
  1.37, 1.62 and 1.79 at delta=0.1, 0.05 and 0.02) were taken at the default
  finite-difference step `dk=0.02` of `topology.chern_density`, and part of
  the deficit was that step, not the broadening: the derivatives in k are
  finite differences over dk in reduced coordinates, and near K the bands move
  by roughly 0.1 to 0.2t over dk=0.02, more than delta. Measured on the
  spinful Haldane model, t2=0.05, es from -3.5 to 0:
  delta=0.05, nk=20: 1.624 (dk=0.02), 1.686 (dk=0.005), 1.690 (dk=0.001);
  nk=40, dk=0.001: 1.699; nk=24, dk=0.002, step delta/2: 1.709.
  delta=0.02, nk=20: 1.788 (dk=0.02), 1.864 (dk=0.002).
  delta=0.01, dk=0.001, step delta/2: 1.924 (nk=20), 1.982 (nk=21, K on the
  mesh), 1.942 (nk=30), so at this delta the plateau is not converged in nk
  to better than about 0.06.
  The energy step: at delta=0.05, nk=12, a step of delta/4, delta/2 and delta
  gives 2.078, 2.077 and 2.200, so delta/2 is enough. With dk converged the
  deficit is about 0.30, 0.14 and 0.05 at delta=0.05, 0.02 and 0.01, still
  the slow Lorentzian tail of the broadened Green's function, but smaller
  than the old fit 2.7 delta ln(1/delta). The guide figure now uses
  delta=0.05, nk=48, dk=0.002 and the full window (about 17 minutes on one
  core), whose plateau is 1.70; the density itself stays a comb of k-shells
  at this nk, since a smooth density at delta=0.05 needs nk above about 100.
  The default dk is now delta/25 (see Fixed), so this remaining deficit is
  the broadening alone, not a defect of the formula.
  Fixed by closing the path: the cumulative integral is the integral of an
  analytic function along Im z = delta, and inside a gap the integrand is
  smooth down to the real axis, so adding vertical segments at the energy
  and at the first energy of es (below the band) removes the broadening
  exactly. `chern_density(..., exact_at=[...])` returns that value
  (`topologytk/green.py`, `dOmega_dE_vertical`, eight Gauss-Legendre nodes,
  the same with sixteen). Spinful Haldane model, t2=0.05, at zero energy:
  nk=12: 2.557 (delta=0.1) and 2.583 (delta=0.05), against 1.63 and 2.08
  broadened; nk=24: 2.011 and 2.019; nk=36: 1.993 and 1.999, so it does
  not depend on delta and goes to C=2 with the mesh. Test:
  `tests/topology/test_chern_density_dk.py`.

## Open


Nothing is open.
