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

## Open

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
  Open for the library: the default dk=0.02 is too large for every delta
  measured here (0.05 and below), and a default tied to delta (dk about delta/25) would
  remove the error; the cost is one evaluation of five Green's functions per
  k-point and energy, about 1.5 ms, which is what limits nk and delta here.
- `h.get_plasmon_bands()` (and the other routines that take a q-path) use a
  list of vectors as the literal list of q-points, so
  `qpath=[[0,0,0],[0.5,0,0]], nq=26` is two points, q=0 and q=1/2, where there
  is no pole, and `nq` is ignored without a word (`klist.get_kpath`). It is
  the same convention as a k-path of `h.get_bands()`, so this is a trap more
  than a bug: either say it in the docstrings, or interpolate the segments
  when `nq` is given with explicit vectors.
- The Keldysh STM-tip conductance (`transport_stmtip`) at `tol=2e-3`,
  `nmax_max=40` returned one point, eV=0.2725, at half its value, with no
  warning, while the run at `tol=5e-4`, `nmax_max=60` is smooth through it.
  The adaptive sideband or integration loop may stop early on a point it
  thinks converged; not traced.
- The TDHF snippet of the honeycomb antiferromagnet at `nk=24`: the mean-field
  loop stalls at an error of about 3e-9, so `maxerror=1e-10` raises the
  convergence error; the guide now uses `maxerror=1e-8`, which leaves a
  Goldstone residual of 3e-8. Why the error stalls is not traced.
- The second TDHF mode near K is the bottom of the continuum and
  is not converged in nk (1.09t, 1.01t, 0.97t at nk=18, 24, 30).
