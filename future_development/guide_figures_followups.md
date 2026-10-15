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
  Chern number only slowly as delta goes down (plateau 1.37, 1.62 and 1.79 at
  delta=0.1, 0.05 and 0.02). This is the Lorentzian tail of a broadened
  Green's function, a deficit of roughly 2.7 delta ln(1/delta), not a defect
  of the formula; the contour-integrated Green's-function Chern number is the
  exact limit. Nothing to fix unless a sharper density in energy is wanted.
