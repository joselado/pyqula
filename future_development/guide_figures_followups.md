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

## Open

- `h.get_magnetization()` on the converged polarized chain above reads -0.4024
  at nk=60 and -0.4100 at nk=100 while the self-consistent density matrix gives
  exactly 0.4: with the Fermi level on the degenerate pair, a count at T=0
  takes one electron too many (1/nk).
- `jupyter-notebooks/functionalities/interacting_mean_field_hamiltonians/10_rpa_collective_modes.ipynb` has a code comment "omega=0 excluded: trivial
  root of every kernel" that is out of date: the pole scans now add omega=0
  themselves (for the Goldstone mode) and no spurious root appears there. Its
  figure plots only |gamma|<0.05 poles, so it should not move; not re-executed.
- transport.md says small voltages are the expensive ones in the MAR
  calculation because they need many sidebands, but with the shared lead
  self-energy built once, `dc_current` took 0.05 s at eV=0.05 against 1.5 s at
  0.15. Not checked further.
- The Berry curvature density in energy (quantum-geometry page) reaches the
  Chern number only slowly as delta goes down (plateau 1.37, 1.62 and 1.79 at
  delta=0.1, 0.05 and 0.02). This is the Lorentzian tail of a broadened
  Green's function, a deficit of roughly 2.7 delta ln(1/delta), not a defect
  of the formula; the contour-integrated Green's-function Chern number is the
  exact limit.
- The README images `images/berry_curvature.png` and `images/2DTSC.png` were
  drawn with the old square window of the Berry curvature map.
