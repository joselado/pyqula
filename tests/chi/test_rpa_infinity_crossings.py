"""The site-basis pole scan (chitk.rpa._poles_from_chi_matrix) returns the
zeros of 1 - V chi0 and not the points where an eigenvalue goes through
infinity next to a pole of chi0, and it adds w=0 to a grid that starts above
zero, so the Goldstone mode is found.

On the onsite U=3 honeycomb Neel state at nk=6, with a grid from 1e-3 to 3
and delta=1e-3, the scan used to return 1.3519 at Gamma with abs(gamma)=16,
and no Goldstone mode. The magnons away from Gamma are those of the pair
basis and of time-dependent Hartree-Fock.
"""
import numpy as np

from pyqula import geometry
from pyqula.bsetk.spinflip import magnon_energies
from pyqula.meanfield import VJinteraction

NK = 6
DELTA = 1e-3
GRID = np.linspace(1e-3, 3., 400)
EMAX = 1.8
TOL = 2e-3


def _neel():
    g = geometry.honeycomb_lattice()
    return VJinteraction(g.get_hamiltonian(), U=3.0, filling=0.5,
                         mf="antiferro", nk=NK, maxerror=1e-10).hamiltonian


def _poles(h, q, method):
    qs, ws, gs = h.get_magnon_bands(method=method, qpath=[q], energies=GRID,
                                    delta=DELTA, nk=NK)
    return ws, gs


def test_rpa_gamma_has_the_goldstone_mode_and_no_infinity_crossing():
    h = _neel()
    ws, gs = _poles(h, [0., 0., 0.], "rpa")
    assert np.min(ws) < 2*DELTA, ws  # Goldstone, at about delta
    assert np.min(np.abs(ws - 1.3519)) > 1e-2, ws
    assert np.max(np.abs(gs[ws < EMAX])) < 0.05, (ws, gs)


def test_rpa_magnons_match_pair_and_tdhf_away_from_gamma():
    h = _neel()
    for q in ([0.05, 0.05, 0.], [0.2, 0.1, 0.]):
        ws, gs = _poles(h, q, "rpa")
        assert np.max(np.abs(gs)) < 2., (q, ws, gs)  # no crossing at infinity
        pair, _ = _poles(h, q, "pair")
        tdhf = np.sort(magnon_energies(h, Q=q, nk=NK).real)
        for w in pair[pair < EMAX]:
            assert np.min(np.abs(ws - w)) < TOL, (q, w, ws)
        for e in tdhf[tdhf < EMAX]:
            assert np.min(np.abs(ws - e)) < TOL, (q, e, ws)
