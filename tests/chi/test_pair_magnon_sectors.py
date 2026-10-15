"""The poles of the pair-basis ladder are the magnons of time-dependent
Hartree-Fock, at Q=Gamma too.

The pair basis of chitk/pairchi.py holds every spin combination, so the
zeros of 1 + K chi0 include the Sz-conserving (charge and longitudinal
spin) excitons as well as the spin-flip ones. For a state that conserves
Sz the two sectors do not couple, and the magnons are the zeros of the
spin-flip sectors alone, which is what bsetk/spinflip.py's default
channel="spinflip" solves for. On the honeycomb U=3, V1=0.5 Neel state at
nk=6 the Sz-conserving sector adds a fourfold pole at 0.96 at Gamma and
poles at 1.146 and 1.198 at (0.05,0.05), all of them eigenvalues of the
TDHF problem with channel="all" and with no weight in the transverse
spin response.
"""
import numpy as np
import pytest

from pyqula import geometry
from pyqula.bsetk.spinflip import magnon_energies
from pyqula.chitk import pairchi, rpa
from pyqula.meanfield import VJinteraction

NK = 6
DELTA = 1e-3
GRID = np.linspace(1e-3, 3., 400)  # the grid of the user guide
EMAX = 1.6  # the window compared
SHARP = 0.05  # |gamma| below which a pole is a mode, see _poles_from_chi_matrix
TOL = 2e-3


def _neel():
    g = geometry.honeycomb_lattice()
    return VJinteraction(g.get_hamiltonian(), U=3.0, V1=0.5, filling=0.5,
                         mf="antiferro", nk=NK, maxerror=1e-10).hamiltonian


def _sharp_poles(h, q, **kw):
    poles = pairchi.pair_rpa_poles(h, q=q, energies=GRID, delta=DELTA,
                                   nk=NK, **kw)
    return np.array([w for w, gm in poles if abs(gm) < SHARP and w < EMAX])


def _tdhf(h, q, **kw):
    es = np.sort(magnon_energies(h, Q=q, nk=NK, **kw).real)
    return es[es < EMAX]


def _close(a, bs, tol=TOL):
    return np.min(np.abs(np.array(bs) - a)) < tol


def test_pair_poles_are_the_tdhf_magnons_at_gamma_and_nearby():
    h = _neel()
    for q in ([0., 0., 0.], [0.05, 0.05, 0.]):
        pair = _sharp_poles(h, q)
        tdhf = _tdhf(h, q)
        assert len(pair) > 0
        for w in pair:  # no pole that TDHF does not have
            assert _close(w, tdhf), f"q={q}: pole {w} not in {tdhf}"
        for e in tdhf:  # and every TDHF magnon is found
            if abs(e) < 1e-6: # Goldstone, moved by the broadening to ~delta
                assert np.min(pair) < 2*DELTA, f"q={q}: no Goldstone, {pair}"
            elif q == [0., 0., 0.] and abs(e - 1.0076) < 1e-3:
                # the bottom of the spin-flip continuum at Gamma, a pair
                # state the interaction does not couple: a pole of chi0,
                # not a zero of 1 + K chi0, and with no weight in
                # get_transverse_spinchi
                continue
            else: assert _close(e, pair), f"q={q}: magnon {e} not in {pair}"


def test_channel_all_keeps_the_sz_conserving_excitons():
    h = _neel()
    q = [0., 0., 0.]
    pair = _sharp_poles(h, q, channel="all")
    tdhf = _tdhf(h, q, channel="all")
    assert np.sum(np.abs(pair - 0.960) < TOL) == 4  # the fourfold exciton
    assert np.sum(np.abs(tdhf - 0.960) < TOL) == 4
    assert np.sum(np.abs(_sharp_poles(h, q) - 0.960) < TOL) == 0


def test_zero_crossing_test_tells_a_zero_from_a_pole():
    """lambda = c (w - w*) and lambda = A/(w* - w), with w* on the side of
    the poles of chi0 (Im w* > 0): the real part changes sign in both, and
    only the first is a zero. The answer does not depend on the complex
    prefactor, so a few phases are checked."""
    ws = np.linspace(0., 2., 201)
    wstar = 1.0 + 0.01j
    for phase in (1., -1., np.exp(0.4j), np.exp(2.5j)):
        zero = (phase*(ws - wstar))[:, None]
        pole = (phase/(wstar - ws))[:, None]
        pz = rpa._zero_crossings(ws, zero)
        assert len(pz) == 1 and abs(pz[0, 0] - 1.0) < 0.02
        assert len(rpa._zero_crossings(ws, pole)) == 0


def test_infinity_crossings_are_not_poles():
    """Next to a pole of chi0 the eigenvalue of 1 + K chi0 goes through
    infinity, and its real part changes sign there too. Before the sign
    test of _zero_crossings these came back as poles with a large
    abs(gamma), at 1.0024 at Gamma and at 1.2430 and 1.3369 at
    (0.05,0.05); the zeros around them, the damped 1.2798 included, stay."""
    h = _neel()
    for q, fake, true in (
            ([0., 0., 0.], [1.0024], [1.4408]),
            ([0.05, 0.05, 0.], [1.2430, 1.3369], [0.5284, 1.2798, 1.5330])):
        poles = pairchi.pair_rpa_poles(h, q=q, energies=GRID, delta=DELTA,
                                       nk=NK)
        ws = poles[:, 0]
        for w in fake: assert np.min(np.abs(ws - w)) > TOL, (q, w, poles)
        for w in true: assert np.min(np.abs(ws - w)) < TOL, (q, w, poles)
        assert np.max(np.abs(poles[ws < EMAX, 1])) < SHARP, (q, poles)
        if q == [0., 0., 0.]: assert np.min(ws) < 2*DELTA  # Goldstone


def test_method_rpa_takes_channel():
    """A non-onsite interaction sends method="rpa" through the pair-basis
    pole search, and channel= goes along with it."""
    h = _neel()
    kw = dict(qpath=[[0., 0., 0.]], energies=GRID, delta=DELTA, nk=NK)
    def n096(**extra):
        qs, ws, gs = h.get_magnon_bands(method="rpa", **kw, **extra)
        return np.sum((np.abs(ws - 0.960) < TOL) & (np.abs(gs) < SHARP))
    assert n096(channel="all") == 4
    assert n096() == 0
    with pytest.raises(ValueError, match="'channel'"):  # listed as accepted
        h.get_magnon_bands(method="rpa", chanel="all", **kw)
