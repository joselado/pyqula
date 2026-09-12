"""Tests for chitk.magneticresponse.rkky_generator.

The generator's Bloch-phase array depends only on R, not on the pair of
sites (ii,jj) the closure is evaluated at, and the k-list it is built
from repeats every distinct k-point once per eigenstate. Those are
invariants of the formula, not of a particular implementation, so they
are what is asserted here: the result must not depend on how many times
geometry.bloch_phase is called, nor on the order in which the different
R are asked for."""
import numpy as np

from pyqula import geometry
from pyqula.chitk.magneticresponse import rkky_generator, rkky_loop


def _model():
    g = geometry.honeycomb_lattice()
    h = g.get_hamiltonian()
    h.add_zeeman([0., 0., 0.2])
    return h


def _rkky_reference(h, R, ii, jj, nk):
    """Independent transcription of the RKKY summation, with the Bloch
    phase evaluated one eigenstate at a time through
    geometry.bloch_phase. Whatever the generator does internally, it has
    to reproduce this."""
    h = h.copy()
    h.remove_spin()
    es, ws, ks = h.get_eigenvectors(kpoints=True, nk=nk)
    delta = 1./nk
    d1s = np.array([w[ii] for w in ws])
    d2s = np.array([w[jj] for w in ws])
    phis = np.array([h.geometry.bloch_phase(np.array(R), k) for k in ks])
    fs = (np.tanh(es/delta) + 1.0)/2.
    nw = len(es)/len(h.geometry.r)
    return rkky_loop(es, phis, fs, d1s, d2s, delta)/(nw**2)


def test_rkky_generator_matches_the_per_state_bloch_phase_reference():
    """The vectorized Bloch phase must reproduce the per-state
    geometry.bloch_phase convention exactly (up to the reassociation of
    the 2*pi factor), for several R and several site pairs."""
    h = _model()
    nk = 8
    get = rkky_generator(h, nk=nk)
    for R in ([0, 0, 0], [1, 0, 0], [0, 1, 0], [1, -2, 0]):
        for (ii, jj) in [(0, 0), (1, 0), (0, 1), (1, 1)]:
            ref = _rkky_reference(h, R, ii, jj, nk)
            assert np.allclose(get(R, ii, jj), ref, rtol=1e-10, atol=1e-12), \
                "rkky_generator disagrees with the reference at R=%s"%str(R)


def test_bloch_phase_is_not_recomputed_per_site_pair():
    """The Bloch phases depend only on R, and every distinct k-point
    appears once per eigenstate in the k-list get_eigenvectors returns,
    so a whole block of (ii,jj) evaluations at one R needs at most one
    Bloch-phase construction per distinct R -- not one per state per
    evaluation."""
    h = _model()
    nk = 8
    hs = h.copy()
    hs.remove_spin()
    calls = {"n": 0}
    orig = hs.geometry.bloch_phase
    def counted(d, k):
        calls["n"] += 1
        return orig(d, k)
    hs.geometry.bloch_phase = counted
    get = rkky_generator(hs, nk=nk)
    calls["n"] = 0
    Rs = [[1, 0, 0], [0, 1, 0]]
    for R in Rs:
        for (ii, jj) in [(0, 0), (1, 0), (0, 1)]:
            get(R, ii, jj)
    assert calls["n"] <= len(Rs), \
        "geometry.bloch_phase called %d times for %d distinct R"%(
                calls["n"], len(Rs))


def test_rkky_generator_does_not_depend_on_the_order_of_the_R():
    """Whatever is memoized on R must be keyed on R: asking for R1, then
    R2, then R1 again has to give the same answer for R1 both times, and
    a different one for R2."""
    h = _model()
    get = rkky_generator(h, nk=8)
    R1, R2 = [1, 0, 0], [0, 1, 0]
    a1 = get(R1, 1, 0)
    b = get(R2, 1, 0)
    a2 = get(R1, 1, 0)
    assert a1 == a2, "the R=%s result changed after evaluating R=%s"%(
            str(R1), str(R2))
    assert not np.isclose(a1, b), \
        "R=%s and R=%s gave the same RKKY interaction"%(str(R1), str(R2))


def test_rkky_generator_works_in_one_dimension():
    """The 1d Bloch-phase convention (only the first component of R and
    k enter) must be honoured too."""
    h = geometry.chain().get_hamiltonian()
    h.add_zeeman([0., 0., 0.2])
    get = rkky_generator(h, nk=20)
    for R in ([0, 0, 0], [1, 0, 0], [3, 0, 0]):
        assert np.allclose(get(R, 0, 0), _rkky_reference(h, R, 0, 0, 20),
                rtol=1e-10, atol=1e-12)
