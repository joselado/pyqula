"""Magnons of a METALLIC magnet, and the exact answer they are checked against.

The pair basis needs a well-defined occupied and empty set. For a gapped
reference that is the band window; for a metal it has to be decided per
k-point, which is what PairBasis(metal=True) plus spinflip.occupancy_masks
do. That is the whole extension -- everything downstream already tolerated
a varying number of pairs per k-point, because the flattened arrays carry
a kindex and the kernel indexes the interaction through it.

It matters because the site-basis RPA, which handles metals fine, has no
vertex at all for a neighbour-shell density-density interaction -- so a
ferromagnet ordered by V1 alone was until now covered by neither route.

There is an exact reference for one case, and it is used here rather than
a symmetry argument: for a SATURATED ferromagnet the single-magnon sector
is a two-body problem (one minority electron, one majority hole) with a
separable interaction, so its dispersion solves

    1 = (U/N) sum_k 1/(dE_k - E),  dE_k = e_up(k+q) + U n_dn - e_dn(k)

over the occupied k. That is a ten-line calculation independent of
everything in pyqula, and the TDHF magnon has to reproduce it exactly.
"""
import numpy as np
import pytest
from scipy.optimize import brentq

from pyqula import geometry
from pyqula.bsetk import spinflip
from pyqula.meanfield import VJinteraction

NK = 200  # a 1d Stoner instability at low filling needs a fine mesh
NOCC = 41  # ODD, so the occupied set is symmetric under k -> -k. See below.


def _metallic_ferromagnet(nk=NK, filling=0.1, **kw):
    """A partially polarized ferromagnetic chain. The exchange seed goes on
    the initial GUESS only and never on h itself: a persistent Zeeman term
    breaks SU(2) explicitly, and then there is no Goldstone mode to find
    (measured: a 1e-2 seed field gives a residual of exactly 2e-2, the
    Zeeman gap, which is right rather than a failure)."""
    g = geometry.chain()
    h = g.get_hamiltonian()
    mf = h.copy()
    mf.add_exchange([0., 0., 0.5])
    scf = VJinteraction(h, filling=filling, mf=mf, nk=nk, mix=0.2,
                         maxerror=1e-10, maxite=3000, **kw)
    return scf.hamiltonian


def _exact_saturated_magnon(q, nk=NK, nocc=NOCC, U=3.0, sign=1):
    """The exact one-magnon energy of a saturated Hubbard ferromagnet on a
    chain, from the two-body problem. Independent of pyqula."""
    ks = np.arange(nk)/nk
    eps = lambda k: -2.0*np.cos(2*np.pi*k)
    occ = np.argsort(eps(ks))[:nocc]
    dE = eps(ks[occ] + sign*q) + U*nocc/nk - eps(ks[occ])
    g = lambda w: 1.0 + U*np.sum(1.0/(w - dE))/nk
    return brentq(g, 1e-12, np.min(dE) - 1e-9, xtol=1e-13)


@pytest.mark.slow
def test_goldstone_of_a_metallic_ferromagnet():
    h = _metallic_ferromagnet(U=3.0)
    assert abs(h.get_gap()) < 1e-6  # genuinely gapless
    assert abs(h.get_vev("sz")[0]) > 0.05  # and genuinely magnetic
    assert h.get_goldstone_residual(nk=NK, metal=True) < 1e-10


@pytest.mark.slow
def test_goldstone_of_a_ferromagnet_ordered_by_v1_alone():
    """The case neither route covered before: a metal ordered purely by a
    neighbour-shell density-density interaction. The site-basis RPA has no
    vertex for it at all (its kernel is the identity, smallest eigenvalue
    1.0 where the Goldstone theorem demands 0), and the TDHF route used to
    refuse it for being gapless."""
    h = _metallic_ferromagnet(V1=1.1)
    assert len(h.V) > 1  # non-onsite
    assert abs(h.get_vev("sz")[0]) > 0.05
    assert h.get_goldstone_residual(nk=NK, metal=True) < 1e-10


@pytest.mark.slow
def test_the_metallic_magnon_reproduces_the_exact_two_body_answer():
    """The strongest check available: a closed-form dispersion, computed
    without pyqula, at three momenta. Measured agreement is to 5 decimals
    (0.00173, 0.01291, 0.07756 at q = 0.02, 0.05, 0.1)."""
    h = _metallic_ferromagnet(U=3.0, filling=NOCC/(2.0*NK))
    # saturated means the minority band is empty, which is exactly the
    # statement that there is no de-excitation half to the problem -- and
    # it is that, not the size of the moment, that the two-body formula
    # assumes
    assert spinflip.magnon_matrix(h, Q=[0.02, 0., 0.], nk=NK,
                                   metal=True).n2 == 0
    for q in (0.02, 0.05, 0.1):
        es, w = spinflip.magnon_spectrum(h, Q=[q, 0., 0.], nk=NK, metal=True)
        tdhf = es[np.argmax(w)].real
        assert abs(tdhf - _exact_saturated_magnon(q)) < 1e-5


@pytest.mark.slow
def test_an_asymmetric_occupied_set_splits_the_plus_and_minus_q_magnons():
    """A trap worth pinning, because it looks like a bug in whichever
    method you check second. E(q) has to be even in q, but on a finite mesh
    that holds only if the OCCUPIED SET is symmetric under k -> -k. With an
    even number of occupied points around k=0 it is not, and the +q and -q
    magnons genuinely differ: at filling 0.1 on this mesh (40 occupied
    points) the exact answers are 0.02413 and 0.00559 at q=0.05, and the
    two methods disagree because they weight them differently -- TDHF
    resolves +q, the RPA's (Sx,Sy,Sz) block mixes the two. With an odd
    count they all agree to 5 decimals (the test above).

    So: nothing is wrong, but a metallic magnon dispersion computed on a
    mesh with an asymmetric occupied set is not the one that was wanted."""
    a = _exact_saturated_magnon(0.05, nocc=40, sign=+1)
    b = _exact_saturated_magnon(0.05, nocc=40, sign=-1)
    assert abs(a - b) > 0.01  # asymmetric: genuinely different
    c = _exact_saturated_magnon(0.05, nocc=41, sign=+1)
    d = _exact_saturated_magnon(0.05, nocc=41, sign=-1)
    assert abs(c - d) < 1e-12  # symmetric: identical, as E(q) must be


@pytest.mark.slow
def test_metal_mode_changes_nothing_for_a_gapped_reference():
    """metal=True only replaces a global band window by a per-k occupancy
    filter, and for a gapped state those say the same thing. Bit-identical
    pair counts and Goldstone residual either way, which is what makes it
    safe to leave on when unsure."""
    g = geometry.honeycomb_lattice()
    nk = 6
    h = g.get_hamiltonian().get_mean_field_hamiltonian(
            U=3.0, filling=0.5, mf="antiferro", nk=nk, maxerror=1e-10)
    out = []
    for metal in (False, True):
        p = spinflip.magnon_matrix(h, Q=[0., 0., 0.], nk=nk, metal=metal)
        out.append((p.n1, p.n2,
                    h.get_goldstone_residual(nk=nk, metal=metal)))
    assert out[0][0] == out[1][0] and out[0][1] == out[1][1]
    assert abs(out[0][2] - out[1][2]) < 1e-15


@pytest.mark.slow
def test_the_spectral_weight_is_what_finds_the_magnon_in_a_metal():
    """In an insulator the magnon is the lowest mode. In a metal it sits
    inside the Stoner continuum, so it has to be found by how much of the
    spin generator it carries. At Q=0 that is all of it, and it drains
    away into the continuum as Q grows (Landau damping): measured 1.00,
    0.96, 0.78, 0.44 at Q = 0, 0.02, 0.05, 0.1."""
    h = _metallic_ferromagnet(V1=1.5)
    es, w = spinflip.magnon_spectrum(h, Q=[0., 0., 0.], nk=NK, metal=True)
    i = np.argmax(w)
    assert w[i] > 0.99  # the Goldstone mode carries everything at Q=0
    assert abs(es[i].real) < 1e-8
    assert abs(np.sum(w) - 1.0) < 1e-10  # weights are normalized
    es2, w2 = spinflip.magnon_spectrum(h, Q=[0.1, 0., 0.], nk=NK, metal=True)
    assert np.max(w2) < np.max(w)  # ... and less of it at finite Q
