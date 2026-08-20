"""The exciton band structure E_X(Q): one BSE per q-point along a path.

The finite-Q physics itself is covered by the supercell-folding test; what
is checked here is the q-path wrapper on top of it -- that it solves the
same problem a direct fixed-Q call solves, and that the dispersion it
returns respects the symmetry of the underlying model."""
import numpy as np
import pytest

from pyqula.bsetk.interaction import density_interaction
from testutils import gapped_ionic_chain

NK = 6
# nv=nc=2, not 1: the chain is spinful with no spin-orbit coupling, so
# every band is two-fold degenerate and an odd window would cut a
# multiplet in half (see select_bands' degeneracy warning)
OPTS = dict(nk=NK, nv=2, nc=2, tda=True)


def _system():
    h = gapped_ionic_chain()
    return h, density_interaction(h, U=1.0, V1=0.5)


def test_bands_match_a_direct_fixed_Q_solve():
    """Every q-point of the path must reproduce, exactly, what a single
    get_exciton_energies call at that same Q gives."""
    h, W = _system()
    qpath = [[0., 0., 0.], [0.15, 0., 0.], [0.4, 0., 0.]]
    qs, es = h.get_exciton_bands(V=W, qpath=qpath, n=3, **OPTS)
    assert len(qs) == len(es) == 3 * len(qpath) # n excitons at every q
    for iq, q in enumerate(qpath): # loop over the path
        direct = h.get_exciton_energies(V=W, Q=q, n=3, **OPTS)
        assert np.max(np.abs(es[qs == iq] - direct)) < 1e-12


def test_exciton_dispersion_is_even_in_Q():
    """The model is time-reversal symmetric, so the exciton at +Q and the
    one at -Q are degenerate: E_X(Q) = E_X(-Q). This is a check on the
    dispersion itself, not just on the plumbing -- it fails if the finite-Q
    pair basis is built with the wrong sign anywhere, and it also fails
    (by ~0.1 here, not by roundoff) if the band window splits the spin
    degeneracy, which is what select_bands warns about."""
    h, W = _system()
    qs_ = [0.1, 0.35]
    qpath = [[q, 0., 0.] for q in qs_] + [[-q, 0., 0.] for q in qs_]
    qs, es = h.get_exciton_bands(V=W, qpath=qpath, n=4, **OPTS)
    for i in range(len(qs_)): # loop over the +-Q pairs
        plus, minus = es[qs == i], es[qs == i + len(qs_)]
        assert np.max(np.abs(plus - minus)) < 1e-10


def test_zero_kernel_bands_are_the_bare_transitions():
    """With the kernel off the exciton bands must collapse onto the
    electron-hole continuum of the mean field at every q-point."""
    h, W = _system()
    qpath = [[0.2, 0., 0.], [0.5, 0., 0.]]
    qs, es = h.get_exciton_bands(V=W, qpath=qpath, kernel="none", n=2,
                                 **OPTS)
    for iq, q in enumerate(qpath): # loop over the path
        b = h.get_bse(V=W, Q=q, kernel="none", **OPTS)
        bare = np.sort(b.pairs.dE)[0:2] # lowest bare transitions
        assert np.max(np.abs(np.sort(es[qs == iq].real) - bare)) < 1e-10


def test_zero_dimensional_is_rejected():
    """A 0d system has no center-of-mass momentum to disperse in"""
    from pyqula import geometry
    h = geometry.chain().supercell(4).get_hamiltonian()
    h.dimensionality = 0
    with pytest.raises(ValueError):
        h.get_exciton_bands(nq=2)


def test_odd_band_window_on_a_degenerate_model_warns():
    """Cutting a two-fold degenerate multiplet in half keeps an arbitrary
    state out of the degenerate subspace, so it must not pass silently."""
    h, W = _system()
    with pytest.warns(UserWarning, match="degenerate multiplet"):
        h.get_exciton_energies(V=W, nk=4, nv=1, nc=1, tda=True)


def test_even_band_window_does_not_warn():
    """The same calculation with the whole multiplet inside the window is
    well defined and must stay quiet."""
    h, W = _system()
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error") # any warning fails the test
        h.get_exciton_energies(V=W, nk=4, nv=2, nc=2, tda=True)
