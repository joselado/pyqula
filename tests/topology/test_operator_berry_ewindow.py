"""topology.operator_berry's ewindow= must actually restrict the sum.

The keyword has the same meaning as on bandstructure.get_bands -- a
callable taking a band energy and returning whether to keep that band --
and it used to be accepted and then never referenced, so an energy window
that keeps no band at all returned exactly the full-band answer.

The invariants asserted here need no recorded number: a window that keeps
everything must reproduce the unwindowed value bit for bit, a window that
keeps nothing must give exactly zero, and any window in between must equal
the sum of operator_berry_bands over the bands it selects -- the
band-resolved sibling being an independent way of reaching the same number.
"""
import numpy as np
import pytest

from pyqula import geometry, topology

K = [0.2, 0.3]


def _hamiltonian():
    g = geometry.honeycomb_lattice()
    h = g.get_hamiltonian(has_spin=True)
    h.add_kane_mele(0.1)
    h.add_sublattice_imbalance(0.2)
    return h


def test_ewindow_that_keeps_nothing_gives_zero():
    h = _hamiltonian()
    assert topology.operator_berry(h, k=K, ewindow=lambda e: False) == 0.0
    # and the unwindowed value it used to return instead is not zero
    assert np.abs(topology.operator_berry(h, k=K)) > 1e-3


def test_ewindow_that_keeps_everything_is_the_unwindowed_value():
    h = _hamiltonian()
    full = topology.operator_berry(h, k=K)
    windowed = topology.operator_berry(h, k=K, ewindow=lambda e: True)
    assert np.isclose(windowed, full, rtol=1e-12, atol=1e-14)


def test_ewindow_selects_the_same_bands_as_operator_berry_bands():
    """A genuine, partial window, against the band-resolved path."""
    h = _hamiltonian()
    (es, bs) = topology.operator_berry_bands(h, k=K)
    es = np.array(es); bs = np.array(bs)
    occ = es[es <= 0.]
    cut = 0.5*(occ[-1] + occ[-2]) # between the two topmost occupied bands
    fwin = lambda e: e > cut
    ref = np.sum(bs[(es <= 0.)*(es > cut)])
    assert np.abs(ref) > 1e-6 # the window must actually select something
    assert not np.isclose(ref, np.sum(bs[es <= 0.])) # and drop something
    got = topology.operator_berry(h, k=K, ewindow=fwin)
    assert np.isclose(got, ref, rtol=1e-10, atol=1e-12)


def test_non_callable_ewindow_raises():
    h = _hamiltonian()
    with pytest.raises(TypeError):
        topology.operator_berry(h, k=K, ewindow=(-1.0, 0.0))
