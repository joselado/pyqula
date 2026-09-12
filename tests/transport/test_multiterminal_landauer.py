"""multiterminal.Device.transmission on a perfectly matched chain.

The oracle is the analytic Landauer limit rather than a recorded number:
a ballistic one-dimensional chain whose leads and central region are the
same lattice with the same hopping has exactly one open channel and no
scattering, so its transmission is 1 everywhere inside the band. A device
built through Device.biterminal never got there -- the lead-to-central
coupling came back square (neighbor.parametric_hopping allocated
(len(r2),len(r2)) rows), so Lead.get_selfenergy's dagger(t)@gr@t failed
on the core dimension -- and landauer_matrix itself was still written
against the removed numpy.matrix API (`.I` and elementwise `*`).
"""
import numpy as np
import pytest

from pyqula import geometry, multiterminal, neighbor


def _chain_cell(x):
    """A one-site chain geometry sitting at position x"""
    g = geometry.chain()
    g.r = np.array([[x, 0., 0.]])
    g.r2xyz()
    return g


def _chain_device(ncentral=4, disorder=0.0):
    """Perfect chain: ncentral sites, one-site leads at either end, all
    the hoppings equal by construction (they come from the distances)"""
    gc = geometry.chain().supercell(ncentral) # central part
    xs = gc.r[:, 0]
    d = multiterminal.Device()
    d.biterminal(right_g=_chain_cell(max(xs)+1.), left_g=_chain_cell(min(xs)-1.),
                  central_g=gc, disorder=disorder)
    return d


def test_the_lead_to_central_coupling_is_rectangular():
    """parametric_hopping(r1,r2) is a hopping from the r1 sites to the r2
    sites, so it has len(r1) rows and len(r2) columns"""
    fun = lambda a, b: 1.0 if 0.7 < (a-b).dot(a-b) < 1.3 else 0.0
    r1 = np.array([[0., 0., 0.], [1., 0., 0.]])
    r2 = np.array([[0., 0., 0.], [1., 0., 0.], [2., 0., 0.], [3., 0., 0.]])
    m = neighbor.parametric_hopping(r1, r2, fun)
    assert m.shape == (len(r1), len(r2))
    ms = neighbor.parametric_hopping(r1, r2, fun, is_sparse=True)
    assert ms.shape == (len(r1), len(r2))
    assert np.allclose(np.array(ms.todense()), m)
    # and the other way round, which used to raise an IndexError
    mt = neighbor.parametric_hopping(r2, r1, fun)
    assert mt.shape == (len(r2), len(r1))
    assert np.allclose(mt, m.T)


def test_ballistic_chain_transmits_one_channel_perfectly():
    d = _chain_device()
    for energy in [-1.0, 0.0, 0.5, 1.3]:
        t = d.transmission(energy=energy)[0]
        assert abs(t - 1.) < 1e-3, (energy, t)


def test_transmission_is_reciprocal_and_bounded_by_the_channel_count():
    """T(0->1)==T(1->0) for any device, and a single-channel lead cannot
    transmit more than one channel however disordered the center is."""
    np.random.seed(1)
    d = _chain_device(ncentral=6, disorder=1.0)
    (tlr, trl) = multiterminal.landauer(d, 0.3, ij=[(0, 1), (1, 0)])
    assert abs(tlr - trl) < 1e-8
    assert -1e-8 < tlr < 1.+1e-8
