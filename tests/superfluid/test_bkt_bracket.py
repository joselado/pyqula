import numpy as np
import pytest

from pyqula import geometry
from pyqula.sctk import superfluidweight

# bkt_temperature bisects T - (pi/8) D_s(T) on [0,tmax], tmax=(pi/8) D_s(0)
# when not given. When the stiffness is still above the line at tmax, which
# needs a D_s that rises with T or a tmax given too low, it used to return
# tmax, a bound, as if it were the root. The stiffness is replaced here by a
# closed form, so the Nelson-Kosterlitz crossing is known exactly.


def _fake_stiffness(g):
    """superfluid_weight returning an isotropic tensor whose scalar
    stiffness is (8/pi) g(T), so that the line T = (pi/8) D_s(T) reads
    T = g(T)"""
    def superfluid_weight(h, ks=None, T=0., **kwargs):
        return 8./np.pi*g(T)*np.identity(2)
    return superfluid_weight


def _h2d():
    return geometry.square_lattice().get_hamiltonian()


def test_a_stiffness_rising_with_T_is_bracketed(monkeypatch):
    """g(T) = 1 + T - T^2/5 crosses T = g(T) at sqrt(5), above the naive
    bound g(0) = 1, where the stiffness is still above the line"""
    monkeypatch.setattr(superfluidweight, "superfluid_weight",
            _fake_stiffness(lambda T: 1. + T - T**2/5.))
    tb = superfluidweight.bkt_temperature(_h2d(), nk=2, tol=1e-10)
    assert abs(tb - np.sqrt(5.)) < 1e-6


def test_a_tmax_given_too_low_is_bracketed(monkeypatch):
    """a falling stiffness, g(T) = 1 - T/2, crosses at 2/3; a tmax of 0.1
    is below that and must not come back as the answer"""
    monkeypatch.setattr(superfluidweight, "superfluid_weight",
            _fake_stiffness(lambda T: 1. - T/2.))
    tb = superfluidweight.bkt_temperature(_h2d(), nk=2, tmax=0.1, tol=1e-10)
    assert abs(tb - 2./3.) < 1e-6


def test_a_stiffness_that_never_crosses_is_refused(monkeypatch):
    monkeypatch.setattr(superfluidweight, "superfluid_weight",
            _fake_stiffness(lambda T: 1. + 2.*T))
    with pytest.raises(ValueError, match="still above"):
        superfluidweight.bkt_temperature(_h2d(), nk=2, maxexpand=5)
