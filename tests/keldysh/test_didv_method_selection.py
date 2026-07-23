import pytest

from pyqula import geometry
from pyqula import heterostructures


def _junction(delta_left, delta_right, transparency=0.5):
    g = geometry.chain()
    h1 = g.get_hamiltonian()
    h2 = g.get_hamiltonian()
    if delta_left: h1.add_swave(delta_left)
    if delta_right: h2.add_swave(delta_right)
    HT = heterostructures.build(h1, h2)
    HT.set_coupling(transparency)
    HT.delta = 1e-4
    return HT


def test_auto_method_matches_smatrix_when_one_lead_is_normal():
    """With only one lead superconducting, Heterostructure.didv() should
    keep using the existing scattering-matrix/BdG formula by default --
    Floquet-Keldysh only applies when both leads are superconducting."""
    HT = _junction(0., 0.3)
    Gauto = HT.didv(energy=1e-4)
    Gsmatrix = HT.didv(energy=1e-4, method="smatrix")
    assert Gauto == Gsmatrix


def test_auto_method_matches_explicit_keldysh_when_both_leads_are_sc():
    """With both leads superconducting, the default ("auto") method must
    route through the Floquet-Keldysh dI/dV, matching an explicit
    method="keldysh" call exactly (same code path)."""
    HT = _junction(0.3, 0.3, transparency=0.3)
    kwargs = dict(nmax=8, nmax_max=20, tol=1e-3)
    Gauto = HT.didv(energy=0.0, **kwargs)
    Gkeldysh = HT.didv(energy=0.0, method="keldysh", **kwargs)
    assert Gauto == Gkeldysh


def test_keldysh_didv_returns_a_derivative_not_a_raw_current():
    """keldysh_didv must return dI/dV (a finite-difference derivative of
    the Floquet-Keldysh DC current), not the current I(V) itself -- for a
    normal-superconductor junction this zero-bias slope should reproduce
    the existing equilibrium Andreev conductance from the smatrix/BdG
    formula (same check as test_andreev_linear_response, but going through
    the didv(method="keldysh") entry point instead of calling
    get_dc_current by hand)."""
    delta, transparency = 0.3, 0.7
    HT = _junction(0., delta, transparency=transparency)
    Gref = HT.didv(energy=1e-4)  # equilibrium smatrix/BdG conductance
    Gkeldysh = HT.didv(energy=0.0, method="keldysh",
                        dv=0.02*delta, nmax=10, nmax_max=50, tol=1e-4)
    assert abs(Gkeldysh-Gref) < 0.05*max(Gref, 1e-8)


def test_unknown_didv_method_raises():
    HT = _junction(0., 0.3)
    with pytest.raises(ValueError):
        HT.didv(energy=1e-4, method="not-a-method")
