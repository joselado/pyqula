import numpy as np
import pytest
from scipy.integrate import quad

from pyqula import geometry, heterostructures
from pyqula.keldyshtk import quadrature as kq
from pyqula.keldyshtk.current import dc_current
from pyqula.transporttk.didv import keldysh_didv

# Tests for keldyshtk.quadrature.adaptive_quad_batch, the batched-integrand
# adaptive Gauss-Kronrod rule that replaced dc_current's per-node
# scipy.integrate.quad callback on the default `quadrature="adaptive"`
# path. Two things need checking, and they are different in kind:
#
#   1. the quadrature itself, against scipy.integrate.quad on analytic
#      integrands where an essentially-exact reference is available -- both
#      the value (accuracy) and the number of integrand evaluations (node
#      economy, the property that made "adaptive" worth keeping as the
#      default over the fixed composite grid in the first place);
#   2. dc_current's own output, against the previous implementation, still
#      reachable as `quadrature="adaptive_scipy"` -- this is NOT expected to
#      be bit-identical (a different adaptive path visits different nodes),
#      only accurate to the same requested tolerance, so it is checked
#      through the physics-level quantities dc_current and keldysh_didv
#      return, including the finite-difference dI/dV where a quadrature
#      error is amplified by catastrophic cancellation.


def _sc_sc_junction(delta_sc, transparency, ht_delta=1e-4):
    """Same junction shapes as tests/keldysh/test_batched_fixed_quadrature.py
    and as the cases benchmarked in dc_current's own docstring."""
    h1 = geometry.chain().get_hamiltonian(); h1.shift_fermi(1.); h1.add_swave(delta_sc)
    h2 = geometry.chain().get_hamiltonian(); h2.shift_fermi(1.); h2.add_swave(delta_sc)
    HT = heterostructures.build(h1, h2)
    HT.set_coupling(transparency)
    HT.delta = ht_delta
    return HT


def _normal_normal_junction(transparency):
    h1 = geometry.chain().get_hamiltonian(); h1.turn_nambu()
    h2 = geometry.chain().get_hamiltonian(); h2.turn_nambu()
    HT = heterostructures.build(h1, h2)
    HT.set_coupling(transparency)
    HT.delta = 1e-3
    return HT


def test_gauss_kronrod_tables_are_self_consistent():
    """The hardcoded DQK21 tables must expand into a rule whose Gauss
    subset is numpy's own leggauss(10) and which integrates the monomials
    each rule is exact for. `_validate_rule` asserts this at import time
    (so a mistyped constant fails loudly rather than silently degrading the
    quadrature); this test makes that check part of the suite too."""
    kq._validate_rule()
    assert len(kq._X) == 21
    assert np.count_nonzero(kq._WG_FULL) == 10
    assert abs(kq._WGK_FULL.sum() - 2.) < 1e-14
    assert abs(kq._WG_FULL.sum() - 2.) < 1e-14


ANALYTIC = [
    ("smooth", lambda x: np.exp(-x)*np.cos(3*x), 0., 1.),
    ("polynomial", lambda x: x**5 - 2*x**2 + 1, 0., 2.),
    ("narrow_peak", lambda x: 1e-4/((x-0.3)**2 + 1e-8), 0., 1.),
    ("sqrt_kink", lambda x: np.sqrt(np.abs(x-0.5)), 0., 1.),
    ("oscillatory", lambda x: np.sin(50*x)*np.exp(-x), 0., 1.),
]


@pytest.mark.parametrize("name,f,a,b", ANALYTIC)
def test_matches_scipy_quad_on_analytic_integrands(name, f, a, b):
    """Same requested tolerance, same answer: adaptive_quad_batch must hit
    epsrel=1e-3 against a tight reference on integrands spanning the shapes
    the Keldysh quasienergy integral actually has (smooth, a narrow
    gap-edge-like peak, a kink, an oscillation)."""
    ref, _ = quad(lambda x: f(np.array([x]))[0], a, b,
                  limit=200, epsrel=1e-12, epsabs=1e-14)
    got = kq.adaptive_quad_batch(f, a, b, epsrel=1e-3)
    assert abs(got-ref) <= 1e-3*abs(ref) + 1e-12


@pytest.mark.parametrize("name,f,a,b", ANALYTIC)
def test_node_economy_matches_scipy_quad(name, f, a, b):
    """The whole reason `"adaptive"` beats the fixed composite grid on easy
    integrands is that it stops after 21 nodes when 21 nodes are enough.
    Batching the evaluation must not cost that: the batched rule may not
    use materially more integrand evaluations than scipy's own adaptive
    quadrature at the same tolerance (a small slack is allowed since the
    two refine along different paths -- bisecting a set of worst panels per
    round vs. the single worst panel per step)."""
    nscipy = [0]

    def fscalar(x):
        nscipy[0] += 1
        return f(np.array([x]))[0]

    quad(fscalar, a, b, limit=50, epsrel=1e-3)
    _, _, info = kq.adaptive_quad_batch(f, a, b, epsrel=1e-3, limit=50,
                                        full_output=True)
    assert info["converged"]
    assert info["nevals"] <= 1.3*nscipy[0] + 21


def test_evaluates_in_batches_not_one_node_at_a_time():
    """The point of the rule: the integrand sees whole arrays of nodes, and
    the number of separate calls is a small number of refinement rounds
    rather than one per node -- that ratio is exactly the Python/numba
    dispatch overhead removed from dc_current's hot loop."""
    calls = []

    def f(x):
        assert isinstance(x, np.ndarray) and x.ndim == 1
        calls.append(x.size)
        return 1e-4/((x-0.3)**2 + 1e-8)

    _, _, info = kq.adaptive_quad_batch(f, 0., 1., epsrel=1e-3,
                                        full_output=True)
    assert len(calls) == info["nrounds"]
    assert sum(calls) == info["nevals"]
    assert min(calls) >= len(kq._X)  # every call is a whole panel, at least
    assert info["nevals"] > 21  # this integrand genuinely needs refinement
    assert len(calls) < info["nevals"]/10  # rounds are far fewer than nodes


def test_smooth_integrand_converges_on_the_first_batch():
    """An integrand with no structure must cost exactly one 21-node round --
    the node-economy floor a fixed composite grid cannot reach."""
    _, _, info = kq.adaptive_quad_batch(lambda x: np.exp(-x), 0., 1.,
                                        epsrel=1e-3, full_output=True)
    assert info["nrounds"] == 1 and info["nevals"] == 21


def test_panel_limit_is_respected_and_reported():
    """A deliberately unresolvable integrand must stop at `limit` panels and
    report converged=False rather than refining forever -- dc_current owns
    the policy for what to do about it (as it did with scipy's own
    non-convergence)."""
    f = lambda x: np.sin(1e7*x)/(np.abs(x-0.5)+1e-12)
    _, _, info = kq.adaptive_quad_batch(f, 0., 1., epsrel=1e-12, limit=20,
                                        full_output=True)
    assert not info["converged"]
    assert info["npanels"] <= 20


@pytest.mark.parametrize("delta_sc,transparency,voltage", [
    (0.3, 0.5, 0.031),   # deep-subgap representative case
    (0.1, 0.3, 0.15),    # worst-accuracy point of the fixed-quad sweep
])
def test_dc_current_matches_previous_scipy_quadrature(delta_sc, transparency,
                                                       voltage):
    """dc_current's default path must return the same current as the
    scipy.integrate.quad implementation it replaced, to within the 1e-3
    relative tolerance both quadratures are asked for."""
    ht = _sc_sc_junction(delta_sc, transparency)
    Inew = dc_current(ht, voltage, nmax_max=40)
    Iold = dc_current(ht, voltage, nmax_max=40, quadrature="adaptive_scipy")
    assert abs(Inew-Iold) <= 2e-3*max(abs(Iold), 1e-12)


def test_dc_current_matches_previous_scipy_quadrature_normal_junction():
    """Same check on a normal-normal junction, whose integrand has no
    gap-edge singularity at all -- the case where the two adaptive paths
    take the fewest refinement rounds and any systematic difference would
    show up undiluted."""
    ht = _normal_normal_junction(0.6)
    Inew = dc_current(ht, 0.3, nmax_max=20)
    Iold = dc_current(ht, 0.3, nmax_max=20, quadrature="adaptive_scipy")
    assert abs(Inew-Iold) <= 2e-3*max(abs(Iold), 1e-12)


def test_dc_current_matches_previous_scipy_quadrature_finite_temperature():
    """The finite-temperature integrand (native Keldysh temperature
    broadening, the default path for finite_T_didv/kappa) through the new
    quadrature."""
    ht = _sc_sc_junction(0.1, 0.5)
    Inew = dc_current(ht, 0.3, nmax_max=20, temperature=0.02)
    Iold = dc_current(ht, 0.3, nmax_max=20, temperature=0.02,
                      quadrature="adaptive_scipy")
    assert abs(Inew-Iold) <= 2e-3*max(abs(Iold), 1e-12)


def test_keldysh_didv_matches_previous_scipy_quadrature():
    """The discriminating check: keldysh_didv differences two dc_current
    calls whose currents nearly cancel, amplifying any per-branch
    quadrature error by |Ip|/(Ip-Im) (~12x on this case, see
    documentation/keldysh_aaa_selfenergy_accuracy_plan.md). A quadrature
    change that looks fine at 1e-4 on the current itself can still move
    dI/dV, so it is checked directly."""
    ht = _sc_sc_junction(0.3, 0.5)
    kw = dict(nmax_max=40, delta=ht.delta)
    gnew = keldysh_didv(ht, voltage=0.18, **kw)
    gold = keldysh_didv(ht, voltage=0.18, quadrature="adaptive_scipy", **kw)
    assert abs(gnew-gold) <= 1e-2*max(abs(gold), 1e-12)
