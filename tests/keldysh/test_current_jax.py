import numpy as np
import pytest

jax = pytest.importorskip("jax")

from pyqula import geometry
from pyqula import heterostructures
from pyqula.transporttk.localprobe import LocalProbe
from pyqula.keldyshtk.current import dc_current, _prepare_bias_target, build_selfenergy_aaa
from pyqula.keldyshtk.current_jax import JaxKeldyshCurrent


def _direct_didv(ht, voltage, nmax, delta=None, dv=None):
    """Same central finite difference keldysh_didv uses, but at a FIXED
    nmax (nmax==nmax_max, so dc_current's adaptive loop never runs) --
    the fair, matched-truncation baseline JaxKeldyshCurrent (also fixed
    nmax) is compared against throughout this file."""
    htb = _prepare_bias_target(ht)
    if delta is None: delta = htb.delta
    if dv is None: dv = max(abs(voltage)*1e-2, 1e-3)
    shared = build_selfenergy_aaa(htb, abs(voltage)+dv, nmax, delta=delta)
    Ip = dc_current(ht, voltage+dv, nmax=nmax, nmax_max=nmax, delta=delta, selfenergy_qtci=shared)
    Im = dc_current(ht, voltage-dv, nmax=nmax, nmax_max=nmax, delta=delta, selfenergy_qtci=shared)
    return (Ip-Im)/(2*dv)


def _normal_normal_junction():
    """Plain two-lead junction with trivial (zero) pairing -- dc_current
    is exactly validated on this shape against a non-Floquet static-bias
    reference in test_normal_junction_gauge_invariance.py, so it is a
    strong, independent correctness check for JaxKeldyshCurrent too (not
    just "matches dc_current's own possibly-imperfect fixed-nmax value").
    shift_fermi avoids a coincidental self-energy singularity exactly at
    E=0 for this bare chain geometry (see tests/keldysh/
    test_andreev_linear_response.py and this module's own docstring)."""
    h0 = geometry.chain().get_hamiltonian()
    h0.shift_fermi(0.6)
    h1 = h0.copy(); h1.turn_nambu()
    h2 = h1.copy()
    HT = heterostructures.build(h1.copy(), h2.copy())
    HT.set_coupling(0.6)
    HT.delta = 1e-4
    return HT


def _sc_probe_localprobe():
    """SC probe + SC sample LocalProbe -- the harder, resonance-rich
    system this module was developed against (see its own docstring for
    the numerical pitfalls found and fixed using exactly this system)."""
    h = geometry.chain().get_hamiltonian(); h.shift_fermi(1.); h.add_swave(0.1)
    lead = geometry.chain().get_hamiltonian(); lead.shift_fermi(1.); lead.add_swave(0.1)
    lp = LocalProbe(h, lead=lead, delta=1e-3)
    lp.T = 0.3
    return lp


@pytest.fixture(scope="module")
def normal_jkc():
    """Built once (module-scoped) since JIT compilation is the expensive
    part -- every test in this file that uses it reuses the same compiled
    functions."""
    return JaxKeldyshCurrent(_normal_normal_junction(), nmax=4, vmax=0.35, gl_order=200)


@pytest.fixture(scope="module")
def sc_probe_jkc():
    """gl_order is given explicitly (validated separately, see this
    module's own docstring) to skip the adaptive search's several extra
    compiles here -- test_gl_order_search_is_exercised below covers that
    code path directly, at cheaper settings."""
    return JaxKeldyshCurrent(_sc_probe_localprobe(), nmax=8, vmax=0.25, gl_order=3200)


def test_current_matches_direct_method_on_normal_junction(normal_jkc):
    ht = _normal_normal_junction()
    for V in (0.2, 0.3):
        ref = dc_current(ht, V, nmax=4, nmax_max=4, delta=ht.delta)
        got = normal_jkc.current(V)
        assert abs(got-ref) < 5e-3*max(abs(ref), 1e-8)


def test_didv_matches_direct_method_on_normal_junction(normal_jkc):
    """The boundary term (see this module's docstring for why it is
    required at all) must be included correctly: without it this exact
    check was off by two orders of magnitude and the wrong sign."""
    ht = _normal_normal_junction()
    for V in (0.2, 0.3):
        ref = _direct_didv(ht, V, nmax=4)
        got = normal_jkc.didv(V)
        assert abs(got-ref) < 1e-2*max(abs(ref), 1e-8)


def test_odd_and_even_symmetry_on_normal_junction(normal_jkc):
    """I(-V)=-I(V) and dI/dV(-V)=dI/dV(V) are exact physical symmetries;
    this system/nmax is well-behaved enough (unlike the SC-probe one
    below) that they should hold tightly, not just approximately."""
    for V in (0.2, 0.3):
        Ip, Im = normal_jkc.current(V), normal_jkc.current(-V)
        assert abs(Ip+Im) < 1e-4*max(abs(Ip), 1e-8)
        dp, dm = normal_jkc.didv(V), normal_jkc.didv(-V)
        assert abs(dp-dm) < 1e-2*max(abs(dp), 1e-8)


def test_current_matches_direct_method_on_sc_probe_localprobe(sc_probe_jkc):
    lp = _sc_probe_localprobe()
    ref = dc_current(lp, 0.25, nmax=8, nmax_max=8, delta=lp.delta)
    got = sc_probe_jkc.current(0.25)
    assert abs(got-ref) < 1e-2*max(abs(ref), 1e-8)


def test_didv_matches_direct_method_on_sc_probe_localprobe_within_documented_tolerance(sc_probe_jkc):
    """A much looser tolerance than the normal-junction case: this system
    is known (see this module's docstring, and keldyshtk.current.
    dc_current's own docstring on non-monotonic nmax convergence) to be
    quantitatively delicate at a small, fixed nmax -- confirmed by
    checking dc_current against ITSELF (not this module) for I(-V) vs
    -I(V) at this same nmax, which shows the same ~40% deviation this
    module also shows, ruling out a JAX-specific bug. 10% is loose enough
    to not be sensitive to that known effect while still catching a
    genuinely broken implementation (which showed >100% or wrong-sign
    errors during development, not a borderline few-percent miss)."""
    lp = _sc_probe_localprobe()
    ref = _direct_didv(lp, 0.25, nmax=8)
    got = sc_probe_jkc.didv(0.25)
    assert abs(got-ref) < 0.10*max(abs(ref), 1e-8)


def test_negative_voltage_current_matches_direct_method(sc_probe_jkc):
    lp = _sc_probe_localprobe()
    ref = dc_current(lp, -0.25, nmax=8, nmax_max=8, delta=lp.delta)
    got = sc_probe_jkc.current(-0.25)
    assert abs(got-ref) < 1e-2*max(abs(ref), 1e-8)


def test_negative_voltage_didv_matches_direct_method(sc_probe_jkc):
    lp = _sc_probe_localprobe()
    ref = _direct_didv(lp, -0.25, nmax=8)
    got = sc_probe_jkc.didv(-0.25)
    assert abs(got-ref) < 0.10*max(abs(ref), 1e-8)


def test_current_and_didv_zero_at_zero_voltage(normal_jkc):
    assert normal_jkc.current(0.) == 0.
    assert normal_jkc.didv(0.) == 0.


def test_current_and_didv_consistent_with_combined_call(normal_jkc):
    val, grad = normal_jkc.current_and_didv(0.25)
    assert val == normal_jkc.current(0.25)
    assert grad == normal_jkc.didv(0.25)


def test_voltage_beyond_vmax_raises(normal_jkc):
    with pytest.raises(ValueError):
        normal_jkc.didv(normal_jkc.vmax*2)


def test_gl_order_search_is_exercised():
    """Cheap enough (small nmax, small vmax) to actually run the adaptive
    gl_order search (skipped by the fixtures above via an explicit
    gl_order, to keep this file's overall runtime reasonable) and confirm
    it terminates with a sane, finite result -- not just that a
    pre-validated fixed order works."""
    ht = _normal_normal_junction()
    jkc = JaxKeldyshCurrent(ht, nmax=2, vmax=0.15, gl_order0=50, gl_order_max=800)
    assert jkc.gl_order >= 50
    ref = _direct_didv(ht, 0.1, nmax=2)
    got = jkc.didv(0.1)
    assert np.isfinite(got)
    assert abs(got-ref) < 0.1*max(abs(ref), 1e-8)
