import numpy as np
import pytest

from pyqula import geometry
from pyqula import heterostructures
from pyqula.keldyshtk import current as keldysh_current
from pyqula.transporttk import kappa as kappa_mod
from pyqula.transporttk import thermaldidv as thermaldidv_mod
from pyqula.transporttk import didv as didv_mod

# Deliberately cheap settings (as in tests/keldysh/test_selfenergy_aaa.py and
# the decay_constant_keldysh example): these tests are about the SHARING
# plumbing (how many times build_selfenergy_aaa is called, and whether every
# sub-call in a sweep sees the same interpolant), not about physical
# convergence, so a small nmax_max keeps them fast. selfenergy_method="aaa"
# is required explicitly since dc_current's default is now "direct" (see
# its own docstring for the accuracy gap that made AAA opt-in only) --
# these tests specifically exercise the AAA-sharing path.
_CHEAP = dict(nmax=2, nmax_max=6, tol=1e-2, selfenergy_method="aaa")


def _sc_junction(delta=0.3, transparency=0.3):
    """Two-lead heterostructure with both leads genuinely superconducting
    -- the case that routes through the Floquet-Keldysh path these sharing
    fixes target (see transporttk.didv._both_leads_superconducting)."""
    g = geometry.chain()
    h1 = g.get_hamiltonian(); h1.add_swave(delta)
    h2 = g.get_hamiltonian(); h2.add_swave(delta)
    HT = heterostructures.build(h1, h2)
    HT.set_coupling(transparency)
    HT.delta = 1e-4
    return HT


def _normal_junction(transparency=0.3):
    """Plain normal-normal junction: didv's "auto" method picks "smatrix",
    never Keldysh, so no self-energy interpolant should ever be built for
    it -- the common case these sharing fixes must leave untouched."""
    g = geometry.chain()
    h1 = g.get_hamiltonian()
    h2 = g.get_hamiltonian()
    HT = heterostructures.build(h1, h2)
    HT.set_coupling(transparency)
    HT.delta = 1e-4
    return HT


def _counting_build(monkeypatch):
    """Count real calls to keldyshtk.current.build_selfenergy_aaa -- every
    sharing fix in this module funnels through build_shared_selfenergy,
    which funnels through this one function, so counting it directly
    measures how many independent AAA fits a sweep actually paid for."""
    calls = []
    orig = keldysh_current.build_selfenergy_aaa
    def counting(*a, **kw):
        calls.append(1)
        return orig(*a, **kw)
    monkeypatch.setattr(keldysh_current, "build_selfenergy_aaa", counting)
    return calls


def test_iv_curve_shares_one_interpolant_across_the_voltage_sweep(monkeypatch):
    """iv_curve used to let every voltage in the sweep independently build
    (and discard) its own default AAA fit via dc_current's own
    selfenergy_method="aaa" default -- N voltages, N builds. It should now
    build exactly one, sized to cover the whole array, and every voltage's
    dc_current call should still agree with what an independent per-voltage
    solve gives (small differences are expected: the shared fit is built
    once over a wider window, an independent one is refit per voltage over
    its own narrower window -- both within AAA's own tolerance)."""
    HT = _sc_junction()
    calls = _counting_build(monkeypatch)
    voltages = [0.05, 0.1, 0.15, 0.2]

    Is_shared = HT.get_iv_curve(voltages, **_CHEAP)

    assert len(calls) == 1

    Is_percall = [keldysh_current.dc_current(HT, v, **_CHEAP) for v in voltages]
    assert np.max(np.abs(np.array(Is_shared)-np.array(Is_percall))) < 2e-2


def test_iv_curve_respects_a_caller_supplied_selfenergy_qtci(monkeypatch):
    """If the caller already threaded their own selfenergy_qtci through
    (e.g. one built to cover a wider window than this particular sweep,
    shared across several separate calls of their own), iv_curve must not
    silently override it with a freshly auto-built one -- the same
    explicit-override escape hatch keldysh_didv/dc_current already
    document and honor elsewhere (see dc_current's own docstring)."""
    HT = _sc_junction()
    calls = _counting_build(monkeypatch)
    seen = []
    def spy_dc_current(ht, voltage, **kwargs):
        seen.append(kwargs.get("selfenergy_qtci"))
        return 0.0  # stand-in, cost-free
    monkeypatch.setattr(keldysh_current, "dc_current", spy_dc_current)

    sentinel = object()
    keldysh_current.iv_curve(HT, [0.05, 0.1], selfenergy_qtci=sentinel, **_CHEAP)

    assert len(calls) == 0  # no auto-build attempted at all
    assert seen == [sentinel, sentinel]


def test_finite_T_didv_shares_one_interpolant_across_its_thermal_quadrature(monkeypatch):
    """A single finite_T_didv call runs its own internal thermal
    quadrature (order-100 zero_T_didv evaluations spanning
    +-thermaldidv.THERMAL_WINDOW*temp around `energy`) -- previously each
    one independently built and discarded its own default AAA fit, almost
    entirely redundant since all of them share the same two leads and only
    need self-energies over one common, boundable window. Confirm exactly
    one build now happens, and that every single quadrature node receives
    that SAME interpolant object rather than each getting its own."""
    HT = _sc_junction()
    calls = _counting_build(monkeypatch)
    node_selfenergies = []
    def fake_zero_T_didv(self, energy=0.0, **kwargs):
        node_selfenergies.append(kwargs.get("selfenergy_qtci"))
        return 1.0  # stand-in, cost-free: this test is about the wiring
    monkeypatch.setattr(didv_mod, "zero_T_didv", fake_zero_T_didv)

    thermaldidv_mod.finite_T_didv(HT, temp=0.02, energy=0.05, nmax_max=6,
                                   selfenergy_method="aaa",
                                   keldysh_thermal_mode="convolution")

    assert len(calls) == 1
    assert len(node_selfenergies) > 1  # the thermal quadrature really did visit many nodes
    assert all(s is not None for s in node_selfenergies)
    assert len({id(s) for s in node_selfenergies}) == 1  # all the SAME object


def test_finite_T_didv_skips_sharing_for_a_non_keldysh_junction(monkeypatch):
    """The ordinary (non-superconducting, or single-lead-superconducting)
    case must be completely unaffected: no self-energy interpolant is
    applicable there (didv's "auto" method picks the already-cheap
    "smatrix" formula), so finite_T_didv must not attempt to build one."""
    HT = _normal_junction()
    calls = _counting_build(monkeypatch)

    val = thermaldidv_mod.finite_T_didv(HT, temp=0.05, energy=0.1)

    assert len(calls) == 0
    assert np.isfinite(val)


def test_get_kappa_ratio_only_shares_the_sc_branchs_interpolant(monkeypatch):
    """get_kappa_ratio evaluates two branches (generate_HT SC=True/False)
    through the same get_kappa/get_conductances/get_single chain; only the
    SC branch can route through Keldysh, so only its call may carry a
    shared selfenergy_qtci -- handing it to the normal branch too would be
    the wrong leads' self-energy entirely (see _with_shared_selfenergy's
    docstring)."""
    HT = _sc_junction()
    calls = _counting_build(monkeypatch)
    seen = []
    orig_get_kappa = kappa_mod.get_kappa
    def spy_get_kappa(**kwargs):
        seen.append(kwargs.get("selfenergy_qtci"))
        return orig_get_kappa(**kwargs)
    monkeypatch.setattr(kappa_mod, "get_kappa", spy_get_kappa)

    kappa_mod.get_kappa_ratio(HT, energy=0.05, **_CHEAP)

    assert len(calls) == 1
    assert len(seen) == 2
    assert seen[0] is not None   # SC branch: got the shared interpolant
    assert seen[1] is None       # normal branch: kwargs untouched


_CHEAP_DIDV = dict(nmax_max=6, fixed_nmax=6)


def test_didv_curve_shares_one_interpolant_across_the_energy_sweep(monkeypatch):
    """A raw loop of `didv(energy=e, use_aaa=True)` calls independently
    builds (and discards) its own AAA fit at every energy, since
    keldysh_didv's own sharing is scoped to just its Ip/Im pair within one
    call -- didv_curve exists to share one build across the whole energy
    sweep instead, the same fix iv_curve already got for dc_current
    sweeps."""
    HT = _sc_junction()
    calls = _counting_build(monkeypatch)
    # Deliberately away from voltage~0 -- dc_current's own adaptive sweep
    # already warns that this cheap nmax_max=6 doesn't converge tightly
    # near there (see test_iv_curve_shares_one_interpolant_across_the_
    # voltage_sweep's identical warning), and keldysh_didv's finite
    # difference amplifies that residual noise the same way the
    # catastrophic-cancellation case in
    # documentation/keldysh_aaa_selfenergy_accuracy_plan.md does -- this
    # test is about the sharing plumbing, not about nailing convergence at
    # a setting this cheap.
    energies = [0.1, 0.15, 0.2, 0.25]

    Gs_shared = didv_mod.didv_curve(HT, energies, use_aaa=True, **_CHEAP_DIDV)

    assert len(calls) == 1

    Gs_percall = [didv_mod.keldysh_didv(HT, voltage=e, **_CHEAP_DIDV)
                  for e in energies]
    assert np.max(np.abs(np.array(Gs_shared)-np.array(Gs_percall))) < 2e-2


def test_didv_curve_skips_sharing_for_a_normal_junction(monkeypatch):
    """The ordinary (non-superconducting) case must be completely
    unaffected: didv's "auto" method picks the already-cheap "smatrix"
    formula there, so didv_curve must not attempt to build any self-energy
    interpolant, and must still match the equivalent per-energy loop
    (this is also the array-input case that used to hard-fail with a numpy
    broadcasting error before didv_curve existed)."""
    HT = _normal_junction()
    calls = _counting_build(monkeypatch)
    energies = [0.05, 0.1, 0.15]

    Gs_curve = didv_mod.didv_curve(HT, energies, use_aaa=True)

    assert len(calls) == 0
    Gs_loop = [HT.didv(energy=e) for e in energies]
    assert np.allclose(Gs_curve, Gs_loop)


def test_didv_curve_respects_a_caller_supplied_selfenergy_qtci(monkeypatch):
    """Same explicit-override escape hatch as iv_curve/dc_current: if the
    caller already threaded their own selfenergy_qtci through, didv_curve
    must not silently replace it with a freshly auto-built one."""
    HT = _sc_junction()
    calls = _counting_build(monkeypatch)
    seen = []
    orig = didv_mod.keldysh_didv
    def spy_keldysh_didv(ht, voltage=0.0, **kwargs):
        seen.append(kwargs.get("selfenergy_qtci"))
        return 0.0  # stand-in, cost-free
    monkeypatch.setattr(didv_mod, "keldysh_didv", spy_keldysh_didv)

    sentinel = object()
    didv_mod.didv_curve(HT, [0.05, 0.1], use_aaa=True,
                         selfenergy_qtci=sentinel, **_CHEAP_DIDV)

    assert len(calls) == 0  # no auto-build attempted at all
    assert seen == [sentinel, sentinel]


def test_didv_energies_keyword_dispatches_to_didv_curve():
    """`didv(energies=array)` (note the plural, distinct from and mutually
    exclusive with the scalar `energy=`) must transparently route to
    didv_curve -- checked on both the smatrix (normal junction) and
    keldysh (SC-SC, use_aaa=True) paths, matching didv_curve's own output
    exactly (not just approximately: it's the same code path, not an
    independent re-implementation). Passing an array as the scalar
    `energy=` itself is not supported (see `didv`'s own docstring)."""
    h1 = geometry.chain().get_hamiltonian()
    h2 = geometry.chain().get_hamiltonian()
    normal_HT = heterostructures.build(h1, h2)
    normal_HT.set_coupling(0.5)
    es = np.array([-0.1, 0.0, 0.1])
    assert np.array_equal(normal_HT.didv(energies=es),
                           didv_mod.didv_curve(normal_HT, es))

    HT = _sc_junction()
    energies = [0.1, 0.15, 0.2]
    assert np.array_equal(
        HT.didv(energies=energies, use_aaa=True, **_CHEAP_DIDV),
        didv_mod.didv_curve(HT, energies, use_aaa=True, **_CHEAP_DIDV))
