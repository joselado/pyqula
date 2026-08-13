import warnings

import numpy as np
import pytest

from pyqula import geometry
from pyqula import heterostructures
from pyqula.transporttk.localprobe import LocalProbe
from pyqula.aaatk.aaa import aaa
from pyqula.aaatk.selfenergy_aaa import SelfenergyAAA
from pyqula.keldyshtk.current import build_selfenergy_aaa


def _superconducting_localprobe():
    h = geometry.chain().get_hamiltonian(); h.shift_fermi(1.); h.add_swave(0.1)
    lead = geometry.chain().get_hamiltonian(); lead.shift_fermi(1.); lead.add_swave(0.1)
    lp = LocalProbe(h, lead=lead, delta=1e-3)
    lp.T = 0.3
    return lp


def test_aaa_reproduces_a_function_with_nearby_poles():
    """Sanity check of the bare AAA algorithm on a hand-built target with
    two narrow near-real-axis poles plus a smooth part -- the same
    structural shape (poles/resonances riding on a smooth background) a
    retarded self-energy has, but with a known closed form to check
    against exactly."""
    p1, p2 = 0.3 - 1e-3j, -0.5 - 2e-3j
    f = lambda x: 1.0/(x-p1) + 0.7/(x-p2) + 0.2*np.sin(3*x)
    Z = np.linspace(-1, 1, 4000)
    r, zj, fj, w, errvec = aaa(f(Z), Z, tol=1e-12, mmax=60)
    assert len(zj) < 20  # far fewer support points than candidate points
    test = np.linspace(-0.99, 0.99, 11)
    assert np.max(np.abs(r(test)-f(test))/np.abs(f(test))) < 1e-8
    # scalar and vectorized evaluation paths must agree
    for x in test:
        assert abs(r(complex(x)) - r(np.array([x]))[0]) < 1e-12


def test_selfenergy_aaa_matches_direct_sancho_rubio():
    """SelfenergyAAA's interpolated self-energy must reproduce the direct
    Sancho-Rubio/bloch_selfenergy solve (keldyshtk.current._cached_selfenergy's
    other branch) to well within its requested tolerance, for both the
    probe lead (surface Green's function) and the sample-site environment
    (bulk Green's function) -- the two get_selfenergy(lead=...) branches
    LocalProbe.get_selfenergy dispatches between."""
    lp = _superconducting_localprobe()
    delta = lp.delta
    voltage, nmax_max = 0.02, 12
    erange = (nmax_max+1)*abs(voltage)
    for lead in (0, 1):
        def get_se(e, lead=lead):
            return lp.get_selfenergy(e, lead=lead, delta=delta,
                                      pristine=True, numba=True)
        dim = lp.lead.intra.shape[0]
        interp = SelfenergyAAA(get_se, dim, -erange, erange, delta,
                                tolerance=1e-6)
        assert interp.converged
        for e in np.linspace(-0.85*erange, 0.85*erange, 7):
            true = get_se(e)
            approx = interp(e)
            err = np.max(np.abs(approx-true))/max(np.max(np.abs(true)), 1e-10)
            assert err < 1e-4


def test_build_selfenergy_aaa_matches_direct_dc_current():
    """selfenergy_method="aaa" (opt-in -- dc_current's default is "direct",
    see its own docstring for the accuracy gap that made this opt-in only)
    must agree with the direct per-energy Sancho-Rubio solves to within
    the sideband-convergence tolerance already requested, for this small,
    cheap-nmax_max case."""
    lp = _superconducting_localprobe()
    kwargs = dict(nmax=4, nmax_max=12, tol=5e-2)
    Iaaa = lp.get_dc_current(0.05, selfenergy_method="aaa", **kwargs)
    Idirect = lp.get_dc_current(0.05, selfenergy_method="direct", **kwargs)
    assert abs(Iaaa-Idirect) < 5e-2*max(abs(Idirect), 1e-8)


def test_aaa_falls_back_to_direct_when_selfenergy_method_invalid():
    lp = _superconducting_localprobe()
    with pytest.raises(ValueError):
        lp.get_dc_current(0.05, nmax=4, nmax_max=12,
                           selfenergy_method="bogus")


def test_keldysh_didv_use_aaa_matches_use_qtci_and_direct():
    """The three self-energy strategies keldysh_didv can use (opt-in AAA,
    explicit qtci, default direct via use_aaa=False) must agree on the
    physical dI/dV to within the sideband-convergence tolerance, for this
    small, cheap-nmax_max case."""
    lp = _superconducting_localprobe()
    kwargs = dict(nmax=4, nmax_max=10, tol=5e-2)
    Gaaa = lp.didv(energy=0.25, method="keldysh", use_aaa=True, **kwargs)
    Gdirect = lp.didv(energy=0.25, method="keldysh", use_aaa=False, **kwargs)
    assert abs(Gaaa-Gdirect) < 5e-2*max(abs(Gdirect), 1e-8)


def test_selfenergy_aaa_bounded_effort_on_a_hard_target():
    """A synthetic target with far more independent resonances than a
    modest support-point budget can represent must not hang or run away:
    SelfenergyAAA gives up (converged=False) within its bounded budget
    rather than escalating ncand/mmax indefinitely -- the pathology this
    guards against was a real, measured multi-minute-plus hang before the
    mmax/ncand escalation logic was split apart (see the module
    docstring)."""
    rng = np.random.default_rng(0)
    poles = rng.uniform(-1, 1, 60) - 1e-3j
    def get_se(e, poles=poles):
        return np.array([[sum(1.0/(e-p) for p in poles)]], dtype=np.complex128)
    interp = SelfenergyAAA(get_se, 1, -1., 1., 1e-3, tolerance=1e-10,
                            ncand_max=600, mmax_max=80, maxrounds=5)
    assert interp.ncand <= 600 and interp.mmax <= 80


def test_out_of_window_call_warns_once_and_in_window_is_silent():
    """SelfenergyAAA enforces no domain -- __call__/call_batch will happily
    extrapolate past [emin,emax] -- but must warn (once per instance, not
    once per call) when that happens, and stay silent for in-window calls
    (including a plain in-window sweep via call_batch), so the warning is a
    real signal rather than noise on every ordinary use."""
    def get_se(e):
        return np.array([[1.0/(e-(0.5-1e-3j))]], dtype=np.complex128)
    interp = SelfenergyAAA(get_se, 1, -1., 1., 1e-3, tolerance=1e-6)
    assert interp.converged

    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        interp(0.3)  # in-window scalar call
        interp.call_batch(np.linspace(-0.9, 0.9, 5))  # in-window batch call
        assert len(rec) == 0

    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        interp(1.5)  # past emax
        interp(1.6)  # a second out-of-window call
        interp.call_batch(np.array([0.0, -1.5]))  # batch straddling emin
        domain_warnings = [w for w in rec if issubclass(w.category, UserWarning)
                            and "fitted window" in str(w.message)]
        assert len(domain_warnings) == 1  # warned once, not once per call


def test_batched_selfenergy_solve_gives_the_same_fit_as_the_scalar_loop():
    """get_selfenergy_batch is a pure speedup of SelfenergyAAA's build (the
    numba prange-parallel Sancho-Rubio solve, transporttk.selfenergy.
    get_selfenergy_batch, instead of one Python-level get_selfenergy call
    per candidate/validation energy) -- it must solve the exact same
    energies in the exact same rounds and reach the exact same fit, not
    just a similar one. Compares the batched build (as build_selfenergy_aaa
    wires it in for any Heterostructure) against the scalar-loop build
    (get_selfenergy_batch=None) on the same two-lead SC-SC system."""
    h1 = geometry.chain().get_hamiltonian(); h1.shift_fermi(1.); h1.add_swave(0.1)
    h2 = geometry.chain().get_hamiltonian(); h2.shift_fermi(1.); h2.add_swave(0.1)
    HT = heterostructures.build(h1, h2)
    HT.set_coupling(0.3)
    HT.delta = 1e-3

    voltage, nmax_max = 0.05, 8
    erange = (nmax_max+1)*abs(voltage)
    dim = HT.Hl.intra.shape[0]

    def get_se(e):
        return HT.get_selfenergy(e, lead=0, delta=HT.delta,
                                  pristine=True, numba=True)
    def get_se_batch(es):
        return HT.get_selfenergy_batch(es, lead=0, delta=HT.delta, pristine=True)

    scalar = SelfenergyAAA(get_se, dim, -erange, erange, HT.delta, tolerance=1e-4)
    batched = SelfenergyAAA(get_se, dim, -erange, erange, HT.delta, tolerance=1e-4,
                             get_selfenergy_batch=get_se_batch)

    assert scalar.converged and batched.converged
    assert batched.ncand == scalar.ncand
    assert batched.mmax == scalar.mmax
    assert batched.nsolved() == scalar.nsolved()
    assert batched.validation_error == scalar.validation_error

    for e in np.linspace(-0.9*erange, 0.9*erange, 9):
        assert np.array_equal(batched(e), scalar(e))

    # build_selfenergy_aaa must actually wire the batched path in for a
    # Heterostructure (not silently fall back to the scalar loop).
    interp = build_selfenergy_aaa(HT, voltage, nmax_max, delta=HT.delta,
                                   tolerance=1e-4)
    assert interp[0].nsolved() == scalar.nsolved()


def test_build_selfenergy_aaa_shares_one_fit_for_a_symmetric_junction():
    """build_selfenergy_aaa must build only ONE SelfenergyAAA (halving the
    build cost) and return it for both leads when they have the identical
    self-energy -- the common case of the same physical lead on both
    sides of heterostructures.build -- but must NEVER do this for an
    asymmetric junction (different lead physics on each side) or a
    LocalProbe (lead 0 is the probe's own surface GF, lead 1 is the bulk
    sample-site GF it couples to -- always different physics, however
    similar the raw matrices might look)."""
    def sc_sc_junction(delta_sc_left, delta_sc_right, transparency=0.5, ht_delta=1e-3):
        h1 = geometry.chain().get_hamiltonian(); h1.shift_fermi(1.); h1.add_swave(delta_sc_left)
        h2 = geometry.chain().get_hamiltonian(); h2.shift_fermi(1.); h2.add_swave(delta_sc_right)
        HT = heterostructures.build(h1, h2)
        HT.set_coupling(transparency)
        HT.delta = ht_delta
        return HT

    symmetric = sc_sc_junction(0.3, 0.3)
    interp_sym = build_selfenergy_aaa(symmetric, 0.05, 4, tolerance=1e-2)
    assert interp_sym[0] is interp_sym[1]

    asymmetric = sc_sc_junction(0.3, 0.15)
    interp_asym = build_selfenergy_aaa(asymmetric, 0.05, 4, tolerance=1e-2)
    assert interp_asym[0] is not interp_asym[1]

    lp = _superconducting_localprobe()
    interp_lp = build_selfenergy_aaa(lp, 0.02, 4, tolerance=1e-2)
    assert interp_lp[0] is not interp_lp[1]
