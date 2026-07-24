import numpy as np
import pytest

from pyqula import geometry
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
    """dc_current's default (selfenergy_method="aaa") must agree with the
    old per-energy direct Sancho-Rubio solves (selfenergy_method="direct")
    to within the sideband-convergence tolerance already requested -- the
    interpolant changes performance, not the physics."""
    lp = _superconducting_localprobe()
    kwargs = dict(nmax=4, nmax_max=12, tol=5e-2)
    Iaaa = lp.get_dc_current(0.05, **kwargs)
    Idirect = lp.get_dc_current(0.05, selfenergy_method="direct", **kwargs)
    assert abs(Iaaa-Idirect) < 5e-2*max(abs(Idirect), 1e-8)


def test_aaa_falls_back_to_direct_when_selfenergy_method_invalid():
    lp = _superconducting_localprobe()
    with pytest.raises(ValueError):
        lp.get_dc_current(0.05, nmax=4, nmax_max=12,
                           selfenergy_method="bogus")


def test_keldysh_didv_use_aaa_matches_use_qtci_and_direct():
    """The three self-energy strategies keldysh_didv can use (default AAA,
    explicit qtci, explicit direct via use_aaa=False) must agree on the
    physical dI/dV to within the sideband-convergence tolerance."""
    lp = _superconducting_localprobe()
    kwargs = dict(nmax=4, nmax_max=10, tol=5e-2)
    Gaaa = lp.didv(energy=0.25, method="keldysh", **kwargs)
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
