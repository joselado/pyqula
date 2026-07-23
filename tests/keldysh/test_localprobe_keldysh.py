import numpy as np
import pytest
from scipy.integrate import quad

from pyqula import geometry
from pyqula.transporttk.localprobe import LocalProbe
from pyqula.operators import get_electron, get_hole


def _tauz(h):
    return np.array((get_electron(h)-get_hole(h)).todense())


def test_auto_method_keeps_smatrix_when_probe_is_normal():
    """A normal-metal probe on a superconducting sample (the setup used by
    examples/transport/decay_constant) must keep using the existing
    scattering-matrix/BdG formula -- Floquet-Keldysh only applies once the
    probe lead itself carries a nonzero pairing amplitude."""
    g = geometry.chain()
    h = g.get_hamiltonian(); h.shift_fermi(1.); h.add_swave(0.1)
    lp = LocalProbe(h, delta=1e-8)
    lp.T = 2e-2
    Gauto = lp.didv(energy=0.05)
    Gsmatrix = lp.didv(energy=0.05, method="smatrix")
    assert Gauto == Gsmatrix


def test_auto_method_routes_to_keldysh_when_probe_and_sample_are_superconducting():
    """With both the probe lead and the sample superconducting, the default
    ("auto") method must route through the Floquet-Keldysh dI/dV, matching
    an explicit method="keldysh" call exactly (same code path)."""
    g = geometry.chain()
    h = g.get_hamiltonian(); h.shift_fermi(1.); h.add_swave(0.1)
    lead = geometry.chain().get_hamiltonian(); lead.shift_fermi(1.); lead.add_swave(0.1)
    lp = LocalProbe(h, lead=lead, delta=1e-3)
    lp.T = 0.3
    kwargs = dict(nmax=4, nmax_max=10, tol=1e-2)
    Gauto = lp.didv(energy=0.25, **kwargs)
    Gkeldysh = lp.didv(energy=0.25, method="keldysh", **kwargs)
    assert Gauto == Gkeldysh


def test_keldysh_linear_response_matches_equilibrium_andreev_for_localprobe():
    """For a LocalProbe with a normal (zero-pairing but Nambu) probe and a
    superconducting sample, the zero-bias slope of the Floquet-Keldysh
    current (LocalProbe.get_dc_current) must match the existing
    equilibrium Andreev conductance (LocalProbe.didv, via
    transporttk.didv.didv_BdG) -- the two are the same physical quantity
    computed by independent code paths, exactly as already checked for a
    two-lead Heterostructure in test_andreev_linear_response.py."""
    g = geometry.chain()
    h = g.get_hamiltonian(); h.shift_fermi(1.); h.turn_nambu()
    lead = geometry.chain().get_hamiltonian(has_spin=False); lead.turn_nambu()
    delta = 0.3
    h.add_swave(delta)
    lp = LocalProbe(h, lead=lead, delta=1e-4)
    lp.T = 0.5

    Gref = lp.didv(energy=1e-4, method="smatrix")
    dv = 0.02*delta
    Gkeldysh = lp.didv(energy=0.0, method="keldysh", dv=dv,
                        nmax=4, nmax_max=10, tol=1e-2)
    assert abs(Gkeldysh-Gref) < 0.05*max(Gref, 1e-8)


@pytest.mark.parametrize("voltage", [0.3, -0.3])
def test_localprobe_normal_junction_matches_static_bias_reference(voltage):
    """A LocalProbe between two plain (non-superconducting) leads, promoted
    to trivial (zero-pairing) Nambu form so the Floquet-Keldysh machinery
    applies, must reduce to ordinary (non-Floquet) biased Landauer
    transport, exactly as already checked for a two-lead Heterostructure in
    test_normal_junction_gauge_invariance.py: the DC current from
    get_dc_current must match the current obtained by directly biasing the
    probe and the sample and integrating the resulting static transmission
    over the bias window."""
    g = geometry.chain()
    h = g.get_hamiltonian(); h.shift_fermi(1.); h.turn_nambu()
    lead = geometry.chain().get_hamiltonian(has_spin=False); lead.turn_nambu()
    transparency = 0.4
    lp = LocalProbe(h, lead=lead, delta=1e-4)
    lp.T = transparency

    Icalc = lp.get_dc_current(voltage, nmax=6, nmax_max=16, tol=1e-3)

    tz_h = _tauz(h)
    hb = h.copy(); hb.intra = hb.intra - (voltage/2)*tz_h
    tz_l = _tauz(lead)
    leadb = lead.copy(); leadb.intra = leadb.intra + (voltage/2)*tz_l
    lpb = LocalProbe(hb, lead=leadb, delta=1e-4)
    lpb.T = transparency
    f = lambda e: lpb.didv(energy=e, method="smatrix")
    Iref, _ = quad(f, -abs(voltage)/2, abs(voltage)/2, limit=100, epsrel=1e-5)
    Iref *= np.sign(voltage)

    assert abs(Icalc-Iref) < 2e-2*max(abs(Iref), 1e-8)


def test_explicit_central_region_style_checks_still_apply():
    """get_dc_current on a LocalProbe still requires a Nambu (BdG) sample,
    just like the two-lead heterostructure case."""
    g = geometry.chain()
    h = g.get_hamiltonian()  # no turn_nambu(): has_eh is False
    lp = LocalProbe(h, delta=1e-4)
    with pytest.raises(NotImplementedError):
        lp.get_dc_current(0.1)
