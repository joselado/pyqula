import numpy as np
import pytest
from scipy.integrate import quad

from pyqula import geometry
from pyqula import heterostructures
from pyqula.operators import get_electron, get_hole
from pyqula.transporttk.fermidirac import fermidirac


def _tauz(h):
    return np.array((get_electron(h) - get_hole(h)).todense())


def _static_bias_current_finite_T(h1, h2, transparency, voltage, delta, temp):
    """Independent, non-Floquet finite-temperature reference: same static
    +-voltage/2 bias construction as test_normal_junction_gauge_invariance.
    py's zero-temperature `_static_bias_current`, but integrating the
    (already zero-temperature, `didv`-validated) transmission T(E) against
    the standard finite-temperature Landauer window `f_T(E-V/2)-f_T(E+V/2)`
    instead of a hard T=0 box -- the textbook thermal generalization of the
    same static-bias picture, computed with no Floquet-Keldysh code
    involved at all (not even at T=0). The integration window widens with
    temp (+-(|voltage|/2 + 20*temp)) to capture the Fermi tails, mirroring
    transporttk.thermaldidv.THERMAL_WINDOW's own use of a 20*temp margin."""
    tauz = _tauz(h1)
    h1b = h1.copy()
    h1b.intra = h1b.intra + (voltage/2)*tauz
    h2b = h2.copy()
    h2b.intra = h2b.intra - (voltage/2)*tauz
    HTb = heterostructures.build(h1b, h2b)
    HTb.set_coupling(transparency)
    HTb.delta = delta
    Tfun = lambda e: HTb.didv(energy=e)  # zero-temperature transmission
    integrand = lambda e: Tfun(e)*(fermidirac(e-voltage/2, temp=temp)
                                    - fermidirac(e+voltage/2, temp=temp))
    win = abs(voltage)/2 + 20*max(temp, 1e-8)
    val, _ = quad(integrand, -win, win, limit=200, epsrel=1e-6)
    return val


@pytest.mark.parametrize("transparency", [0.3, 0.6, 1.0])
@pytest.mark.parametrize("voltage", [0.3, -0.3, -0.6])
@pytest.mark.parametrize("temp", [0.02, 0.1])
def test_dc_current_temperature_matches_static_bias_finite_T_reference(
        transparency, voltage, temp):
    """Blocking gate for using keldyshtk.current.dc_current's `temperature`
    parameter anywhere else (see documentation/
    keldysh_sideband_decimation_plan.md's "direct finite-T Keldysh
    evaluation" thread): this parameter had ZERO test coverage before this
    file. For a normal-normal junction (turn_nambu, zero pairing -- the
    same reduction test_normal_junction_gauge_invariance.py's zero-
    temperature test uses), `dc_current(voltage, temperature=temp)` must
    match the independent finite-temperature Landauer reference above to
    good accuracy: this is the one regime where the Floquet-Keldysh
    formalism's finite-T occupation factors reduce to an exactly
    checkable, non-Floquet closed form. (Measured directly while building
    this test: relative error <=8e-4 across this whole parametrize grid,
    comfortably inside the tolerance below, which matches the margin the
    zero-temperature sibling test already uses.)"""
    h0 = geometry.chain().get_hamiltonian()
    h1 = h0.copy()
    h1.turn_nambu()
    h2 = h1.copy()

    HT = heterostructures.build(h1.copy(), h2.copy())
    HT.set_coupling(transparency)
    HT.delta = 1e-4

    Icalc = HT.get_dc_current(voltage, nmax=8, nmax_max=30, tol=1e-4,
                               temperature=temp)
    Iref = _static_bias_current_finite_T(h1, h2, transparency, voltage,
                                          HT.delta, temp)

    assert abs(Icalc-Iref) < 2e-2*max(abs(Iref), 1e-8)


def test_dc_current_temperature_reduces_to_zero_temperature_as_temp_to_0():
    """Sanity check on the `temperature` parameter itself (independent of
    the reference above): a tiny nonzero temperature must reproduce the
    temperature=0. result closely, not jump discontinuously -- guards
    against e.g. a sign error in `_fermi_scalar`'s finite-T branch that
    the coarse temp grid above (0.02, 0.1) might not catch."""
    h0 = geometry.chain().get_hamiltonian()
    h1 = h0.copy()
    h1.turn_nambu()
    h2 = h1.copy()

    HT = heterostructures.build(h1.copy(), h2.copy())
    HT.set_coupling(0.6)
    HT.delta = 1e-4

    I0 = HT.get_dc_current(0.6, nmax=8, nmax_max=30, tol=1e-4, temperature=0.)
    Ismall = HT.get_dc_current(0.6, nmax=8, nmax_max=30, tol=1e-4,
                                temperature=1e-4)
    assert abs(I0-Ismall) < 1e-3*max(abs(I0), 1e-8)
