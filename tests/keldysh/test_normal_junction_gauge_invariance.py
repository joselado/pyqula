import numpy as np
import pytest
from scipy.integrate import quad

from pyqula import geometry
from pyqula import heterostructures
from pyqula.operators import get_electron, get_hole


def _tauz(h):
    return np.array((get_electron(h) - get_hole(h)).todense())


def _static_bias_current(h1, h2, transparency, voltage, delta):
    """Independent, non-Floquet reference: apply the bias directly as a
    static +-voltage/2 shift (via the Nambu tauz operator) on each lead's
    onsite term, and integrate the (now manifestly time-independent)
    zero-temperature Landauer current over the resulting chemical-potential
    window [-voltage/2, voltage/2]. This is the same physics the
    Floquet-Keldysh gauge transform describes, computed a completely
    different way (no Floquet sidebands at all)."""
    tauz = _tauz(h1)
    h1b = h1.copy()
    h1b.intra = h1b.intra + (voltage/2)*tauz
    h2b = h2.copy()
    h2b.intra = h2b.intra - (voltage/2)*tauz
    HTb = heterostructures.build(h1b, h2b)
    HTb.set_coupling(transparency)
    HTb.delta = delta
    f = lambda e: HTb.didv(energy=e)
    val, _ = quad(f, -abs(voltage)/2, abs(voltage)/2, limit=100, epsrel=1e-5)
    return val*np.sign(voltage)


@pytest.mark.parametrize("transparency", [0.3, 0.6, 1.0])
@pytest.mark.parametrize("voltage", [0.3, 0.6, 1.0])
def test_normal_junction_matches_static_bias_reference(transparency, voltage):
    """A two-terminal junction between two plain (non-superconducting)
    leads, promoted to trivial (zero-pairing) Nambu form so the
    Floquet-Keldysh machinery applies, must reduce to ordinary
    (non-Floquet) biased Landauer transport: the DC current computed via
    the gauge-transformed Floquet-Keldysh formalism
    (Heterostructure.get_dc_current) must match the current obtained by
    directly biasing each lead's chemical potential and integrating the
    resulting (static) transmission over the bias window. These are two
    physically equivalent but numerically unrelated ways of describing the
    same bias, related only by an exact gauge transform -- any bug in the
    Floquet-sideband bookkeeping would break this equivalence."""
    h0 = geometry.chain().get_hamiltonian()
    h1 = h0.copy()
    h1.turn_nambu()
    h2 = h1.copy()

    HT = heterostructures.build(h1.copy(), h2.copy())
    HT.set_coupling(transparency)
    HT.delta = 1e-4

    Icalc = HT.get_dc_current(voltage, nmax=8, nmax_max=30, tol=1e-4)
    Iref = _static_bias_current(h1, h2, transparency, voltage, HT.delta)

    assert abs(Icalc-Iref) < 2e-2*max(abs(Iref), 1e-8)
