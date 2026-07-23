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
@pytest.mark.parametrize("voltage", [0.3, 0.6, 1.0, -0.3, -0.6])
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
    Floquet-sideband bookkeeping would break this equivalence. Negative
    voltages are included so that a current-reversal (I(-V) = -I(V)) sign
    bug cannot slip through undetected."""
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


def _static_bias_current_with_central(h1, hc, h2, transparency, voltage, delta):
    """Same independent reference as _static_bias_current, but for a
    heterostructure with one explicit (dense, block_diagonal=False) central
    site -- pyqula's default representation whenever a single central
    Hamiltonian is passed to heterostructures.build. The central site sits
    in the unbiased gauge together with the left lead (the weak link is the
    bond adjacent to the right lead), so it is shifted the same way as h1."""
    tauz = _tauz(h1)
    h1b = h1.copy()
    h1b.intra = h1b.intra + (voltage/2)*tauz
    hcb = hc.copy()
    hcb.intra = hcb.intra + (voltage/2)*tauz
    h2b = h2.copy()
    h2b.intra = h2b.intra - (voltage/2)*tauz
    HTb = heterostructures.build(h1b, h2b, central=[hcb])
    HTb.set_coupling(transparency)
    HTb.delta = delta
    f = lambda e: HTb.didv(energy=e)
    val, _ = quad(f, -abs(voltage)/2, abs(voltage)/2, limit=100, epsrel=1e-5)
    return val*np.sign(voltage)


@pytest.mark.parametrize("transparency", [0.3, 0.6, 1.0])
@pytest.mark.parametrize("voltage", [0.3, 0.6, -0.3])
def test_single_central_site_matches_static_bias_reference(transparency, voltage):
    """heterostructures.build(h1, h2, central=[hc]) with a single explicit
    central Hamiltonian is pyqula's default representation of a junction
    with one normal weak-link site, and is always block_diagonal=False
    (a dense central_intra, not a block list) -- a case get_dc_current used
    to reject outright. It must satisfy the same gauge-invariance check as
    the no-explicit-central-site case above."""
    h0 = geometry.chain().get_hamiltonian()
    h1 = h0.copy()
    h1.turn_nambu()
    hc = h1.copy()
    h2 = h1.copy()

    HT = heterostructures.build(h1.copy(), h2.copy(), central=[hc.copy()])
    assert not HT.block_diagonal
    HT.set_coupling(transparency)
    HT.delta = 1e-4

    Icalc = HT.get_dc_current(voltage, nmax=8, nmax_max=30, tol=1e-4)
    Iref = _static_bias_current_with_central(h1, hc, h2, transparency,
                                              voltage, HT.delta)

    assert abs(Icalc-Iref) < 2e-2*max(abs(Iref), 1e-8)
