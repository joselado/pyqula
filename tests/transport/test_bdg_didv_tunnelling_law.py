import numpy as np
import pytest

from pyqula import geometry, heterostructures


def _junction(has_spin, coupling):
    g = geometry.chain()
    lead = g.get_hamiltonian(has_spin=has_spin)
    lead.add_swave(0.0)  # Nambu basis, no pairing
    ht = heterostructures.build(lead, lead)
    ht.set_coupling(coupling)
    return ht


def test_spinful_nambu_junction_follows_the_tunnelling_law():
    """A weakly coupled junction is in the tunnelling limit, where the
    conductance goes as the square of the hopping across the barrier, so
    halving the coupling quarters dI/dV."""
    cs = np.array([0.005, 0.010, 0.020])
    gs = np.array([_junction(True, c).didv(energy=0.2) for c in cs])
    assert np.all(gs > 0.)
    ratios = gs[1:]/gs[:-1]
    assert np.allclose(ratios, 4.0, rtol=0.05)  # G ~ t^2


def test_a_junction_built_from_spinless_chains_is_spinful():
    """A Nambu Hamiltonian is always spinful, so leads built from spinless
    chains give the spinful junction above. It used to build spinless Nambu
    leads, which the electron-hole reordering (4 components per site)
    turned into zero reflection blocks, so dI/dV came out as the channel
    count, 1.0 at every coupling and energy."""
    for c in [0.005, 0.010, 0.020]:
        g0 = _junction(False, c).didv(energy=0.2)
        g1 = _junction(True, c).didv(energy=0.2)
        assert abs(g0 - g1) < 1e-10*max(abs(g1), 1e-12), (c, g0, g1)
