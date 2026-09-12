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
    """Control for the guard below. A weakly coupled junction is in the
    tunnelling limit, where the conductance goes as the square of the
    hopping across the barrier, so halving the coupling quarters dI/dV.

    This is the behaviour the spinless branch could not produce, and it is
    asserted here so that the guard cannot be "satisfied" by breaking the
    working case too."""
    cs = np.array([0.005, 0.010, 0.020])
    gs = np.array([_junction(True, c).didv(energy=0.2) for c in cs])
    assert np.all(gs > 0.)
    ratios = gs[1:]/gs[:-1]
    assert np.allclose(ratios, 4.0, rtol=0.05)  # G ~ t^2


def test_spinless_nambu_junction_is_refused_rather_than_answered_wrongly():
    """A spinless Nambu lead does not match the 4-degrees-of-freedom-per-site
    layout that the electron-hole reordering assumes (sctk.reorder's
    block2nambu_matrix builds nr = n//4, which is 0 here), so both
    reflection blocks used to come back as the zero matrix and didv_BdG
    returned exactly the channel count -- 1.0 at every coupling and every
    energy, indistinguishable from a perfectly transparent contact.

    transporttk.kappa_jax.applicable already refuses this layout for the
    same reason. The requirement is now named instead of being answered
    with a plausible-looking number."""
    for c in [0.009, 0.010, 0.011]:
        with pytest.raises(NotImplementedError, match="spinful Nambu"):
            _junction(False, c).didv(energy=0.2)
