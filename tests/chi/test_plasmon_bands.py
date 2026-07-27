import numpy as np

from pyqula import geometry
from pyqula.chitk.densitychi import _density_v
from pyqula.chitk.rpa import rpa_kernel_poles, interaction_at_q
from pyqula.chi import chiAB


def test_plasmon_bands_matches_direct_kernel_poles():
    """get_plasmon_bands must agree exactly with calling the lower-level
    rpa_kernel_poles directly, using the same V1/V2/V3/U/Vr-built
    interaction dict -- mirrors tests/chi/test_magnon_bands.py's
    consistency check for the spin channel, here for the charge channel."""
    g = geometry.chain()
    h = g.get_hamiltonian()
    energies = np.linspace(0.0002, 0.3, 60)
    q0 = [0.5, 0., 0.]

    qs, ws, gammas = h.get_plasmon_bands(V1=-0.5, qpath=[q0], nq=1,
                                          energies=energies, delta=1e-2, nk=2000)

    v = _density_v(h, V1=-0.5)
    direct_poles = rpa_kernel_poles(h, V=v, q=q0, energies=energies,
                                     delta=1e-2, nk=2000)

    assert np.all(qs == 0)
    assert len(ws) == len(direct_poles)
    if len(ws) > 0:
        order = np.argsort(ws)
        direct_order = np.argsort(direct_poles[:, 0])
        assert np.allclose(ws[order], direct_poles[direct_order, 0])
        assert np.allclose(gammas[order], direct_poles[direct_order, 1])


def test_plasmon_bands_shape_and_no_interaction_gives_no_poles():
    g = geometry.chain()
    h = g.get_hamiltonian()
    energies = np.linspace(0.01, 1.0, 30)
    qs, ws, gammas = h.get_plasmon_bands(qpath=[[0.1, 0., 0.], [0.3, 0., 0.]],
                                          nq=2, energies=energies, delta=1e-2, nk=200)
    assert qs.shape == ws.shape == gammas.shape
    assert len(ws) == 0  # V1=V2=V3=U=0 by default: no interaction, no poles


def test_nesting_enhances_charge_instability_at_half_filling():
    """Physical motivation for a non-onsite (V1) charge-channel RPA: a 1D
    chain has PERFECT Fermi-surface nesting at q=pi exactly at half
    filling (every occupied k pairs with an empty k+pi at the same
    energy), giving a strongly enhanced static charge susceptibility
    there -- the charge-channel analog of the low-filling-enhanced
    ferromagnetic instability in test_rpa_nononsite_interaction.py. For a
    FIXED nearest-neighbor V1, the static (omega->0) kernel eigenvalue at
    q=pi must be much closer to (or past) the CDW instability threshold
    exactly at half filling than away from it."""
    g = geometry.chain()
    q = [0.5, 0., 0.]  # q = pi
    # V1(q=pi) = -V1 (the +1/-1 neighbor bonds pick up a pi Bloch phase at
    # the zone boundary), so a REPULSIVE V1 is the one that drives a CDW
    # instability there -- the standard "electrons avoid each other on
    # neighboring sites -> checkerboard charge order" mechanism, enhanced
    # here by perfect nesting at half filling.
    V1 = 0.6
    energies = np.array([0.0])

    def static_kernel_eig(filling):
        h = g.get_hamiltonian()
        mu = h.get_fermi4filling(filling, nk=2000)
        h.shift_fermi(-mu)
        v = _density_v(h, V1=V1)
        es, chis = chiAB(h, mode="matrix", q=q, energies=energies, delta=1e-2, nk=2000)
        vq = interaction_at_q(v, h, q)
        return float((1 - vq @ chis[0])[0, 0].real)

    eig_half = static_kernel_eig(0.5)
    eig_away = static_kernel_eig(0.3)
    assert eig_half < eig_away, \
        f"expected nesting at half filling to push the kernel closer to " \
        f"instability: half-filling={eig_half}, away={eig_away}"
    assert eig_half < 0, "expected this V1 to be past threshold exactly at half filling"
    assert eig_away > 0, "expected this V1 to stay stable away from half filling"
