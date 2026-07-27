import numpy as np

from pyqula import geometry
from pyqula.selfconsistency.spinspin import _build_v
from pyqula.chitk.spinchi import _full_spin_operators, _full_spin_U
from pyqula.chitk.rpa import rpa_kernel_poles_ops, _chi_ops_matrix_vectorized, \
        interaction_at_q


def _chain_with_nn_exchange(filling, J1, nk_fermi=4000):
    """A bare (no SCF) chain with a nearest-neighbor-only SzSz exchange
    interaction set directly on H.V -- i.e. H.V has the (0,0,0), (1,0,0)
    and (-1,0,0) keys, none of them optional/prunable, exercising the
    non-onsite RPA support directly (no mean field is needed: at U=0 the
    bare band structure already is the exact paramagnetic reference
    state; only the RPA dressing interaction is non-onsite here)."""
    g = geometry.chain()
    h = g.get_hamiltonian()
    mu = h.get_fermi4filling(filling, nk=nk_fermi)
    h.shift_fermi(-mu)
    h.V = _build_v(h, J1=J1)
    return h


def test_multishell_interaction_has_more_than_one_key():
    """Sanity check on the test fixture itself: the whole point is that
    H.V is NOT onsite-only (that used to make _full_spin_U/magnon_bands
    raise)."""
    h = _chain_with_nn_exchange(filling=0.1, J1=-1.0)
    assert len(h.V) > 1
    assert (0, 0, 0) in h.V


def test_magnon_bands_matches_direct_kernel_poles_for_nononsite_interaction():
    """get_magnon_bands must agree exactly with calling the lower-level
    rpa_kernel_poles_ops directly, using the same Sx,Sy,Sz operators and
    the same (now non-onsite) interaction dict -- mirrors
    tests/chi/test_magnon_bands.py's onsite-only version of this check,
    here specifically for a H.V with neighbor-shell (not just onsite)
    support, i.e. exercising interaction_at_q's dict-Fourier-transform
    path end to end."""
    h = _chain_with_nn_exchange(filling=0.1, J1=-1.5)
    energies = np.linspace(0.0002, 0.08, 60)
    q0 = [0.02, 0., 0.]

    qs, ws, gammas = h.get_magnon_bands(qpath=[q0], nq=1, energies=energies,
                                         delta=1e-3, nk=4000)

    Ss = _full_spin_operators(h)
    U = _full_spin_U(h)
    direct_poles = rpa_kernel_poles_ops(h, ops=Ss, V=U, q=q0,
                                         energies=energies, delta=1e-3, nk=4000)

    assert np.all(qs == 0)
    assert len(ws) == len(direct_poles)
    order = np.argsort(ws)
    direct_order = np.argsort(direct_poles[:, 0])
    assert np.allclose(ws[order], direct_poles[direct_order, 0])
    assert np.allclose(gammas[order], direct_poles[direct_order, 1])


def test_low_filling_enhances_ferromagnetic_instability():
    """Physical motivation for supporting non-onsite interactions in RPA:
    a 1D chain's density of states diverges at the bottom of the band, so
    for a FIXED nearest-neighbor ferromagnetic exchange J1 (J1<0, no
    onsite U at all), the static (omega->0) RPA kernel eigenvalue at a
    fixed small q must move monotonically closer to the Stoner
    instability (1 - J1(q)*chi0(q,0) -> 0) as the filling is lowered
    towards the band edge. This is the direct, robust signature of "low
    filling favours a ferromagnetic instability" -- checked as a
    monotonic trend in the static kernel eigenvalue (not by trying to
    resolve a dynamic pole's exact location, which is far more sensitive
    to numerical resolution)."""
    J1 = -1.0
    q = [0.05, 0., 0.]
    energies = np.array([0.01])

    fillings = [0.5, 0.3, 0.1]
    eigs = []
    for filling in fillings:
        h = _chain_with_nn_exchange(filling=filling, J1=J1)
        Ss = _full_spin_operators(h)
        U = _full_spin_U(h)
        es, chis = _chi_ops_matrix_vectorized(h, ops=Ss, q=q, energies=energies,
                                               delta=1e-2, nk=4000)
        Uq = interaction_at_q(U, h, q)
        iden = np.identity(chis[0].shape[0])
        eigs.append(np.min(np.linalg.eigvals(iden - Uq @ chis[0]).real))

    assert eigs[0] > eigs[1] > eigs[2], \
        f"expected the kernel eigenvalue to shrink monotonically as filling " \
        f"drops (more unstable towards ferromagnetism): {list(zip(fillings, eigs))}"
    # sanity: still comfortably on the stable side at half filling
    assert eigs[0] > 0.5
