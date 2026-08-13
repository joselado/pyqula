import numpy as np
import pytest

from pyqula import geometry
from pyqula.scftk.spinspin import _build_v
from pyqula.chitk.spinchi import _full_spin_operators, V2K_matrix, replicateU
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


def _nononsite_spin_U(h):
    """Build the same vertex _full_spin_U(h) would (V2K_matrix/replicateU,
    the +2 prefactor), bypassing chitk.spinchi._require_onsite_only_V's
    guard -- h.V here is deliberately non-onsite, which that guard now
    rejects at the public API (get_magnon_bands/get_spinchi_full) because
    non-onsite spin-channel RPA isn't properly verified in general (see
    that function's docstring). These tests intentionally exercise the
    underlying vertex/Fourier-transform math directly, the same way this
    module's guard-bypassing pattern is documented and recommended for."""
    return {d: 2*replicateU(V2K_matrix(m), n=3) for d, m in h.V.items()}


def test_multishell_interaction_has_more_than_one_key():
    """Sanity check on the test fixture itself: the whole point is that
    H.V is NOT onsite-only."""
    h = _chain_with_nn_exchange(filling=0.1, J1=-1.0)
    assert len(h.V) > 1
    assert (0, 0, 0) in h.V


def test_get_magnon_bands_raises_for_nononsite_interaction():
    """get_magnon_bands must raise ValueError for this hand-built,
    genuinely non-onsite H.V too (not just an SCF-derived one, see
    tests/scf/test_rpa_nononsite_ferro_chain.py) -- non-onsite spin-channel
    RPA is not yet properly verified, see
    chitk.spinchi._require_onsite_only_V's docstring."""
    h = _chain_with_nn_exchange(filling=0.1, J1=-1.5)
    energies = np.linspace(0.0002, 0.08, 60)
    with pytest.raises(ValueError):
        h.get_magnon_bands(qpath=[[0.02, 0., 0.]], nq=1, energies=energies,
                            delta=1e-3, nk=4000)


def test_direct_kernel_poles_are_finite_for_nononsite_interaction():
    """Keeps regression coverage for interaction_at_q's dict-Fourier-
    transform path (the actual thing this fixture exercises) by calling
    rpa_kernel_poles_ops directly with a manually-built vertex, bypassing
    the now-guarded get_magnon_bands/_full_spin_U -- see
    _nononsite_spin_U's docstring."""
    h = _chain_with_nn_exchange(filling=0.1, J1=-1.5)
    energies = np.linspace(0.0002, 0.08, 60)
    q0 = [0.02, 0., 0.]

    Ss = _full_spin_operators(h)
    U = _nononsite_spin_U(h)
    poles = rpa_kernel_poles_ops(h, ops=Ss, V=U, q=q0,
                                  energies=energies, delta=1e-3, nk=4000)
    assert poles.shape[1] == 2
    assert np.all(np.isfinite(poles))


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
    to numerical resolution).

    Builds the vertex directly (bypassing the guard, see
    _nononsite_spin_U's docstring) since this deliberately exercises a
    non-onsite interaction."""
    J1 = -1.0
    q = [0.05, 0., 0.]
    energies = np.array([0.01])

    fillings = [0.5, 0.3, 0.1]
    eigs = []
    for filling in fillings:
        h = _chain_with_nn_exchange(filling=filling, J1=J1)
        Ss = _full_spin_operators(h)
        U = _nononsite_spin_U(h)
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
