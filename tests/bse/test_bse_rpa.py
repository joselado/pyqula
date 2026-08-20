"""Cross-check against chitk/rpa.py. Dropping the direct term from the BSE
kernel leaves the time-dependent Hartree problem, which is exactly the RPA
-- so every eigenvalue of the exchange-only full BSE must be a zero of the
RPA kernel det(1 - W chi0(omega)) that chitk/chiAB.py builds by an
independent frequency-scan route.

The comparison is made by evaluating the kernel at the BSE eigenvalue
rather than by locating poles on a frequency grid: chi0 is computed with a
finite broadening delta, so the kernel never reaches exactly zero, but its
smallest singular value there is proportional to delta and vanishes with
it. That proportionality is itself asserted below -- a wrong eigenvalue
would give a delta-independent residual instead.

Note the frequency convention: pyqula's chi has poles at
e_a(k) - e_b(k+q), so a BSE exciton at +E shows up in chi0 at -E."""
import numpy as np
import pytest

from pyqula import geometry, specialhopping
from pyqula.chi import chiAB
from pyqula.bsetk.interaction import interaction_at_q

NK = 6
V1 = 0.5


def _system():
    """A gapped spinless honeycomb: exactly one valence and one conduction
    band, so the BSE pair basis and chi0 contain the same electron-hole
    pairs. V1 is deliberately an extended (non-onsite) interaction, so
    W(q) genuinely depends on q."""
    h = geometry.honeycomb_lattice().get_hamiltonian(has_spin=False)
    h.add_sublattice_imbalance(0.8)
    h = h.get_multicell().get_dense()
    g = h.geometry
    mg = specialhopping.distance_hopping_matrix([V1], g.neighbor_distances()[0:1])
    W = g.get_hamiltonian(has_spin=False, is_multicell=True,
                          mgenerator=mg).get_hopping_dict()
    return h, W


def _kernel_smallest_singular_value(h, W, q, w, delta):
    """Smallest singular value of 1 - W(q) chi0(q,w); zero at an RPA pole"""
    _, chis = chiAB(h, mode="matrix", q=np.array(q), nk=NK,
                    energies=np.array([w]), delta=delta, T=1e-4)
    Vq = interaction_at_q(W, h.geometry, np.array(q))
    m = np.identity(chis[0].shape[0], dtype=np.complex128) - Vq @ chis[0]
    return np.min(np.abs(np.linalg.svd(m, compute_uv=False)))


@pytest.mark.parametrize("q", [[0., 0., 0.], [0.5, 0., 0.]])
def test_exchange_only_bse_eigenvalue_is_an_rpa_pole(q):
    h, W = _system()
    b = h.get_bse(V=W, Q=q, nk=NK, kernel="exchange")
    es = np.sort(b.get_energies().real)
    # take the collective mode: the eigenvalue pushed furthest away from
    # any bare transition, so that the finite broadening of chi0 does not
    # confuse it with a pole of chi0 itself
    sep = np.array([np.min(np.abs(e - b.pairs.dE)) for e in es])
    E = es[np.argmax(sep)]
    assert np.max(sep) > 10 * 1e-3, "no collective mode to test against"
    res = {d: _kernel_smallest_singular_value(h, W, q, -E, d)
           for d in (1e-5, 1e-6, 1e-7)}
    off = _kernel_smallest_singular_value(h, W, q, -E - 0.05, 1e-6)
    # on resonance the residual is delta-limited, and off resonance it is
    # orders of magnitude larger and delta-independent
    assert res[1e-6] < off / 100.
    # and it scales linearly with delta, i.e. the eigenvalue is an exact zero
    assert abs(res[1e-5] / res[1e-6] - 10.) < 0.5
    assert abs(res[1e-6] / res[1e-7] - 10.) < 0.5


def test_full_kernel_is_not_the_rpa():
    """A guard on the test above: with the direct term switched back on the
    spectrum must move, otherwise the cross-check would be passing for the
    trivial reason that the direct term does nothing."""
    h, W = _system()
    e_x = np.sort(h.get_exciton_energies(V=W, nk=NK, kernel="exchange").real)
    e_f = np.sort(h.get_exciton_energies(V=W, nk=NK, kernel="full").real)
    assert np.max(np.abs(e_x - e_f)) > 1e-2
