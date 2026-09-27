"""The charge-channel RPA of h.get_densitychi_RPA/h.get_plasmon_bands takes
its interaction as U, V1, V2, V3 and Vr, and has to mean by them the same
Hamiltonian the mean field and the Bethe-Salpeter equation mean,

    H_int = U sum_i n_i,up n_i,dn + sum_shells V_s sum_<ij>_s n_i n_j

with each bond counted once. The RPA kernel of that Hamiltonian in the
charge channel is a(q) = U/2 + V(q), where V(q) = sum_d V_d exp(2 pi i q.d)
over the bond vectors d, so on a chain V(q) = 2 V1 cos(2 pi q).

chitk.densitychi._density_v used to build the neighbour shells as V1/2,
V2/2, V3/2, the halved values the SCF stores in h.V, but without the
doubling the SCF's get_mf_normal applies to them (bsetk.interaction.
bare_interaction explains that factor of two), so every neighbour-shell
coupling entered the charge RPA at half its strength: V(q) = V1 cos(2 pi q)
on a chain, and a CDW instability at twice the V1 the mean field and the
BSE put it at. Both tests below fail on that code.

The oracle is independent of _density_v: dropping the direct term from the
Bethe-Salpeter kernel leaves the time-dependent Hartree problem, which is
the RPA, built from the interaction of bsetk.interaction.density_interaction
(whose convention tests/bse/test_bse_interaction.py pins against the SCF).
Its charge collective mode has to be a zero of 1 - a(q) chi0(q,omega)."""
import numpy as np
import pytest

from pyqula import geometry
from pyqula.chi import chiAB
from pyqula.chitk.densitychi import _density_v
from pyqula.chitk.rpa import interaction_at_q
from pyqula.bsetk.interaction import density_interaction
from pyqula.bsetk.screening import spin_collapse

NK = 6
U, V1, V2, V3 = 0.3, 0.5, 0.2, 0.1


def _gapped_honeycomb(has_spin):
    h = geometry.honeycomb_lattice().get_hamiltonian(has_spin=has_spin)
    h.add_sublattice_imbalance(0.8)
    return h.get_multicell().get_dense()


def test_the_kernel_is_the_charge_channel_of_the_bse_interaction():
    """Spinless, the charge kernel is the interaction itself; spinful, it
    is the spin-summed interaction divided by four (bsetk.screening's v^c,
    whose onsite entry is U/2). Checked entry by entry, every shell."""
    h = _gapped_honeycomb(has_spin=False)
    a = _density_v(h, V1=V1, V2=V2, V3=V3)
    W = density_interaction(h, V1=V1, V2=V2, V3=V3)
    for d in set(a) | set(W):
        ad = np.array(a.get(d, 0.*W[d]))
        assert np.max(np.abs(ad - W.get(d, 0.*ad))) < 1e-12, d
    h = _gapped_honeycomb(has_spin=True)
    a = _density_v(h, U=U, V1=V1, V2=V2, V3=V3)
    W = density_interaction(h, U=U, V1=V1, V2=V2, V3=V3)
    ns = len(h.geometry.r)
    for d in set(a) | set(W):
        vc = spin_collapse(W[d], ns)/4. if d in W else 0.
        ad = np.array(a[d]) if d in a else 0.
        assert np.max(np.abs(ad - vc)) < 1e-12, d
    assert abs(a[(0, 0, 0)][0, 0] - U/2.) < 1e-12 # onsite enters as U/2


def _kernel_smallest_singular_value(h, a, q, w, delta):
    """Smallest singular value of 1 - a(q) chi0(q,w), charge channel"""
    _, chis = chiAB(h, mode="matrix", q=np.array(q), nk=NK,
                    energies=np.array([w]), delta=delta, T=1e-4)
    aq = interaction_at_q(a, h, np.array(q))
    m = np.identity(chis[0].shape[0], dtype=np.complex128) - aq @ chis[0]
    return np.min(np.abs(np.linalg.svd(m, compute_uv=False)))


@pytest.mark.parametrize("q", [[0., 0., 0.], [0.5, 0., 0.]])
def test_the_charge_mode_of_the_hartree_bse_is_a_zero_of_the_kernel(q):
    """Spinful, with U and V1 together, so that both the U/2 onsite entry
    and the per-bond V1 are checked. The spin modes of the exchange-only
    BSE feel only -U/2 and stay close to the bare transitions for a small
    U, while the charge kernel U/2 + V(q) is large in the staggered channel,
    so the eigenvalue furthest from every bare transition is the charge
    mode. pyqula's chi has its poles at e_a(k) - e_b(k+q), so the mode at
    +E is at -E."""
    h = _gapped_honeycomb(has_spin=True)
    W = density_interaction(h, U=U, V1=V1)
    a = _density_v(h, U=U, V1=V1)
    b = h.get_bse(V=W, Q=q, nk=NK, kernel="exchange")
    es = np.sort(b.get_energies().real)
    sep = np.array([np.min(np.abs(e - b.pairs.dE)) for e in es])
    E = es[np.argmax(sep)] # the charge collective mode
    assert np.max(sep) > 5e-2, "no charge mode away from the continuum"
    res = {d: _kernel_smallest_singular_value(h, a, q, -E, d)
           for d in (1e-5, 1e-6, 1e-7)}
    off = _kernel_smallest_singular_value(h, a, q, -E - 0.05, 1e-6)
    assert res[1e-6] < off / 100.
    assert abs(res[1e-5] / res[1e-6] - 10.) < 0.5 # an exact zero,
    assert abs(res[1e-6] / res[1e-7] - 10.) < 0.5 # limited by delta only
