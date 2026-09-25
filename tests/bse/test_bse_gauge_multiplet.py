"""The projection gauge may rotate only inside a degenerate multiplet.

The pair basis keeps the one-body part as the diagonal
dE = e_c(k+Q) - e_v(k) on the band labels, which stays the one-body part
of the rotated states only while <w_n|H(k)|w_m> is diagonal. A rotation
that mixes two bands of different energy breaks that, and the matrix that
gets diagonalized is then no longer a unitary transform of the BSE block.
Every model in test_bse_gauge.py has windows that are either exactly
degenerate or one band wide, where the two coincide, so the models here
are the ones where they do not: a spinful chain with Rashba coupling and
an exchange field along a generic direction, which leaves every band
non-degenerate with two bands in each window, and the Rashba chain alone,
whose bands are degenerate only at the time-reversal invariant momenta.
"""
import numpy as np
import pytest

from testutils import gapped_ionic_chain
from pyqula import geometry
from pyqula.bsetk.interaction import density_interaction
from pyqula.bsetk.pairbasis import PairBasis
from pyqula.bsetk.gauge import default_trials


def rashba_chain(exchange=None):
    """Spinful ionic chain with Rashba coupling and, optionally, an
    exchange field; nv=nc=2 is the whole window"""
    g = geometry.chain().supercell(2)
    h = g.get_hamiltonian(has_spin=True)
    h.add_onsite(lambda r: 0.9 * (-1) ** int(round(r[0] - 0.5)))
    h.add_rashba(0.3)
    if exchange is not None: h.add_exchange(exchange)
    return h.get_multicell().get_dense()


MODELS = {"rashba+exchange": lambda: rashba_chain([0.3, 0.5, 0.2]),
          "rashba": lambda: rashba_chain()}


@pytest.mark.parametrize("Q", [[0., 0., 0.], [0.25, 0., 0.]])
@pytest.mark.parametrize("model", sorted(MODELS))
def test_projection_gauge_leaves_the_spectrum_alone(model, Q):
    h = MODELS[model]()
    W = density_interaction(h, U=1.0, V1=0.3)
    kw = dict(V=W, Q=Q, nk=8, tda=True, kernel="full", nv=2, nc=2)
    ref = np.sort(h.get_bse(gauge=None, **kw).get_energies().real)
    got = np.sort(h.get_bse(gauge="projection", **kw).get_energies().real)
    assert np.max(np.abs(got - ref)) < 1e-10, (model, Q)


def test_one_body_part_stays_diagonal():
    """<w_n|H(k)|w_m> inside each window must be diag(ek), at k and at
    k+Q, which is what lets dE stay a diagonal"""
    h = rashba_chain([0.3, 0.5, 0.2])
    pb = PairBasis(h, Q=[0.25, 0., 0.], nk=8, nv=2, nc=2,
                   gauge="projection")
    hk = h.get_hk_gen()
    for ik, k in enumerate(pb.kpoints):
        for c, e, kk in ((pb.ck, pb.ek, k), (pb.ckq, pb.ekq, k + pb.Q)):
            for grp in (pb.vbands, pb.cbands):
                w = c[ik][grp]
                m = np.conj(w) @ hk(kk) @ w.T
                assert np.max(np.abs(m - np.diag(e[ik][grp]))) < 1e-10, ik


def test_degenerate_window_is_still_rotated():
    """On the non-magnetic spinful chain every window is one two-fold
    multiplet, so the gauge must still rotate it as a block: the overlap
    of the gauged states with their trial orbitals is then Hermitian and
    positive, the defining property of U = A(A^dag A)^-1/2, which a phase
    fix band by band would not give"""
    h = gapped_ionic_chain()
    raw = PairBasis(h, nk=8)
    pb = PairBasis(h, nk=8, gauge="projection")
    groups = [pb.vbands, pb.cbands]
    trials = default_trials(raw.ck, groups)
    mixed = 0.
    for ik in range(len(pb.kpoints)):
        for grp, t in zip(groups, trials):
            a = np.conj(pb.ck[ik][grp]) @ t
            assert np.max(np.abs(a - a.conj().T)) < 1e-12, ik
            assert np.min(np.linalg.eigvalsh(a)) > 0., ik
            u = np.conj(raw.ck[ik][grp]) @ pb.ck[ik][grp].T
            assert np.max(np.abs(u.conj().T @ u - np.eye(len(grp)))) < 1e-12
            mixed = max(mixed, np.max(np.abs(u - np.diag(np.diag(u)))))
    assert mixed > 0.1, mixed  # a genuine rotation, not a phase


def test_qtt_default_gauge_matches_iterative():
    """solver="qtt" turns the projection gauge on by default, so on a
    model with non-degenerate bands in a window it returned a lowest
    exciton 0.014 below the exact one before the gauge was restricted"""
    pytest.importorskip("dmrgpy", reason="the quantics BSE solver uses "
                        "dmrgpy.pyitensor's DMRG")
    h = rashba_chain([0.3, 0.5, 0.2])
    W = density_interaction(h, U=1.0, V1=0.3)
    kw = dict(V=W, nk=16, tda=True)
    ref = h.get_bse(solver="iterative", neig=1, **kw).get_energies()[0].real
    got = h.get_bse(solver="qtt", neig=1, tolerance=1e-8, coarse_nk=8,
                    **kw).get_energies()[0].real
    assert abs(got - ref) < 1e-6, (got, ref)
