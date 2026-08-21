"""The low-rank factorization of the BSE kernel is EXACT.

bsetk/factorize.py rewrites every block of the BSE matrix as a diagonal
plus a sum of rank-one terms, one per non-zero entry of the real-space
interaction. Everything downstream of it -- the matrix-free iterative
solver and the quantics tensor-train solver -- is only as correct as that
rewrite, and it is exact rather than approximate, so it is tested against
the dense blocks kernel.build_blocks itself produces, to machine
precision.

The finite-Q cases are the ones that matter: the antiresonant block takes
W(-Q) and the coupling block's second pair index enters unconjugated, and
both are invisible at Q=0 (see kernel.build_blocks' comments). A
factorization that mixed them up would still pass every Q=0 check.
"""
import numpy as np
import pytest

from testutils import gapped_ionic_chain, gapped_honeycomb
from pyqula.bsetk.pairbasis import PairBasis
from pyqula.bsetk.interaction import bare_interaction, density_interaction
from pyqula.bsetk.kernel import build_blocks
from pyqula.bsetk.factorize import KernelFactorization

NK = 6


def _model(name):
    if name == "chain":
        h = gapped_ionic_chain()
        return h, dict(U=1.0, V1=0.5, V2=0.3)
    h = gapped_honeycomb(spinful=False)
    return h, dict(V1=0.6, V2=0.2)


@pytest.mark.parametrize("name", ["chain", "honeycomb"])
@pytest.mark.parametrize("Q", [[0., 0., 0.], [0.17, 0., 0.]])
@pytest.mark.parametrize("kernel", ["full", "direct", "exchange", "none"])
def test_factorization_reproduces_dense_blocks(name, Q, kernel):
    h, kw = _model(name)
    W = bare_interaction(h, V=density_interaction(h, **kw))
    pb = PairBasis(h, Q=Q, nk=NK)
    A, Abar, B = build_blocks(pb, W, Wx=W, kernel=kernel)
    for block, ref in (("A", A), ("Abar", Abar), ("B", B)):
        f = KernelFactorization(pb, W, Wx=W, kernel=kernel, block=block)
        err = np.max(np.abs(f.to_dense() - ref))
        scale = max(1., np.max(np.abs(ref)))
        assert err / scale < 1e-13, (name, Q, kernel, block, err)


def test_screened_direct_term_factorizes():
    """A screened direct term differs from the exchange term, which is
    what Wx is for. Truncated to a real-space dictionary it still
    factorizes exactly."""
    h = gapped_ionic_chain()
    Wx = bare_interaction(h, V=density_interaction(h, U=1.0, V1=0.5))
    # a different (here just rescaled and further-ranged) direct
    # interaction, standing in for a truncated screened one
    W = bare_interaction(h, V=density_interaction(h, U=0.7, V1=0.4, V2=0.2))
    pb = PairBasis(h, Q=[0.11, 0., 0.], nk=NK)
    A, Abar, B = build_blocks(pb, W, Wx=Wx, kernel="full")
    for block, ref in (("A", A), ("Abar", Abar), ("B", B)):
        f = KernelFactorization(pb, W, Wx=Wx, kernel="full", block=block)
        assert np.max(np.abs(f.to_dense() - ref)) < 1e-13, block


def test_matvec_matches_dense():
    h = gapped_ionic_chain()
    W = bare_interaction(h, V=density_interaction(h, U=1.0, V1=0.5))
    pb = PairBasis(h, Q=[0.11, 0., 0.], nk=NK)
    rng = np.random.default_rng(1234)
    for block in ("A", "Abar", "B"):
        f = KernelFactorization(pb, W, Wx=W, kernel="full", block=block)
        M = f.to_dense()
        x = rng.normal(size=pb.npair) + 1j * rng.normal(size=pb.npair)
        assert np.max(np.abs(f.matvec(x) - M @ x)) < 1e-12, block
        X = rng.normal(size=(pb.npair, 3)) + 0j  # several columns at once
        assert np.max(np.abs(f.matvec(X) - M @ X)) < 1e-12, block
        assert np.max(np.abs(f.diagonal() - np.diag(M))) < 1e-12, block


def test_rank_is_independent_of_the_mesh():
    """The point of the whole construction: the number of rank-one terms
    is set by the interaction, not by nk."""
    h = gapped_ionic_chain()
    W = bare_interaction(h, V=density_interaction(h, U=1.0, V1=0.5))
    nterms = []
    for nk in (4, 8, 16, 32):
        pb = PairBasis(h, Q=[0., 0., 0.], nk=nk)
        nterms.append(KernelFactorization(pb, W, kernel="full").nterm)
    assert len(set(nterms)) == 1, nterms


def test_tabulated_interaction_is_refused():
    """A tabulated screened interaction has no fixed-rank factorization;
    it must say so rather than silently doing something expensive."""
    from pyqula.bsetk.screening import screened_interaction
    h = gapped_ionic_chain()
    V = density_interaction(h, U=1.0, V1=0.5)
    S = screened_interaction(h, V=V, nk=4)
    pb = PairBasis(h, Q=[0., 0., 0.], nk=4)
    with pytest.raises(ValueError, match="real-space"):
        KernelFactorization(pb, S, Wx=None, kernel="direct")
