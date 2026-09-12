import numpy as np

from pyqula import geometry
from pyqula.sctk import superfluidweight as sfw


def test_diamagnetic_term_equals_the_explicit_three_operand_contraction():
    """The diamagnetic piece of the Kubo superfluid weight is
    sum_i n_F(E_i) [w^dag B w]_ii. Written as one gemm and a reduction it
    must reproduce the explicit three-index contraction it replaces; the
    two differ only in the order of the floating point sums, so the check
    is a tolerance on the same quantity, not on a recorded number."""
    rng = np.random.default_rng(3)
    n = 24
    a = rng.normal(size=(n, n)) + 1j*rng.normal(size=(n, n))
    ws = np.linalg.qr(a)[0]              # a unitary set of eigenvectors
    b = rng.normal(size=(n, n)) + 1j*rng.normal(size=(n, n))
    B = b + np.conjugate(b.T)            # B is Hermitian by construction
    es = np.sort(rng.normal(size=n))
    for T in [0., 0.1]:
        nf = sfw._fermi(es, T)
        wsc = np.conjugate(ws)
        ref = np.sum(nf*np.einsum("ij,jk,ki->i", wsc.T, B, ws)).real
        (para, dia) = sfw._superfluid_weight_at(es, ws, [B], {(0, 0): B}, T, 1)
        assert abs(dia[0, 0] - ref) < 1e-12*max(1., abs(ref)), (T, dia, ref)


def test_superfluid_weight_still_matches_the_finite_difference_oracle():
    """End-to-end, gauge-invariant control on the whole Kubo route: the
    analytic tensor must keep agreeing with a brute-force second derivative
    of the grand potential with respect to the twist, which shares none of
    the contraction code."""
    g = geometry.square_lattice()
    h = g.get_hamiltonian()
    h.add_onsite(-0.7)
    h.add_swave(0.35)
    kubo = h.get_superfluid_weight(nk=8)
    fd = h.get_superfluid_weight(nk=8, mode="finite_difference")
    assert np.allclose(kubo, fd, atol=1e-6), (kubo, fd)
