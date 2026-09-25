import numpy as np

from pyqula import geometry
from pyqula import embedding


def _dense(m):
    return np.array(m.todense()) if hasattr(m, "todense") else np.array(m)


def _rashba_chain_with_defect():
    g = geometry.chain()
    h = g.get_hamiltonian(has_spin=True)
    h.add_rashba(0.5)
    h.add_exchange([0.3, 0.5, 0.2])
    h.add_onsite(0.3)
    hd = h.copy()
    hd.add_onsite(1.0)
    hd.add_exchange([0.4, -0.6, 0.3])
    return h, hd


def _finite_chain_defect_block(h, hd, n):
    """Occupied projector on the defect cell of an n-cell open chain whose
    middle cell carries the defect onsite matrix, from exact diagonalization.
    Reversing the chain maps the middle cell onto itself, so the direction
    in which the hopping h.inter is laid down does not change this block."""
    intra, inter, m = _dense(h.intra), _dense(h.inter), _dense(hd.intra)
    d = intra.shape[0]
    H = np.zeros((n*d, n*d), dtype=np.complex128)
    for i in range(n):
        H[i*d:(i+1)*d, i*d:(i+1)*d] = intra
        if i < n-1:
            H[i*d:(i+1)*d, (i+1)*d:(i+2)*d] = inter
            H[(i+1)*d:(i+2)*d, i*d:(i+1)*d] = inter.conj().T
    c = n//2
    H[c*d:(c+1)*d, c*d:(c+1)*d] = m
    e, v = np.linalg.eigh(H)
    vo = v[c*d:(c+1)*d, e < 0.]
    return vo @ vo.conj().T


def test_defect_density_matrix_matches_finite_chain():
    """The contour integral of Embedding.get_density_matrix used to run at a
    fixed absolute tolerance of 1e-2 that no keyword reached, leaving an
    error of 3e-3 on the defect block that neither delta nor nk removed.
    With the default tolerance the diagonal of the defect block must agree
    with a long finite chain carrying the same defect; the 601-cell chain
    is within 4e-4 of a 2401-cell one. The diagonal does not depend on the
    index convention of the returned matrix."""
    h, hd = _rashba_chain_with_defect()
    eb = embedding.Embedding(h, m=hd.intra)
    dm = np.array(eb.get_density_matrix(delta=1e-3))
    rho = _finite_chain_defect_block(h, hd, 601)
    assert np.allclose(np.diag(dm).real, np.diag(rho).real, atol=1e-3, rtol=0.)
