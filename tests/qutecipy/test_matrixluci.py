import numpy as np

from pyqula.qutecipytk.matrix.aca import MatrixCI
from pyqula.qutecipytk.matrix.luci import MatrixLUCI

rng = np.random.default_rng(7)


def test_approximation_in_luci():
    A = rng.random((8, 6))

    luci = MatrixLUCI.from_matrix(A, maxrank=4)
    assert luci.size() == A.shape
    assert len(luci.rowindices()) == 4
    assert len(luci.colindices()) == 4

    ci = MatrixCI(luci.rowindices(), luci.colindices(), A[:, luci.colindices()], A[luci.rowindices(), :])
    assert np.allclose(luci.colstimespivotinv(), ci.leftmatrix())
    assert np.allclose(luci.pivotinvtimesrows(), ci.rightmatrix())

    L = luci.left()
    assert L.shape == (A.shape[0], 4)
    U = luci.right()
    assert U.shape == (4, A.shape[1])
    assert np.allclose(L @ U, ci[:, :])

    A2 = np.hstack([A, A + 1e-3 * rng.random((8, 6))])
    luci2 = MatrixLUCI.from_matrix(A2, reltol=1e-2)
    assert luci2.size() == A2.shape
    assert len(luci2.rowindices()) < A2.shape[0]
    assert len(luci2.colindices()) < A2.shape[1]
    assert np.max(np.abs(luci2.left() @ luci2.right() - A2)) < 1e-2


def test_luci_exact_low_rank():
    p = rng.random((10, 3))
    q = rng.random((3, 10))
    A = p @ q
    luci = MatrixLUCI.from_matrix(A)

    assert luci.npivots() == 3
    assert np.allclose(luci.left() @ luci.right(), A)
    pivotmatrix = luci.colmatrix()[: luci.npivots(), :]
    assert np.linalg.cond(pivotmatrix) < 1e12


def test_luci_not_leftorthogonal():
    A = rng.random((8, 6))
    luci = MatrixLUCI.from_matrix(A, leftorthogonal=False, maxrank=4)
    L = luci.left()
    U = luci.right()
    assert L.shape == (A.shape[0], 4)
    assert U.shape == (4, A.shape[1])
    ci = MatrixCI(luci.rowindices(), luci.colindices(), A[:, luci.colindices()], A[luci.rowindices(), :])
    assert np.allclose(L @ U, ci[:, :])
