import numpy as np
import pytest

from pyqula.qutecipytk.matrix.rrlu import arrlu, rrlu, rrlu_inplace, submatrixargmax

rng = np.random.default_rng(1234)


def test_submatrixargmax():
    A = rng.random((10, 8))
    assert submatrixargmax(A, 0) == np.unravel_index(np.argmax(A), A.shape)
    assert submatrixargmax(A, 3, f=lambda x: x) == tuple(
        3 + np.array(np.unravel_index(np.argmax(A[3:, 3:]), A[3:, 3:].shape))
    )


def test_submatrixargmax_complex():
    A = np.array([
        [0, 1, 2, 3, 4, 5],
        [1j, 2 + 1j, 3 + 1j, 4 + 1j, 5 + 1j, 6 + 1j],
        [1 + 2j, 2 + 2j, 3 + 2j, 4 + 2j, 5 + 2j, 6 + 2j],
    ])
    got = submatrixargmax(A, 0, f=lambda x: np.abs(x) ** 2)
    expect = np.unravel_index(np.argmax(np.abs(A) ** 2), A.shape)
    assert got == expect


def test_rrlu_basic():
    A = rng.random((4, 4))
    lu = rrlu(A)
    assert lu.size() == A.shape
    L = lu.left(permute=False)
    assert np.allclose(np.tril(L, -1), L - np.eye(*L.shape))
    U = lu.right(permute=False)
    assert np.allclose(np.triu(U), U)
    assert np.allclose(lu.left() @ lu.right(), A)


def test_arrlu_basic():
    A = rng.random((4, 4))
    lu = arrlu(np.float64, lambda i, j: A[i, j], A.shape, [0], [0])
    assert lu.size() == A.shape
    L = lu.left(permute=False)
    assert np.allclose(np.tril(L, -1), L - np.eye(*L.shape))
    U = lu.right(permute=False)
    assert np.allclose(np.triu(U), U)
    assert np.allclose(lu.left() @ lu.right(), A)


def test_arrlu_matches_full_reconstruction_various_sizes():
    for shape in [(5, 6), (6, 5), (8, 8), (3, 10)]:
        A = rng.random(shape)
        lu_full = rrlu(A)
        lu_rook = arrlu(np.float64, lambda i, j: A[i, j], A.shape, [0], [0], numrookiter=6)
        assert np.allclose(lu_full.left() @ lu_full.right(), A)
        assert np.allclose(lu_rook.left() @ lu_rook.right(), A)


def test_truncated_rank_revealing_lu():
    A = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    ])
    lu = rrlu(A)
    assert lu.npivot == 1


def test_approximation_in_rrlu():
    A = rng.random((8, 6))
    lu = rrlu(A, maxrank=4)
    assert lu.size() == A.shape
    assert len(lu.rowindices()) == 4
    assert len(lu.colindices()) == 4

    L = lu.left(permute=False)
    assert L.shape == (A.shape[0], 4)
    assert np.allclose(L, np.tril(L))
    U = lu.right(permute=False)
    assert U.shape == (4, A.shape[1])
    assert np.allclose(U, np.triu(U))

    A2 = np.hstack([A, A + 1e-3 * rng.random((8, 6))])
    lu2 = rrlu(A2, reltol=1e-2)
    assert lu2.size() == A2.shape
    assert len(lu2.rowindices()) < A2.shape[0]
    assert len(lu2.colindices()) < A2.shape[1]
    assert np.max(np.abs(lu2.left() @ lu2.right() - A2)) < 1e-2


def test_rrlu_exact_low_rank():
    p = rng.random((10, 3))
    q = rng.random((3, 10))
    A = p @ q
    lu = rrlu(A)
    assert lu.npivots() == 3
    assert np.allclose(lu.left() @ lu.right(), A)


def test_lastpivoterror_full_rank():
    A = np.eye(2)
    lu = rrlu(A)
    assert np.allclose(lu.pivoterrors(), [1.0, 1.0, 0.0])
    assert lu.lastpivoterror() == 0.0


def test_lastpivoterror_limited():
    A = rng.random((5, 5))
    lu = rrlu(A, maxrank=2)
    assert len(lu.pivoterrors()) == 3
    assert lu.lastpivoterror() > 0

    lu2 = rrlu(A, abstol=0.5)
    assert lu2.lastpivoterror() < 0.5

    lu3 = rrlu(A, abstol=0.0)
    assert lu3.lastpivoterror() == 0.0


def test_lu_very_small_values():
    A = 1e-13 * rng.random((4, 4))
    lu = rrlu(A, abstol=1e-3)
    assert lu.npivots() == 1
    assert len(lu.pivoterrors()) > 0
    assert lu.lastpivoterror() > 0
    assert lu.size() == A.shape
    assert np.max(np.abs(lu.left() @ lu.right() - A)) < 1e-3


def test_transpose():
    A = rng.random((5, 10))
    tlu = rrlu(A).transpose()
    assert np.allclose(tlu.left() @ tlu.right(), A.T)


def test_solve_by_rrlu():
    N = 5
    M = 2
    L = np.tril(rng.random((N, N)))
    U = np.triu(rng.random((N, N)))
    b = rng.random((N, M))

    A = L @ U
    lua = rrlu(A)
    assert np.allclose(lua.left() @ lua.right(), A)
    assert np.allclose(A @ lua.solve(b), b)
