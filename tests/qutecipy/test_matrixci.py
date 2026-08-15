import numpy as np
import pytest

from pyqula.qutecipytk.matrix.aca import MatrixCI, a_inv_times_b, a_times_binv, crossinterpolate_matrix

rng = np.random.default_rng(42)


def test_a_times_binv_identity():
    A = rng.random((5, 3))
    eye3 = np.eye(3)
    eye5 = np.eye(5)
    assert np.allclose(A, a_times_binv(A, eye3))
    assert np.allclose(A, a_inv_times_b(eye5, A))

    B = rng.random((3, 3))
    C = rng.random((5, 5))
    assert np.allclose(eye3, a_times_binv(B, B))
    assert np.allclose(eye3, a_inv_times_b(B, B))
    assert np.allclose(eye5, a_times_binv(C, C))
    assert np.allclose(eye5, a_inv_times_b(C, C))

    assert np.allclose(A @ np.linalg.inv(B), a_times_binv(A, B))
    assert np.allclose(np.linalg.inv(C) @ A, a_inv_times_b(C, A))


def test_empty_constructor():
    ci = MatrixCI.empty(np.float64, 10, 25)
    assert ci.rowindices == []
    assert ci.colindices == []
    assert np.array_equal(ci.pivotcols, np.zeros((10, 0)))
    assert np.array_equal(ci.pivotrows, np.zeros((0, 25)))
    assert ci.nrows() == 10
    assert ci.ncols() == 25
    assert ci.size() == (10, 25)
    assert ci.rank() == 0
    assert np.allclose(ci[:, :], np.zeros((10, 25)))
    for i in range(10):
        assert np.allclose(ci.row(i), np.zeros(25))
        assert np.allclose(ci[i, :], np.zeros(25))
    for j in range(25):
        assert np.allclose(ci.col(j), np.zeros(10))
        assert np.allclose(ci[:, j], np.zeros(10))


def test_full_constructor():
    A = rng.random((8, 5))
    rowindices = [7, 1, 2]
    colindices = [0, 4, 3]

    ci = MatrixCI(rowindices, colindices, A[:, colindices], A[rowindices, :])

    assert ci.rowindices == rowindices
    assert ci.colindices == colindices
    assert np.array_equal(ci.pivotcols, A[:, colindices])
    assert np.array_equal(ci.pivotrows, A[rowindices, :])
    assert ci.nrows() == 8
    assert ci.ncols() == 5
    assert ci.size() == A.shape
    assert ci.rank() == 3

    Apivot = A[np.ix_(rowindices, colindices)]
    assert np.array_equal(ci.pivotmatrix(), Apivot)
    assert np.allclose(ci.leftmatrix(), A[:, colindices] @ np.linalg.inv(Apivot))
    assert np.allclose(ci.rightmatrix(), np.linalg.inv(Apivot) @ A[rowindices, :])

    assert ci.available_rows() == [0, 3, 4, 5, 6]
    assert ci.available_cols() == [1, 2]

    for i in rowindices:
        for j in colindices:
            assert np.isclose(ci.evaluate(i, j), A[i, j])

    for i in rowindices:
        assert np.allclose(ci.row(i)[colindices], A[i, colindices])
    for j in colindices:
        assert np.allclose(ci.col(j)[rowindices], A[rowindices, j])

    assert np.allclose(ci.submatrix(rowindices, colindices), Apivot)
    assert np.allclose(ci.to_matrix()[np.ix_(rowindices, colindices)], Apivot)


def test_finding_pivots_trivial():
    A = np.full((5, 3), 1.0)
    ci = MatrixCI.empty(np.float64, *A.shape)

    with pytest.raises(ValueError):
        ci.add_pivot(np.zeros((6, 6)))
    with pytest.raises((IndexError, ValueError)):
        ci.add_pivot(A, (5, 2))
    with pytest.raises((IndexError, ValueError)):
        ci.add_pivot(A, (4, 3))
    with pytest.raises((IndexError, ValueError)):
        ci.add_pivot(A, (-1, 1))
    with pytest.raises(ValueError):
        ci.find_new_pivot(A, [], [1, 2])
    with pytest.raises(ValueError):
        ci.find_new_pivot(A, [0, 1], [])

    assert ci.rank() == 0

    ci.add_pivot(A, (1, 2))
    assert ci.rowindices == [1]
    assert ci.colindices == [2]
    assert np.array_equal(ci.pivotrows, np.full((1, 3), 1.0))
    assert np.array_equal(ci.pivotcols, np.full((5, 1), 1.0))
    assert ci.rank() == 1
    for i in range(5):
        for j in range(3):
            assert np.isclose(ci.evaluate(i, j), 1.0)

    ci.add_pivot(A)
    assert ci.pivotrows.shape == (2, 3)
    assert ci.pivotcols.shape == (5, 2)
    assert ci.rank() == 2
    ci.add_pivot(A, (ci.available_rows()[0], ci.available_cols()[0]))
    assert ci.pivotrows.shape == (3, 3)
    assert ci.pivotcols.shape == (5, 3)
    assert ci.rank() == 3


def test_finding_pivots_rank1():
    A = np.outer([1.0, 2.0, 3.0], [2.0, 4.0, 8.0, 16.0])
    ci = MatrixCI.empty(np.float64, 3, 4)

    assert np.allclose(ci.local_error(A), A)
    pivot, err = ci.find_new_pivot(A)
    assert pivot == (2, 3)
    assert np.isclose(err, 48.0)
    ci.add_pivot(A)

    assert ci.rowindices == [2]
    assert ci.colindices == [3]
    assert np.allclose(ci.pivotrows, 3.0 * np.array([[2.0, 4.0, 8.0, 16.0]]))
    assert np.allclose(ci.pivotcols, 16.0 * np.array([[1.0], [2.0], [3.0]]))
    assert np.allclose(ci[:, :], A)
    assert ci.available_rows() == [0, 1]
    assert ci.available_cols() == [0, 1, 2]

    ci.add_pivot(A)
    assert len(ci.rowindices) == 2
    assert len(ci.colindices) == 2
    assert len(set(ci.rowindices)) == 2
    assert len(set(ci.colindices)) == 2
    assert ci.pivotrows.shape == (2, 4)
    assert ci.pivotcols.shape == (3, 2)
    assert np.allclose(ci[:, :], A)

    ci.add_pivot(A)
    assert len(ci.rowindices) == 3
    assert len(ci.colindices) == 3

    with pytest.raises(ValueError):
        ci.find_new_pivot(A)
    with pytest.raises(ValueError):
        ci.add_pivot(A)


def test_crossinterpolate_smooth_functions():
    grid = np.linspace(0, 1, 21)

    gauss = np.exp(-grid[:, None] ** 2 - grid[None, :] ** 2)
    ci_gauss = crossinterpolate_matrix(gauss)
    assert ci_gauss.rank() == 1
    assert ci_gauss.nrows() == 21
    assert ci_gauss.ncols() == 21
    assert np.allclose(ci_gauss[:, :], gauss, atol=1e-5)

    lorentz = 1.0 / (1.0 + grid[:, None] ** 2 + grid[None, :] ** 2)
    ci_lorentz = crossinterpolate_matrix(lorentz, tolerance=1e-6, maxiter=10)
    assert ci_lorentz.rank() == 5
    assert np.allclose(ci_lorentz[:, :], lorentz, atol=1e-5)
