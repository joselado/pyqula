import numpy as np

from pyqula.qutecipytk.matrix.aca import MatrixACA


def test_3x3_real():
    A = np.array([
        [1.0, 0.1, -1.0],
        [-0.1, 2.0, -1.0],
        [0.5, 0.2, 0.3],
    ])

    aca = MatrixACA.from_matrix(A, (0, 0))

    assert aca.ncols() == 3
    assert aca.nrows() == 3
    assert aca.npivots() == 1
    assert aca.rowindices == [0]
    assert aca.colindices == [0]

    assert np.isclose(aca.evaluate(0, 0), A[0, 0])
    assert np.isclose(aca[0, 0], A[0, 0])
    assert np.allclose(aca.row(0), A[0, :])
    assert np.allclose(aca[0, :], A[0, :])
    assert np.allclose(aca.col(0), A[:, 0])
    assert np.allclose(aca[:, 0], A[:, 0])

    aca.add_pivot(A, (1, 2))

    assert aca.npivots() == 2
    assert aca.rowindices == [0, 1]
    assert aca.colindices == [0, 2]

    assert np.isclose(aca[1, 2], A[1, 2])
    assert np.isclose(aca.evaluate(1, 2), A[1, 2])
    assert np.allclose(aca[[0, 1], [0, 2]], A[np.ix_([0, 1], [0, 2])])
    assert np.allclose(aca.submatrix([0, 1], [0, 2]), A[np.ix_([0, 1], [0, 2])])

    aca.add_pivot(A)

    assert aca.npivots() == 3
    assert aca.rowindices == [0, 1, 2]
    assert aca.colindices == [0, 2, 1]

    assert np.allclose(aca.to_matrix(), A)
    assert np.allclose(aca[:, :], A)


def test_3x3_complex():
    A = np.array([
        [0.641325 + 0.331139j, 0.63414 + 0.902753j, 0.385012 + 0.359676j],
        [0.89194 + 0.783782j, 0.236955 + 0.0828438j, 0.98353 + 0.729723j],
        [0.219505 + 0.429946j, 0.544289 + 0.378888j, 0.14397 + 0.701327j],
    ])

    aca = MatrixACA.from_matrix(A, (0, 0))

    assert aca.ncols() == 3
    assert aca.nrows() == 3
    assert aca.npivots() == 1
    assert aca.rowindices == [0]
    assert aca.colindices == [0]

    aca.add_pivot(A)
    aca.add_pivot(A)

    assert np.allclose(aca.to_matrix(), A)
    assert np.allclose(aca[:, :], A)
