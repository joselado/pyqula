import numpy as np

from pyqula import algebra


def test_smalleig_dense_fallback_returns_row_eigenvectors():
    """Regression check for algebra.smalleig: when ARPACK's eigsh raises
    (e.g. k too close to N, or a singular shift-invert factorization) it
    falls back to a dense eigh, but returned eigh's raw output
    (eigenvectors as columns) without the .transpose() that both the
    ARPACK-success branch and the num_bands=None caller (waves.py) apply
    -- every consumer (e.g. ldos_waves, which iterates `for v in
    eigvec`) expects eigvec[j] to be the j-th eigenvector (row-major),
    so the untransposed fallback silently fed callers a transposed
    matrix instead. Force the fallback with k=N-1 (triggers ARPACK's own
    "k >= N-1" error path) and check the returned eigenvectors actually
    diagonalize the matrix, not just have the right shape."""
    rng = np.random.default_rng(0)
    n = 8
    a = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    m = a + a.conj().T  # Hermitian
    eig, eigvec = algebra.smalleig(m, numw=n - 1, evecs=True, e0=0.0)
    assert eigvec.shape[1] == n  # row-major: each row must have length n
    for e, v in zip(eig, eigvec):
        v = v / np.linalg.norm(v)
        resid = m @ v - e * v
        assert np.max(np.abs(resid)) < 1e-8
