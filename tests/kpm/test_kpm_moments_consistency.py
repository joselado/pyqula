import numpy as np

from pyqula import kpm


def test_python_and_numba_kpm_moments_agree():
    """The plain-Python and numba Chebyshev-moment implementations must
    produce the same moments for the same random sparse Hermitian matrix
    and starting vector."""
    n = 20
    m = np.random.random((n, n)) + 1j * np.random.random((n, n))
    m = m + np.conjugate(m).T
    m = m / np.max(np.abs(np.linalg.eigvalsh(m))) / 1.1  # rescale into (-1, 1)
    v = np.random.random(n) + 1j * np.random.random(n)
    v = v / np.sqrt(np.abs(np.vdot(v, v)))

    mus_python = kpm.python_kpm_moments(v, m, n=30)
    mus_numba = kpm.get_moments_v(v, m, n=30)

    assert np.max(np.abs(mus_python - mus_numba)) < 1e-6
