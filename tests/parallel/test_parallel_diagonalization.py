import numpy as np
import numba

from pyqula import algebra
from pyqula.htk.eigenvectors import parallel_diagonalization, peigvalsh


def _random_hermitian_batch(n, nh, complex_valued=True):
    mats = []
    for _ in range(nh):
        m = np.random.random((n, n))
        if complex_valued:
            m = m + 1j * np.random.random((n, n))
        m = m + np.conjugate(m).T
        mats.append(m)
    return np.array(mats)


def test_parallel_diagonalization_matches_serial_eigh_complex():
    """The numba-prange batch diagonalization must agree with the
    ordinary (serial) algebra.eigh, matrix by matrix, for complex
    Hermitian input -- the case densitymatrix.full_dm_accumulate uses."""
    mats = _random_hermitian_batch(n=12, nh=6, complex_valued=True)
    es_p, vs_p = parallel_diagonalization(mats)
    for i, m in enumerate(mats):
        es_s, vs_s = algebra.eigh(m)
        order = np.argsort(es_s)
        assert np.max(np.abs(np.sort(es_p[i]) - es_s[order])) < 1e-8
        # eigenvectors can differ by a phase; compare |<v_p|v_s>| == 1
        for j in range(len(order)):
            v_s = vs_s[:, order[j]]
            v_p = vs_p[i][:, np.argsort(es_p[i])[j]]
            overlap = np.abs(np.vdot(v_s, v_p))
            assert abs(overlap - 1.0) < 1e-6


def test_parallel_diagonalization_matches_serial_eigh_real():
    """Same, for real symmetric input."""
    mats = _random_hermitian_batch(n=10, nh=5, complex_valued=False)
    es_p, vs_p = parallel_diagonalization(mats.astype(np.complex128))
    for i, m in enumerate(mats):
        es_s = np.linalg.eigvalsh(m)
        assert np.max(np.abs(np.sort(es_p[i]) - np.sort(es_s))) < 1e-8


def test_parallel_diagonalization_independent_of_thread_count():
    """Results must not depend on how many numba threads are used."""
    mats = _random_hermitian_batch(n=14, nh=8, complex_valued=True)
    default_threads = numba.get_num_threads()
    try:
        results = []
        for nthreads in (1, 2, 4):
            numba.set_num_threads(nthreads)
            es, vs = parallel_diagonalization(mats)
            results.append(np.sort(es, axis=1))
        for r in results[1:]:
            assert np.max(np.abs(r - results[0])) < 1e-8
    finally:
        numba.set_num_threads(default_threads)


def test_peigvalsh_matches_parallel_diagonalization_eigenvalues():
    mats = _random_hermitian_batch(n=8, nh=4, complex_valued=True)
    es_full, _ = parallel_diagonalization(mats)
    es_vals = peigvalsh(mats)
    assert np.max(np.abs(np.sort(es_full, axis=1) - np.sort(es_vals, axis=1))) < 1e-10
