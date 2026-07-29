import numba
import numpy as np

from pyqula import parallel
from pyqula.htk.eigenvectors import parallel_diagonalization


def test_set_enabled_false_forces_serial_pcall_even_if_cores_requested():
    """Once disabled, requesting a multi-process pool must be a no-op:
    pcall must run serially and parallel.cores must read back as 1."""
    try:
        parallel.set_enabled(False)
        parallel.set_cores(4) # must be ignored while disabled
        assert parallel.cores == 1
        out = parallel.pcall(lambda x: x * 2, [1, 2, 3, 4])
        assert out == [2, 4, 6, 8]
    finally:
        parallel.set_enabled(True)
        parallel.set_cores(1)


def test_set_enabled_false_clamps_numba_threads_to_one():
    """Disabling must force numba's global thread count to 1, so
    @jit(parallel=True) kernels run single-threaded."""
    default_threads = numba.get_num_threads()
    try:
        parallel.set_enabled(False)
        assert numba.get_num_threads() == 1
    finally:
        parallel.set_enabled(True)
        numba.set_num_threads(default_threads)


def test_disabled_results_match_enabled_results():
    """Disabling parallelism must change only how results are computed,
    not the results themselves -- mirrors the existing
    test_parallel_diagonalization_independent_of_thread_count."""
    mats = np.random.random((6, 8, 8)) + 1j * np.random.random((6, 8, 8))
    mats = mats + np.conjugate(np.transpose(mats, (0, 2, 1)))
    default_threads = numba.get_num_threads()
    try:
        parallel.set_enabled(True)
        es_enabled, _ = parallel_diagonalization(mats)
        parallel.set_enabled(False)
        es_disabled, _ = parallel_diagonalization(mats)
    finally:
        parallel.set_enabled(True)
        numba.set_num_threads(default_threads)
    diff = np.abs(np.sort(es_enabled, axis=1) - np.sort(es_disabled, axis=1))
    assert np.max(diff) < 1e-8


def test_set_enabled_true_does_not_auto_restore_previous_core_count():
    """Re-enabling only lifts the restriction -- it must not silently
    resurrect whatever pool size was configured before disabling."""
    try:
        parallel.set_cores(3)
        parallel.set_enabled(False)
        assert parallel.cores == 1
        parallel.set_enabled(True)
        assert parallel.cores == 1 # still 1, not auto-restored to 3
    finally:
        parallel.set_enabled(True)
        parallel.set_cores(1)
