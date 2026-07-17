import os
import subprocess
import sys
import textwrap

_SRC = os.path.join(os.path.dirname(__file__), "..", "..", "src")

# numba's default threading layer (tbb, where available) does not survive
# fork(): running a @jit(parallel=True)/prange function in the main process
# initializes tbb's thread pool, and forking a worker process afterward
# (parallel.set_cores -> multiprocess.Pool, which forks on Linux) then
# deadlocks -- confirmed by hand before parallel.py started forcing the
# fork-safe 'workqueue' threading layer. This runs the scenario in a real
# subprocess with a hard timeout, since a regression here hangs forever
# rather than raising, and pytest has no built-in timeout of its own.

_SCRIPT = textwrap.dedent("""
    import sys
    sys.path.insert(0, {src!r})
    import numpy as np
    from pyqula.htk.eigenvectors import parallel_diagonalization
    from pyqula import parallel

    mats = np.random.random((4, 10, 10)) + 1j * np.random.random((4, 10, 10))
    mats = mats + np.conjugate(np.transpose(mats, (0, 2, 1)))
    _ = parallel_diagonalization(mats)  # initializes numba's threading layer

    parallel.set_cores(4)  # forks worker processes
    out = parallel.pcall(lambda x: x * 2, [1, 2, 3, 4])
    parallel.set_cores(1)
    assert out == [2, 4, 6, 8]
    print("OK")
""")


def test_prange_then_pool_creation_does_not_deadlock():
    script = _SCRIPT.format(src=os.path.abspath(_SRC))
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, (
        f"subprocess failed (returncode={result.returncode}):\n"
        f"stdout={result.stdout}\nstderr={result.stderr}"
    )
    assert "OK" in result.stdout
