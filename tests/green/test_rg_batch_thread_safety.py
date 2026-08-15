"""Concurrent entry into the numba prange Sancho-Rubio batch kernel.

numba's 'workqueue' threading layer -- which parallel.py selects package-wide
for fork-safety (tbb deadlocks across multiprocess.Pool's fork) -- is NOT
threadsafe. Entering a parallel=True kernel from two Python threads at once
aborts the interpreter outright:

    Fatal Python error: Aborted
    Current thread ... [ThreadPoolExecu]
      greentk/rg.py:... in green_renormalization_jit_batch
      transporttk/selfenergy.py:... in get_selfenergy_batch
      aaatk/selfenergy_aaa.py:... in full_matrix_many
      keldyshtk/current.py:... in build_one

That is reachable from shipped defaults: keldyshtk/current.py's
build_selfenergy_aaa builds the two leads' AAA interpolants concurrently in a
ThreadPoolExecutor (4a086f5), AAA is the unconditional default for every
sweep entry point, and both threads land in the same kernel. Being a race it
fires intermittently -- it was first seen killing a `pytest tests/keldysh` run
partway through, which is exactly the kind of failure that gets written off as
a flake.

greentk.rg guards the kernel with a lock. This test drives many threads
through it at once; without the guard it aborts the whole interpreter (so a
regression shows up as the test session dying, not as a normal failure).
"""
import numpy as np
from concurrent.futures import ThreadPoolExecutor

from pyqula.greentk.rg import green_renormalization_jit_batch


def _lead_matrices(n=4):
    rng = np.random.default_rng(0)
    a = rng.normal(size=(n, n)) + 1j*rng.normal(size=(n, n))
    intra = (a + a.conj().T)/2.0          # Hermitian onsite block
    inter = 0.3*(rng.normal(size=(n, n)) + 1j*rng.normal(size=(n, n)))
    return intra, inter


def test_batch_kernel_survives_concurrent_calls_from_many_threads():
    intra, inter = _lead_matrices()
    energies = np.linspace(-1.5, 1.5, 64)
    delta = 1e-2

    def call(_):
        return green_renormalization_jit_batch(intra, inter, energies=energies,
                                               delta=delta)

    with ThreadPoolExecutor(max_workers=8) as ex:
        outs = list(ex.map(call, range(24)))

    assert len(outs) == 24
    for o in outs:
        arr = np.asarray(o[0] if isinstance(o, tuple) else o)
        assert np.all(np.isfinite(arr))


def test_concurrent_results_match_the_serial_ones():
    """The lock must serialize, not perturb: concurrent calls have to return
    exactly what a plain serial call returns."""
    intra, inter = _lead_matrices()
    energies = np.linspace(-1.0, 1.0, 32)
    delta = 1e-2

    ref = green_renormalization_jit_batch(intra, inter, energies=energies,
                                          delta=delta)
    ref0 = np.asarray(ref[0] if isinstance(ref, tuple) else ref)

    def call(_):
        out = green_renormalization_jit_batch(intra, inter, energies=energies,
                                              delta=delta)
        return np.asarray(out[0] if isinstance(out, tuple) else out)

    with ThreadPoolExecutor(max_workers=6) as ex:
        outs = list(ex.map(call, range(12)))

    for o in outs:
        assert np.allclose(o, ref0, rtol=0, atol=0), "concurrent result differs"
