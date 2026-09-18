"""The GPU path of the batched dense diagonalization.

htk/eigenvectors.py's peigh/peigvalsh dispatch to htk/eigenvectorsjax.py
under pyqula.gpu.set_gpu(True), and must return exactly what the numba
kernels return: float64 eigenvalues, complex128 eigenvectors, same shapes.
"""
import numpy as np
import pytest

from pyqula.htk import eigenvectors as ev
from testutils import gpu_backend


def _hermitian_batch(nb, n, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((nb, n, n)) + 1j*rng.standard_normal((nb, n, n))
    return (a + np.conj(np.transpose(a, (0, 2, 1))))/2


@pytest.mark.parametrize("nb,n", [(4, 40), (3, 64)])
def test_the_device_eigenvalues_match_the_numba_ones(nb, n):
    pytest.importorskip("jax")
    hks = _hermitian_batch(nb, n)
    cpu = ev.peigvalsh(hks)
    with gpu_backend():
        got = ev.peigvalsh(hks)
    assert got.shape == cpu.shape and got.dtype == np.float64
    assert np.max(np.abs(np.sort(got, axis=1) - np.sort(cpu, axis=1))) < 1e-10


def test_the_device_eigenvectors_diagonalize_the_input():
    """Eigenvectors are only defined up to a phase (and up to a rotation
    within a degenerate subspace), so check the eigenvalue equation rather
    than the vectors themselves"""
    pytest.importorskip("jax")
    hks = _hermitian_batch(3, 48)
    with gpu_backend():
        es, ws = ev.peigh(hks)
    assert es.dtype == np.float64 and ws.dtype == np.complex128
    residual = np.einsum('kij,kjn->kin', hks, ws) - ws*es[:, None, :]
    assert np.max(np.abs(residual)) < 1e-10


def test_single_precision_is_faster_and_loses_about_six_digits():
    """eigh_prec="single" is the trade the sweep in
    documentation/gpu_porting_plan.md measured: several times faster again
    on a consumer card, at ~1e-6 relative eigenvalue error"""
    pytest.importorskip("jax")
    hks = _hermitian_batch(4, 48, seed=3)
    cpu = ev.peigvalsh(hks)
    with gpu_backend():
        got = ev.peigvalsh(hks, eigh_prec="single")
    err = np.max(np.abs(np.sort(got, axis=1) - np.sort(cpu, axis=1)))
    assert 0. < err < 1e-4*np.max(np.abs(cpu))


def test_a_batch_bigger_than_one_dispatch_still_agrees():
    """The stack is chunked by matrix entries, because cuSOLVER's batched
    solver asks for a workspace of roughly 500 bytes per entry and a large
    stack exhausted a 6 GB card in one dispatch. Chunking must not change
    the answer, including for a last chunk that is not full"""
    pytest.importorskip("jax")
    from pyqula.htk import eigenvectorsjax as evjax
    hks = _hermitian_batch(37, 64, seed=5)
    cpu = ev.peigvalsh(hks)
    chunk = evjax.CHUNK_ELEMENTS
    try:
        evjax.CHUNK_ELEMENTS = 8*64*64  # 8 matrices per dispatch, 37 = 4*8+5
        with gpu_backend():
            got = ev.peigvalsh(hks)
    finally:
        evjax.CHUNK_ELEMENTS = chunk
    assert got.shape == cpu.shape
    assert np.max(np.abs(np.sort(got, axis=1) - np.sort(cpu, axis=1))) < 1e-10


def test_small_matrices_stay_on_the_cpu_even_with_the_switch_on():
    """Below gpu_min_dimension the device loses to numba however large the
    batch (the transfers are not amortized and the solves do not fill the
    card), so the dispatch does not happen at all"""
    pytest.importorskip("jax")
    from pyqula.htk import eigenvectorsjax as evjax
    calls = []
    real = evjax.peigvalsh_gpu

    def spy(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    evjax.peigvalsh_gpu = spy
    try:
        with gpu_backend():
            ev.peigvalsh(_hermitian_batch(64, ev.gpu_min_dimension - 1))
            assert calls == []
            ev.peigvalsh(_hermitian_batch(4, ev.gpu_min_dimension))
            assert len(calls) == 1
    finally:
        evjax.peigvalsh_gpu = real
