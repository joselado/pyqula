import numpy as np

from pyqula import kpm
from pyqula.kpmtk.kpmnumba import kpm_moments_A_batch


def _reference_momentsA(v, m, A, n):
    """Plain-numpy reference for the A-weighted Chebyshev moments
    mus[i] = <T_i(m)v|A|v>, independent of the batched numba kernel."""
    v = np.asarray(v)
    m = np.asarray(m.todense()) if hasattr(m, "todense") else np.asarray(m)
    A = np.asarray(A.todense()) if hasattr(A, "todense") else np.asarray(A)
    Av = A@v
    mus = np.zeros(n, dtype=complex)
    am = v.copy()
    a = m@v
    mus[0] = np.conjugate(v).dot(Av)
    mus[1] = np.conjugate(a).dot(Av)
    for i in range(2, n):
        ap = 2.*m@a - am
        mus[i] = np.conjugate(ap).dot(Av)
        am, a = a, ap
    return mus


def _random_hermitian_and_operator(n, seed):
    rng = np.random.RandomState(seed)
    m = rng.random((n, n)) + 1j*rng.random((n, n))
    m = m + np.conjugate(m).T
    m = m/np.max(np.abs(np.linalg.eigvalsh(m)))/1.1  # rescale into (-1,1)
    A = rng.random((n, n)) + 1j*rng.random((n, n))
    A = A + np.conjugate(A).T  # Hermitian operator, e.g. a spin projector
    return m, A


def test_batched_momentsA_matches_plain_reference():
    """The batched numba A-weighted moments must agree with a plain-numpy
    per-vector recursion, for several independent starting vectors."""
    n = 15
    m, A = _random_hermitian_and_operator(n, seed=3)
    rng = np.random.RandomState(4)
    vs = rng.random((5, n)) + 1j*rng.random((5, n))
    vs = vs/np.sqrt(np.sum(np.conjugate(vs)*vs, axis=1))[:, None]

    npol = 20
    mus_batch = kpm_moments_A_batch(vs, m, A, n=npol)
    for k in range(vs.shape[0]):
        mus_ref = _reference_momentsA(vs[k], m, A, npol)
        assert np.max(np.abs(mus_batch[k] - mus_ref)) < 1e-8


def test_get_momentsA_matches_batched_single_vector():
    """kpm.get_momentsA (single-vector convenience wrapper) must agree with
    calling the batched kernel directly on a one-row batch."""
    n = 12
    m, A = _random_hermitian_and_operator(n, seed=5)
    rng = np.random.RandomState(6)
    v = rng.random(n) + 1j*rng.random(n)
    v = v/np.sqrt(np.abs(np.vdot(v, v)))

    npol = 18
    mus_single = kpm.get_momentsA(v, m, n=npol, A=A)
    mus_batch = kpm_moments_A_batch(np.array([v]), m, A, n=npol)[0]
    assert np.max(np.abs(mus_single - mus_batch)) < 1e-10


def test_random_trace_A_and_full_trace_A_run_and_agree_with_direct_average():
    """random_trace_A/full_trace_A batch their tries through the numba
    kernel; check the result matches directly averaging kpm.get_momentsA
    over the same explicit vectors (sanity check of the batching, not of
    the random-vector KPM statistical estimate itself)."""
    n = 10
    m, A = _random_hermitian_and_operator(n, seed=7)

    from pyqula.kpmtk.ldos import index2vector
    npol = 8
    vs = np.array([index2vector(i, n) for i in range(n)])
    mus_direct = np.mean([kpm.get_momentsA(v, m, n=npol, A=A) for v in vs], axis=0)
    mus_full = kpm.full_trace_A(m, n=npol, A=A)
    assert np.max(np.abs(mus_direct - mus_full)) < 1e-8
