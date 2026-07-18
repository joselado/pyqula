import numpy as np
import pytest

from pyqula import kpm
from pyqula.kpmtk.kpmnumba import kpm_moments_v


def _random_hermitian_and_vector(n, complex_input, seed):
    rng = np.random.RandomState(seed)
    if complex_input:
        m = rng.random((n, n)) + 1j * rng.random((n, n))
        v = rng.random(n) + 1j * rng.random(n)
    else:
        m = rng.random((n, n))
        v = rng.random(n)
    m = m + np.conjugate(m).T
    m = m / np.max(np.abs(np.linalg.eigvalsh(m))) / 1.1  # rescale into (-1, 1)
    v = v / np.sqrt(np.abs(np.vdot(v, v)))
    return m, v


@pytest.mark.parametrize("complex_input", [False, True])
@pytest.mark.parametrize("kpm_prec,tol", [("double", 1e-8), ("single", 1e-4)])
def test_numba_cpu_moments_match_reference(complex_input, kpm_prec, tol):
    """The numba CPU backend must agree with the plain-Python reference
    implementation, for real and complex input, at both precisions."""
    m, v = _random_hermitian_and_vector(20, complex_input, seed=1)
    ref = kpm.python_kpm_moments(v.astype(complex), m.astype(complex), n=25)
    mus = kpm_moments_v(v, m, n=25, kpm_prec=kpm_prec, kpm_cpugpu="CPU")
    assert np.max(np.abs(mus - ref)) < tol


@pytest.mark.parametrize("complex_input", [False, True])
@pytest.mark.parametrize("kpm_prec,tol", [("double", 1e-8), ("single", 1e-4)])
def test_jax_backend_matches_numba_cpu(complex_input, kpm_prec, tol):
    """The JAX backend (GPU if available, otherwise transparently the CPU)
    must agree with the numba CPU backend, for real and complex input, at
    both single and double precision."""
    pytest.importorskip("jax")
    m, v = _random_hermitian_and_vector(20, complex_input, seed=2)
    mus_cpu = kpm_moments_v(v, m, n=25, kpm_prec=kpm_prec, kpm_cpugpu="CPU")
    mus_jax = kpm_moments_v(v, m, n=25, kpm_prec=kpm_prec, kpm_cpugpu="GPU")
    assert np.max(np.abs(mus_cpu - mus_jax)) < tol
