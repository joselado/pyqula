import numpy as np

from pyqula import kpm, correlator


def _cluster(n=12):
    m = np.zeros((n, n))
    for a in range(n-1): m[a, a+1] = m[a+1, a] = 1.
    return m


def _error(lam, delta=0.02, scale=3.):
    """Largest difference between the Lorentz-kernel Green's function and
    the matrix inversion at the same delta, relative to its largest value,
    on a cluster whose levels the broadening resolves"""
    m = _cluster()
    es = np.linspace(-0.5, 0.5, 501)
    npol = int(round(lam*scale/(2*delta)))  # half width lam*scale/(2 npol)
    _, r, i = kpm.correlator0d(m, i=3, j=3, scale=scale, npol=npol, x=es,
            kernel="lorentz", lorentz_lambda=lam, write=False)
    _, r0, i0 = correlator.correlator0d(m, energies=es, i=3, j=3,
            delta=delta, write=False)
    g, g0 = r - 1j*i, r0 - 1j*i0
    return np.max(np.abs(g - g0))/np.max(np.abs(g0))


def test_a_larger_lambda_removes_the_truncation_error():
    """The Lorentz kernel had lambda=3 fixed, which leaves a truncation of
    order exp(-3) and caps the KPM Green's function at 5 to 10% of the
    matrix inversion on resolved poles. At the same delta a larger lambda
    takes more moments and brings it below the 0.3% that the energy
    dependence of the Lorentz half width, delta*sqrt(1-(E/scale)^2),
    leaves on its own."""
    assert _error(3.) > 0.05  # the default, unchanged
    assert _error(12.) < 5e-3


def test_the_default_lambda_is_the_old_kernel():
    from pyqula.kpmtk.kernels import lorentz_kernel
    mus = np.random.RandomState(0).random_sample(50)
    n = len(mus)
    old = mus*np.sinh(3.*(1. - np.arange(n)/n))/np.sinh(3.)
    assert np.allclose(lorentz_kernel(mus), old, atol=1e-15)
