import numpy as np

from pyqula import potentials


def test_interpolate2d_runs_without_removed_scipy_function():
    """Regression check for potentials.interpolate2d: it used
    scipy.interpolate.interp2d, removed in SciPy >=1.14. It was replaced
    with RegularGridInterpolator; check it reproduces a simple scalar
    field at the sampled points."""
    n = 20
    x = np.linspace(-1., 1., n)
    y = np.linspace(-1., 1., n)
    xx, yy = np.meshgrid(x, y)
    r = np.array([xx.ravel(), yy.ravel()]).T
    v = r[:, 0] + 2 * r[:, 1]  # a simple linear field
    f = potentials.interpolate2d(r, v)
    out = f(np.array([0.3, -0.4, 0.]))
    assert np.isclose(out[0], 0.3 + 2 * (-0.4), atol=0.2)
