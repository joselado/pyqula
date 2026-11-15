import numpy as np
import pytest
from scipy.sparse import csc_matrix

from pyqula import kpm


def _level(e0=0.3, n=4):
    """n states at the energy e0. Every normalized vector gives the same
    Chebyshev moments T_n(e0/scale), so the random trace is exact and two
    routes with different random vectors give the same moments"""
    return csc_matrix(e0*np.identity(n, dtype=np.complex128))


@pytest.mark.parametrize("scale", [2., 4.])
def test_kpm_dos_is_per_state_and_per_unit_energy(scale):
    """kpm.dos is the density of states per state and per unit energy, the
    same as kpm.tdos at the same energies, so it integrates to one whatever
    the scale. It used to return the density per unit x = E/scale, complex,
    which integrated to the scale instead of one"""
    m = _level()
    xs = np.linspace(-0.95*scale, 0.95*scale, 4001)
    ys = kpm.dos(m, xs, ntries=3, n=200, scale=scale)
    (_, ref) = kpm.tdos(m, x=xs, npol=200, ntries=3, scale=scale)
    assert np.max(np.abs(ys-ref)) < 1e-8*np.max(np.abs(ref))
    assert abs(np.trapezoid(ys, xs)-1.) < 1e-3
