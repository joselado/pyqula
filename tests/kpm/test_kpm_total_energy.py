import numpy as np
import pytest
from scipy.sparse import csc_matrix

from pyqula import kpm


@pytest.mark.parametrize("e0,exact", [(-0.5, -0.5*4), (0.7, 0.)])
def test_kpm_total_energy_is_the_sum_of_the_occupied_levels(e0, exact):
    """kpm.total_energy is the sum of the eigenvalues below zero, the
    chemical potential, as the diagonalization path of
    spectrum.total_energy. On four states at e0 every normalized vector
    gives the same Chebyshev moments, so the random trace is exact: four
    states at -0.5 give -2 and four states at 0.7 give 0.

    It used to integrate E rho(E) over E > 0, the empty states, with rho
    the density per state, so it gave 0 for the first matrix and 0.7 for
    the second"""
    m = csc_matrix(e0*np.identity(4, dtype=np.complex128))
    e = kpm.total_energy(m, scale=2., npol=200, ne=2000, ntries=2)
    assert abs(e-exact) < 1e-3
