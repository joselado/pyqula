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


def test_kpm_total_energy_sums_the_levels_below_the_fermi_level():
    """With use_kpm=True spectrum.total_energy sums the states below
    fermi, as the diagonalization path does. A one-orbital chain on two
    k-points shifted by 3.5 has the levels 1.5 and 5.5. Every H(k) is a
    single level, so the random trace is exact, and both levels are more
    than 1.5 from zero and from the Fermi level 3.5, several widths of the
    kernel at npol=100 and scale=10. Below 3.5 there is one level, so the
    energy per k-point is 0.75, while below zero there is none.

    The KPM path used to ignore fermi and always gave the energy at zero"""
    from pyqula import geometry
    h = geometry.chain().get_hamiltonian(has_spin=False)
    h.add_onsite(3.5)
    e_ed = h.get_total_energy(nk=2, fermi=3.5)
    assert abs(e_ed - 0.75) < 1e-10
    e_kpm = h.get_total_energy(nk=2, fermi=3.5, use_kpm=True)
    assert abs(e_kpm - e_ed) < 0.02
