import numpy as np
import pytest

from pyqula import geometry
from pyqula.chitk import chiAB


def _chain():
    g = geometry.chain()
    h = g.get_hamiltonian(has_spin=True)
    h.add_exchange([0., 0., 0.2])
    return h


def test_get_chi_takes_a_list_of_energies():
    """h.get_chi(energies=[0.]) failed to compile inside numba, since the
    kernels read energies.shape and subtract the array from a float. A
    list, an array and a single number must give the same response."""
    h = _chain()
    es_list, chi_list = h.get_chi(energies=[0., 0.5], q=[0.3, 0., 0.], nk=20)
    es_arr, chi_arr = h.get_chi(energies=np.array([0., 0.5]), q=[0.3, 0., 0.],
                                nk=20)
    assert np.allclose(es_list, es_arr)
    assert np.max(np.abs(np.array(chi_list) - np.array(chi_arr))) < 1e-12
    es_one, chi_one = h.get_chi(energies=0.5, q=[0.3, 0., 0.], nk=20)
    assert np.max(np.abs(np.array(chi_one)[0] - np.array(chi_arr)[1])) < 1e-12


@pytest.mark.parametrize("kwargs", [dict(mode="matrix"),
                                    dict(mode="diagonal"),
                                    dict(mode="matrix", ij_mode="accelerated",
                                         A="sz", B="sz")])
def test_every_mode_takes_a_list_of_energies(kwargs):
    h = _chain()
    ref = chiAB.chiAB_q(h, energies=np.array([-0.5, 0.5]), q=[0.3, 0., 0.],
                        nk=12, **kwargs)[1]
    out = chiAB.chiAB_q(h, energies=[-0.5, 0.5], q=[0.3, 0., 0.], nk=12,
                        **kwargs)[1]
    assert np.max(np.abs(np.array(out) - np.array(ref))) < 1e-12
