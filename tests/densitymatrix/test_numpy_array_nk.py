import numpy as np
import pytest

from pyqula import geometry
from pyqula.kpointstk.kmesh import kmesh

# nk is one number or one per direction, and a numpy array of them has to
# mean the same as a list: kmesh used to test nk==1 directly, which on an
# array is an array with an ambiguous truth value, so every density-matrix
# route crashed on nk=np.array([6,6]) while nk=[6,6] worked.


@pytest.mark.parametrize("dim,nk", [
    (1, 6), (1, [6]), (1, np.array(6)), (1, np.array([6])),
    (2, 5), (2, [5, 5]), (2, np.array([5, 5])), (2, np.array(5)),
    (3, 3), (3, [3, 3, 3]), (3, np.array([3, 3, 3])),
])
def test_array_nk_gives_the_same_mesh_as_a_number(dim, nk):
    n = int(np.ravel(nk)[0])
    assert np.allclose(np.array(kmesh(dim, nk=nk)), np.array(kmesh(dim, nk=n)))


def test_anisotropic_array_nk_matches_the_list():
    assert np.allclose(kmesh(2, nk=np.array([3, 5])), kmesh(2, nk=[3, 5]))
    assert len(kmesh(2, nk=np.array([3, 5]))) == 15


def test_too_few_entries_are_refused():
    with pytest.raises(ValueError, match="one per direction"):
        kmesh(3, nk=[4, 4])


def test_density_matrix_takes_an_array_nk():
    h = geometry.honeycomb_lattice().get_hamiltonian()
    h.add_exchange([0.1, 0.2, 0.3])
    ds = [(0, 0, 0), (1, 0, 0)]
    dml = h.get_density_matrix(ds=ds, nk=[6, 6])
    dma = h.get_density_matrix(ds=ds, nk=np.array([6, 6]))
    for d in ds:
        assert np.max(np.abs(dml[d] - dma[d])) < 1e-12
