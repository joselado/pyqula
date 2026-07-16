import numpy as np

from pyqula import geometry
from pyqula.sctk import reorder
from testutils import temporary_attr


def _build_pairing(h, dense, d):
    with temporary_attr(reorder, "dense", dense):
        h1 = h.copy()
        h1.add_pairing(mode="pwave", d=d)
        h1.add_swave(0.2)
    return h1


def test_nambu_reordering_dense_matches_sparse():
    """Building a pairing Hamiltonian in the Nambu basis must give the same
    result regardless of whether the dense or sparse reordering path is used."""
    g = geometry.triangular_lattice()
    g = g.get_supercell((2, 2))
    h = g.get_hamiltonian()
    d = np.random.random(3)

    h1 = _build_pairing(h, False, d)
    h2 = _build_pairing(h, True, d)

    assert (h1 - h2).is_zero(), "Nambu reordering differs between dense and sparse paths"
