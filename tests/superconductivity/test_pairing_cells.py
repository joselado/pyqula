"""A pairing of known range reaches every cell within that range.

add_pairing used to visit the box of geometry.ncells=2 cells, so on the
one-site chain a pairing of the third neighbors (distance 3, the cell 3)
was dropped without a warning. The same pairing on a supercell of three
sites sits in the cell 1 and was kept, so the two descriptions of the same
crystal gave different spectra. A supercell at the momentum kS must have
the spectrum of the unit cell at the three momenta (kS+j)/3 that fold onto
it, whichever cell the pairing was built on.
"""

import numpy as np
import pytest
from scipy.linalg import eigh

from pyqula import geometry


def dense(m):
    return np.array(m.todense()) if hasattr(m, "todense") else np.array(m)


@pytest.mark.parametrize("mode", ["swave", "dx2y2"])
def test_third_neighbor_pairing_of_the_chain(mode):
    M = [[3, 0, 0], [0, 1, 0], [0, 0, 1]]
    h0 = geometry.chain().get_hamiltonian()
    h1 = h0.copy()
    h1.add_pairing(delta=0.2, mode=mode, nn=3) # on the unit cell
    hs = h0.get_supercell(M)
    hs.add_pairing(delta=0.2, mode=mode, nn=3) # on the supercell
    kS = 0.37
    es = eigh(dense(hs.get_hk_gen()([kS, 0., 0.])), eigvals_only=True)
    e0 = np.sort(np.concatenate([eigh(dense(h1.get_hk_gen()(
        [(kS + j)/3., 0., 0.])), eigvals_only=True) for j in range(3)]))
    assert len(es) == len(e0)
    assert np.max(np.abs(es - e0)) < 1e-10


def test_third_neighbor_pairing_is_there():
    """The pairing of the third neighbors changes the spectrum of the
    chain at a generic momentum: before, it was dropped and the BdG
    spectrum was the one of the normal state with no pairing at all"""
    h0 = geometry.chain().get_hamiltonian()
    h0.setup_nambu_spinor() # the normal state in the Nambu basis
    h1 = geometry.chain().get_hamiltonian()
    h1.add_pairing(delta=0.2, mode="swave", nn=3)
    k = [0.13, 0., 0.]
    e0 = eigh(dense(h0.get_hk_gen()(k)), eigvals_only=True)
    e1 = eigh(dense(h1.get_hk_gen()(k)), eigvals_only=True)
    assert np.max(np.abs(e1 - e0)) > 1e-3
