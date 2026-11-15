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


# A pairing given as a callable mode has no range of its own. It was
# evaluated on the box of geometry.ncells=2 cells, so on the one-site chain
# this third-neighbor singlet added no pairing, without a warning. It now
# reaches rcut, 5.0 by default in a periodic system, as a hopping function
# of get_hamiltonian does, and every pair beyond rcut is zero.

def third_neighbor_singlet(r1, r2):
    dr = r1 - r2
    return np.identity(2)*(abs(dr@dr - 9.) < 1e-4)


def odd_third_neighbor_singlet(r1, r2):
    """a singlet odd under r1 <-> r2, which is not a pairing"""
    dr = r1 - r2
    return np.sign(dr[0])*np.identity(2)*(abs(dr@dr - 9.) < 1e-4)


def sixth_neighbor_singlet(r1, r2):
    dr = r1 - r2
    return np.identity(2)*(abs(dr@dr - 36.) < 1e-4)


def bdg_spectrum(h, k):
    return eigh(dense(h.get_hk_gen()(k)), eigvals_only=True)


def normal_state_spectrum(k):
    h0 = geometry.chain().get_hamiltonian()
    h0.setup_nambu_spinor() # the normal state in the Nambu basis
    return bdg_spectrum(h0, k)


def test_a_third_neighbor_callable_is_the_registered_mode():
    h1 = geometry.chain().get_hamiltonian()
    h1.add_pairing(delta=0.2, mode=third_neighbor_singlet)
    h2 = geometry.chain().get_hamiltonian()
    h2.add_pairing(delta=0.2, mode="swave", nn=3)
    for k in [0.13, 0.41]:
        m1 = dense(h1.get_hk_gen()([k, 0., 0.]))
        m2 = dense(h2.get_hk_gen()([k, 0., 0.]))
        assert np.max(np.abs(m1 - m2)) < 1e-12
    k = [0.13, 0., 0.]
    assert np.max(np.abs(bdg_spectrum(h1, k) - normal_state_spectrum(k))) > 1e-3


def test_a_third_neighbor_callable_on_the_supercell():
    """The supercell of three sites at kS has the spectrum of the unit cell
    at the three momenta that fold onto it"""
    M = [[3, 0, 0], [0, 1, 0], [0, 0, 1]]
    h0 = geometry.chain().get_hamiltonian()
    h1 = h0.copy()
    h1.add_pairing(delta=0.2, mode=third_neighbor_singlet)
    hs = h0.get_supercell(M)
    hs.add_pairing(delta=0.2, mode=third_neighbor_singlet)
    kS = 0.37
    es = bdg_spectrum(hs, [kS, 0., 0.])
    e0 = np.sort(np.concatenate([bdg_spectrum(h1, [(kS + j)/3., 0., 0.])
                                 for j in range(3)]))
    assert len(es) == len(e0)
    assert np.max(np.abs(es - e0)) < 1e-10


def test_an_odd_third_neighbor_callable_is_refused():
    """The Fermi antisymmetry is checked on the pairs the pairing reaches,
    and it used to be checked on the box of two cells only"""
    h = geometry.chain().get_hamiltonian()
    with pytest.raises(ValueError, match="Fermi antisymmetry"):
        h.add_pairing(delta=0.3, mode=odd_third_neighbor_singlet)
    assert not h.has_eh # refused before the Hamiltonian was touched


def test_rcut_is_a_hard_cut_for_a_callable():
    k = [0.2, 0., 0.] # where both pairings have a sizable gap function
    e0 = normal_state_spectrum(k)
    def spectrum(mode, **kw):
        h = geometry.chain().get_hamiltonian()
        h.add_pairing(delta=0.2, mode=mode, **kw)
        return bdg_spectrum(h, k)
    assert np.max(np.abs(spectrum(third_neighbor_singlet, rcut=3.) - e0)) > 1e-3
    assert np.max(np.abs(spectrum(third_neighbor_singlet, rcut=2.5) - e0)) < 1e-12
    # beyond the default 5.0 only with an rcut that reaches it
    assert np.max(np.abs(spectrum(sixth_neighbor_singlet) - e0)) < 1e-12
    assert np.max(np.abs(spectrum(sixth_neighbor_singlet, rcut=6.) - e0)) > 1e-3


def test_rcut_with_a_registered_mode_is_refused():
    h = geometry.chain().get_hamiltonian()
    with pytest.raises(TypeError, match="rcut"):
        h.add_pairing(delta=0.2, mode="swave", rcut=3.)
