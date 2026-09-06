import numpy as np

from pyqula import algebra, geometry
from pyqula.multihopping import MultiHopping

# add_hamiltonian looped over the directions the *target* already had, so
# any lattice direction present only in the Hamiltonian being added was
# dropped, silently: adding a second-neighbour hopping to a first-neighbour
# chain left it bit-identical.


def second_neighbor(r1, r2):
    """Hopping matrix connecting sites two lattice constants apart"""
    m = np.zeros((len(r1), len(r2)), dtype=np.complex128)
    for i in range(len(r1)):
        for j in range(len(r2)):
            if 1.9 < np.linalg.norm(r1[i] - r2[j]) < 2.1:
                m[i, j] = 0.5
    return m


def test_new_lattice_directions_are_merged():
    g = geometry.chain()
    h0 = g.get_hamiltonian()
    h = h0.copy()
    h.add_hopping_matrix(second_neighbor, nc=3)
    dirs = sorted(tuple(int(x) for x in t.dir) for t in h.hopping)
    assert (2, 0, 0) in dirs and (-2, 0, 0) in dirs
    assert MultiHopping(h.get_dict()).norm() > MultiHopping(h0.get_dict()).norm()


def test_it_matches_the_hamiltonian_sum():
    """Hamiltonian addition (algebratk/hamiltonianalgebra) merges the two
    multihopping dictionaries correctly, so it is an in-repo reference for
    what add_hamiltonian must produce."""
    g = geometry.chain()
    h0 = g.get_hamiltonian()
    h = h0.copy()
    h.add_hopping_matrix(second_neighbor, nc=3)
    ref = h0 + g.get_hamiltonian(mgenerator=second_neighbor,
                                 is_multicell=True, nc=3)
    (k1, e1) = h.get_bands(nk=30, write=False)
    (k2, e2) = ref.get_bands(nk=30, write=False)
    assert np.max(np.abs(np.sort(e1) - np.sort(e2))) < 1e-10


def test_shared_directions_still_add_up():
    """The common case -- add_kekule and friends stay inside the
    directions the target already has -- must be untouched."""
    g = geometry.honeycomb_lattice()
    h = g.get_hamiltonian()
    h.turn_multicell()
    h2 = h.copy()
    h2.add_hamiltonian(h)
    d1, d2 = h.get_dict(), h2.get_dict()
    assert set(d1.keys()) == set(d2.keys())
    for key in d1:
        m1 = np.array(algebra.todense(d1[key]))
        m2 = np.array(algebra.todense(d2[key]))
        assert np.max(np.abs(2 * m1 - m2)) < 1e-12
