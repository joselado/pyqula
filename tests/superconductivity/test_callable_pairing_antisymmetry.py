import numpy as np
import pytest

from pyqula import geometry

# add_pairing(mode=callable) took any 2x2-valued function of two positions.
# Fermi antisymmetry, D(r1,r2) = sigma_y D(r2,r1)^T sigma_y, holds for every
# registered mode by construction, but a callable that breaks it (an odd
# singlet, an onsite triplet) added a term that is not a pairing and still
# shows up in the BdG spectrum, and only h.check() noticed.


def _neighbors(r1, r2):
    dr = r1 - r2
    return 0.99 < dr.dot(dr) < 1.01


def odd_singlet(r1, r2):
    """the shape of the removed "haldane" mode: a singlet odd under swap"""
    dr = r1 - r2
    return np.sign(dr[0] + 1e-3*dr[1])*np.identity(2)*_neighbors(r1, r2)


def onsite_triplet(r1, r2):
    """the shape of the removed "swavez" mode: a d-vector on the same site"""
    return (np.linalg.norm(r1 - r2) < 1e-3)*np.array([[1., 0.], [0., -1.]])


def extended_s_by_hand(r1, r2):
    return np.identity(2)*_neighbors(r1, r2)


@pytest.mark.parametrize("weight", [odd_singlet, onsite_triplet])
def test_a_callable_breaking_antisymmetry_is_refused(weight):
    h = geometry.honeycomb_lattice().get_hamiltonian()
    with pytest.raises(ValueError, match="Fermi antisymmetry"):
        h.add_pairing(delta=0.3, mode=weight)
    assert not h.has_eh # refused before the Hamiltonian was touched


def test_a_valid_callable_equals_the_registered_mode():
    h1 = geometry.honeycomb_lattice().get_hamiltonian()
    h1.add_pairing(delta=0.3, mode=extended_s_by_hand)
    h2 = geometry.honeycomb_lattice().get_hamiltonian()
    h2.add_pairing(delta=0.3, mode="extended_swave")
    assert (h1 - h2).is_zero()


def test_a_callable_must_return_a_two_by_two_matrix():
    h = geometry.square_lattice().get_hamiltonian()
    with pytest.raises(ValueError, match="2x2"):
        h.add_pairing(delta=0.3, mode=lambda r1, r2: 1.0)
