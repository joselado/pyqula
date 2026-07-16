import numpy as np

from pyqula import geometry
from testutils import SCF_MAXERROR


def _gap_for_random_direction(h0):
    v = np.random.random(3) - .5  # random Zeeman direction
    v = 4 * v / np.sqrt(v.dot(v))  # normalize
    h1 = h0.copy()
    h1.add_exchange(v)
    h1.turn_nambu()
    h, etot = h1.get_mean_field_hamiltonian(nk=20, mf="random", V1=-2.,
                                             filling=.3,
                                             maxerror=SCF_MAXERROR,
                                             return_total_energy=True)
    return h.get_gap()


def test_superconducting_gap_is_rotationally_invariant():
    """The self-consistent superconducting gap must not depend on the
    direction of the (arbitrary) Zeeman field used to seed the SCF loop."""
    g = geometry.bichain()
    h0 = g.get_hamiltonian()
    gaps = np.array([_gap_for_random_direction(h0) for _ in range(6)])
    diff = gaps - np.mean(gaps)
    assert np.max(np.abs(diff)) < SCF_MAXERROR * 10, \
        f"SCF gap is not rotationally invariant: {diff}"
