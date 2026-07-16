import numpy as np

from pyqula import geometry
from testutils import SCF_MAXERROR


def _total_energy_for_random_direction(h0):
    v = np.random.random(3) - .5  # random exchange direction
    v = 2 * v / np.sqrt(v.dot(v))  # normalize
    vs = [v, -v]
    mf = h0.copy()
    mf.add_exchange(vs)  # initial guess
    h1 = h0.copy()
    h1.add_exchange(.4 * v)  # add some bias
    h, etot = h1.get_mean_field_hamiltonian(nk=20, mf=mf, U=2.,
                                             maxerror=SCF_MAXERROR,
                                             return_total_energy=True)
    return etot


def test_scf_total_energy_is_rotationally_invariant():
    """The self-consistent total energy must not depend on the direction
    of the (arbitrary) initial exchange field used to seed the SCF loop."""
    g = geometry.bichain()
    h0 = g.get_hamiltonian()
    etots = np.array([_total_energy_for_random_direction(h0) for _ in range(6)])
    diff = etots - np.mean(etots)
    assert np.max(np.abs(diff)) < SCF_MAXERROR * 10, \
        f"SCF total energy is not rotationally invariant: {diff}"
