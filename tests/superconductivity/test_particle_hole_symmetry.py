import numpy as np

from testutils import random_hermitian_hamiltonian
from pyqula import geometry


def _assert_ph_symmetric(h, tol=1e-8):
    (k, e) = h.get_bands()
    e = np.sort(np.array(e))
    assert np.max(np.abs(e + e[::-1])) < tol, \
        f"BdG spectrum is not particle-hole symmetric: {e}"


def test_bdg_spectrum_is_particle_hole_symmetric_0d():
    """A Bogoliubov-de Gennes spectrum is always symmetric under E -> -E,
    regardless of the normal-state Hamiltonian or the pairing amplitude."""
    for _ in range(4):
        h = random_hermitian_hamiltonian(geometry.chain, supercell=3)
        h.add_swave(0.3)
        _assert_ph_symmetric(h)


def test_bdg_spectrum_is_particle_hole_symmetric_periodic():
    """Same invariant for a periodic (k-dependent) BdG Hamiltonian."""
    g = geometry.honeycomb_lattice()
    h = g.get_hamiltonian()
    h.add_rashba(0.2)
    h.add_zeeman([0., 0., 0.3])
    h.add_swave(0.15)
    _assert_ph_symmetric(h)
