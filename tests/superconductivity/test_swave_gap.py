import numpy as np

from pyqula import geometry


def test_swave_gap_is_proportional_to_delta():
    """With only a bare s-wave pairing added (no chemical potential or
    other competing terms), the quasiparticle gap must equal 2*delta."""
    g = geometry.chain()
    for delta in (0.1, 0.3, 0.5):
        h = g.get_hamiltonian()
        h.add_swave(delta)
        assert abs(h.get_gap() - 2 * delta) < 1e-8
