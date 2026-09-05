import numpy as np

from pyqula import geometry
from pyqula.greentk.kchain import green_kchain_NN


def test_surface_onsite_matrix_does_not_raise():
    """green_kchain_NN's `hs` branch calls np.identity, but
    greentk/kchain.py imported only .rg and ..algebra -- so asking for a
    modified surface onsite matrix raised NameError instead of returning a
    surface Green's function."""
    h = geometry.square_lattice().get_hamiltonian(has_spin=False)
    n = h.intra.shape[0]
    args = dict(k=0.2, energy=0.1, delta=0.02, only_bulk=False)
    (gb0, sf0) = green_kchain_NN(h, **args)
    (gb1, sf1) = green_kchain_NN(h, hs=np.identity(n) * 0.3, **args)
    # the bulk Green's function does not see the surface onsite matrix
    assert np.max(np.abs(np.array(gb0) - np.array(gb1))) < 1e-10
    # the surface one does
    assert np.max(np.abs(np.array(sf0) - np.array(sf1))) > 1e-3
