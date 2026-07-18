import numpy as np
import pytest

from pyqula import specialgeometry
from pyqula.specialhopping import twisted_matrix
from pyqula import topology


@pytest.mark.slow
def test_berry_map_tbg_density_matches_reference(tmp_path, monkeypatch):
    """Regression check for the spatially resolved Berry density of twisted
    bilayer graphene with an interlayer bias, at the smallest commensurate
    moire index (n=1): the total density must match the value recorded from
    a known-good run. Marked slow: the twisted-hopping matrix generator's
    setup cost stays a few seconds even at the smallest moire cell."""
    monkeypatch.chdir(tmp_path)  # spatial_berry_density writes BERRY_RMAP.OUT to cwd
    g = specialgeometry.twisted_bilayer(1)
    h = g.get_hamiltonian(is_sparse=True, has_spin=False, is_multicell=False,
                           mgenerator=twisted_matrix(ti=0.4, lambi=7.0))
    h.turn_dense()

    def ff(r):
        return r[2] * 0.05

    h.add_onsite(ff)
    h.set_filling(0.5, nk=1)
    b = topology.spatial_berry_density(h, k=[-0.333333, 0.33333, 0.0], nrep=2,
                                        operator="valley")
    assert np.isclose(np.sum(b), -37.85472013728288, atol=1e-6)
