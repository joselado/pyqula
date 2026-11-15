"""The crystal field of a site counts every neighbor within rcut, so it is a
property of the lattice and cannot depend on the unit cell we use to write
it: on a supercell every site must get the value of the unit cell.

The range used to be a box of int(2*rcut/|a1|)+2 cells along every
direction, set only by a1. A square-lattice cell of 12 x 1 has |a1|=12, so
the box was 3 cells along a2, shorter than rcut, and the sites along a2
between 4 and 6.3 were lost. The test takes rcut=6.3 so that no pair sits on the cut.
"""

import numpy as np
import pytest

from pyqula import geometry
from pyqula.crystalfield import cf_potential


@pytest.mark.parametrize("mode", ["full", "stacking"])
def test_crystal_field_same_on_a_long_supercell(mode):
    g = geometry.square_lattice()
    v0 = cf_potential(g, rcut=6.3, vc=1.0, mode=mode)
    gs = g.get_supercell((12, 1))
    vs = cf_potential(gs, rcut=6.3, vc=1.0, mode=mode)
    assert len(vs) == 12
    assert np.max(np.abs(vs - v0[0])) < 1e-10


def test_crystal_field_same_on_a_long_supercell_along_a2():
    g = geometry.square_lattice()
    v0 = cf_potential(g, rcut=6.3, vc=1.0)
    vs = cf_potential(g.get_supercell((1, 12)), rcut=6.3, vc=1.0)
    assert np.max(np.abs(vs - v0[0])) < 1e-10
