"""The qtci LDOS does not depend on the unit of energy.

With the Hamiltonian, the energies and the broadening all multiplied by
lam, the LDOS is divided by lam and nothing else changes. The triangle sum
on the full mesh has this invariance to rounding. The qtci path
reconstructs the band energies and the site weights from one interpolation
of all of them, whose tolerance is relative to the largest value it meets.
The band energies go as lam and the site weights do not, so on a shared
scale a large unit of energy loosens the tolerance of the weights and a
small one that of the energies: the two families must be normalized apart.
"""
import random

import numpy as np
import pytest

from pyqula import geometry
from pyqula.dostk import simplexdos


def _graphene(lam):
    """Honeycomb lattice with sublattice imbalance 0.3, every energy
    multiplied by lam"""
    h = geometry.honeycomb_lattice().get_hamiltonian(has_spin=False)
    h.add_sublattice_imbalance(0.3) # the two sites differ
    return lam*h


def _ldos(counted,lam):
    """lam times the LDOS in the unit lam, which does not depend on lam"""
    es = (np.linspace(-3.5,3.5,101)+0.0037)*lam
    return lam*counted(_graphene(lam),es,nk=128,delta=1e-3*lam)[1]


@pytest.mark.parametrize("lam", [1e4, 1e-4])
def test_qtci_ldos_does_not_depend_on_the_unit_of_energy(lam):
    t1 = _ldos(simplexdos.ldos_triangle_counted,1.)
    tl = _ldos(simplexdos.ldos_triangle_counted,lam)
    scale = np.max(np.abs(t1))
    # the triangle sum on the full mesh is exactly invariant
    assert np.max(np.abs(tl-t1)) < 1e-10*scale
    # the pivot search of the interpolation draws from the random module,
    # so each map is compared with its triangle and not with the other one
    random.seed(0)
    q1 = _ldos(simplexdos.ldos_qtci_counted,1.)
    ql = _ldos(simplexdos.ldos_qtci_counted,lam)
    # 1e-6 to 7e-6 was measured over five seeds at these three units
    assert np.max(np.abs(q1-t1)) < 2e-4*scale
    assert np.max(np.abs(ql-tl)) < 2e-4*scale


def test_weights_that_vanish_are_not_interpolated():
    """Without magnetism the sz weight of every band on every site
    vanishes up to rounding. These weights share one interpolation with
    the band energies. Divided by their own largest value they would be
    noise of order one, and the interpolation would follow it at full
    rank through every point of the mesh. The weights have no unit, so
    they enter as they are and stay rounding noise next to the energies"""
    h = geometry.honeycomb_lattice().get_hamiltonian(has_spin=True)
    h.add_sublattice_imbalance(0.2)
    es = np.linspace(-3.,3.,61)+0.0037
    nk = 256
    random.seed(0)
    _,d,ndiag,_ = simplexdos.ldos_qtci_counted(h,es,nk=nk,delta=2e-2,
            operator="sz")
    assert np.max(np.abs(d)) < 1e-10 # no spin density anywhere
    assert ndiag < nk*nk//2 # 22218 of 65536 points were measured
