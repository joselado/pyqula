import numpy as np

from pyqula import geometry
from pyqula import topology


def _haldane():
    g = geometry.honeycomb_lattice()
    h = g.get_hamiltonian(has_spin=False)
    h.add_haldane(0.05)
    return h


def test_default_dk_is_converged_for_the_broadening():
    """The finite-difference step of chern_density follows delta: with the
    default, a step four times smaller gives the same Berry density, while
    the old fixed step dk=0.02 is off. At delta=0.05 the old step lowered the
    plateau of the guide's Haldane model from 1.69 to 1.62."""
    h = _haldane()
    es = np.array([-0.6, -0.4, -0.3, -0.25]) # below the gap, where the bands are steep
    kw = dict(nk=6, delta=0.05, es=es)
    (_, c0, _) = topology.chern_density(h, **kw) # default dk
    (_, c1, _) = topology.chern_density(h, dk=0.05/100, **kw) # four times smaller
    (_, c2, _) = topology.chern_density(h, dk=0.02, **kw) # the old default
    scale = np.max(np.abs(c1))
    assert np.max(np.abs(c0 - c1)) < 1e-2*scale
    assert np.max(np.abs(c2 - c1)) > 5e-2*scale
