import numpy as np
import pytest

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


def test_exact_at_removes_the_broadening():
    """The cumulative integral inside the gap depends strongly on delta,
    since the broadening moves curvature of the states above the gap into
    it; closing the path on the real axis with exact_at removes that
    dependence. For this spinless model at nk=12 the broadened plateau is
    0.51 and 0.81 at delta=0.2 and 0.1, and the exact one 1.24 and 1.28
    (for the spinful model it is 2.01 at nk=24 and 1.99 at nk=36, the
    Chern number, at delta=0.1 and 0.05)"""
    h = _haldane()
    out = {}
    for delta in [0.2, 0.1]:
        es = np.arange(-3.5, 0.+1e-9, delta/2) # from below the band to midgap
        out[delta] = topology.chern_density(h, nk=12, delta=delta, es=es,
                                            exact_at=[0.])
    (_, _, csi1, cex1), (_, _, csi2, cex2) = out[0.2], out[0.1]
    assert abs(csi1[-1] - csi2[-1]) > 0.25 # the broadened plateau moves
    assert abs(cex1[0] - cex2[0]) < 0.2*abs(csi1[-1] - csi2[-1])


def test_exact_at_outside_es_raises():
    h = _haldane()
    with pytest.raises(ValueError, match="exact_at"):
        topology.chern_density(h, nk=2, es=np.linspace(-3.5, 0., 5),
                               exact_at=[1.])
