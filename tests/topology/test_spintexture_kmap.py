"""spintexture.kfun_map must evaluate its map on the grid it advertises.

The function builds two grids, kxs and kys, each centred on the
corresponding component of k0, and writes both back to the caller (and,
through conduction_texture, into TRACE_TEXTURE.OUT / DET_TEXTURE.OUT). Its
inner loop iterated kxs for both axes, so the y axis of the map was the kx
grid; the default k0=[0.,0.] makes the two grids identical, which is why
the single example in examples/2d never showed it.

The invariant asserted here needs no reference values: the map is a
translation of one grid, so shifting k0 by (dx,dy) must shift the returned
kx by dx and the returned ky by dy, independently. Plus the obvious
consistency requirement that the value reported at (kx,ky) is the operator
evaluated at that k-point.
"""
import numpy as np

from pyqula import algebra
from pyqula import geometry
from pyqula import spintexture


def _hamiltonian():
    g = geometry.honeycomb_lattice()
    h = g.get_hamiltonian(has_spin=True)
    h.add_rashba(0.2)
    return h


def _fun(hk):
    """Any k-dependent scalar: the off-diagonal Bloch element."""
    return algebra.todense(hk)[0, 1]


def test_k0_shifts_the_two_axes_independently():
    h = _hamiltonian()
    kx0, ky0, _ = spintexture.kfun_map(h, nk=4, operator=_fun, k0=[0., 0.])
    kx1, ky1, _ = spintexture.kfun_map(h, nk=4, operator=_fun, k0=[0.3, 0.5])
    assert np.allclose(np.array(kx1), np.array(kx0) + 0.3)
    assert np.allclose(np.array(ky1), np.array(ky0) + 0.5)


def test_the_values_belong_to_the_k_points_reported():
    h = _hamiltonian()
    hk_gen = h.get_hk_gen()
    R = np.array(h.geometry.get_k2K())
    kx, ky, out = spintexture.kfun_map(h, nk=3, operator=_fun, k0=[0.1, 0.4])
    for (x, y, o) in zip(kx, ky, out):
        ref = _fun(hk_gen(R@np.array([x, y, 0.])))
        assert np.isclose(o, ref), (x, y, o, ref)
