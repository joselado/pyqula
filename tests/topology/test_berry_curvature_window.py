import numpy as np
import pytest

from pyqula import geometry
from pyqula.multihopping import MultiHopping
from pyqula.topologytk.berry import whole_zone_window


def _haldane():
    """Spinful Haldane model, Chern number 2"""
    h = geometry.honeycomb_lattice().get_hamiltonian()
    h.add_haldane(0.2)
    return h


def _qwz(m=-1.0):
    """Qi-Wu-Zhang model on the square lattice, Chern number 1 for -2<m<0,
    H(k) = sin kx sx + sin ky sy + (m + cos kx + cos ky) sz"""
    sx = np.array([[0., 1.], [1., 0.]])
    sy = np.array([[0., -1j], [1j, 0.]])
    sz = np.array([[1., 0.], [0., -1.]])
    h = geometry.square_lattice().get_hamiltonian(has_spin=True)
    tx = sz/2. + sx/2j
    ty = sz/2. + sy/2j
    d = {(0, 0, 0): m*sz + 0j, (1, 0, 0): tx, (-1, 0, 0): tx.conj().T,
         (0, 1, 0): ty, (0, -1, 0): ty.conj().T}
    h.set_multihopping(MultiHopping(d))
    return h


@pytest.mark.parametrize("name,nzones", [("honeycomb_lattice", 2),
    ("triangular_lattice", 2), ("kagome_lattice", 2), ("square_lattice", 1),
    ("lieb_lattice", 1)])
def test_window_holds_whole_zones(name, nzones):
    """The default window of the map is an axis-aligned rectangle made of
    whole Brillouin zones: its area in reduced momenta is an integer"""
    g = getattr(geometry, name)()
    vx, vy, n = whole_zone_window(g)
    assert abs(vx[1]) < 1e-12 and abs(vy[0]) < 1e-12 # a rectangle
    R = np.array(g.get_k2K())[:2, :2]
    area = abs(np.linalg.det(R @ np.array([vx, vy]).T))
    assert n == nzones
    assert abs(area - n) < 1e-10


def test_window_oblique_falls_back_to_one_zone():
    """A rotated lattice has no reciprocal vector along the axes, and the
    window is then the parallelogram of the reciprocal lattice vectors"""
    g = geometry.honeycomb_lattice().rotate(15.)
    vx, vy, n = whole_zone_window(g)
    R = np.array(g.get_k2K())[:2, :2]
    assert n == 1
    assert abs(abs(np.linalg.det(R @ np.array([vx, vy]).T)) - 1.) < 1e-10
    h = g.get_hamiltonian()
    h.add_haldane(0.2)
    (kx, ky, b) = h.get_berry_curvature(nk=30, write=False)
    assert abs(np.mean(b)/(2*np.pi) - h.get_chern(nk=30)) < 1e-3


@pytest.mark.parametrize("model", [_haldane, _qwz])
def test_mean_of_default_map_is_chern(model):
    """The mean of the default map over its window, divided by 2 pi, is the
    Chern number, on the honeycomb lattice (two zones in the window) and on
    the square lattice (one zone)"""
    h = model()
    C = h.get_chern(nk=40)
    assert abs(C - round(C)) < 1e-6 and round(C) != 0
    (kx, ky, b) = h.get_berry_curvature(nk=40, write=False)
    assert len(b) == 40*40
    assert abs(np.mean(b)/(2*np.pi) - C) < 1e-4


def test_path_curvature_equals_map():
    """The curvature along a path through points of the map is the map's
    value at those points, for the Haldane model"""
    h = _haldane()
    nk = 20
    dk = 1./(2*nk)
    (kx, ky, b) = h.get_berry_curvature(nk=nk, dk=dk, write=False)
    ii = [int(np.argmax(b)), int(np.argmin(np.abs(b))), 7, 123]
    kpath = [[kx[i], ky[i], 0.] for i in ii] # Cartesian momenta of the map
    (ik, bp) = h.get_berry_curvature_path(kpath=kpath, dk=dk,
                                          reciprocal=True, write=False)
    assert len(ik) == len(ii)
    assert np.max(np.abs(bp - b[ii])) < 1e-8*np.max(np.abs(b))


@pytest.mark.parametrize("e", [-1.0, -0.5])
def test_berry_density_map_mean_is_chern_density(e):
    """The energy-resolved Berry curvature map dOmega/dE covers the same
    window of whole zones as the Berry curvature map, so its mean over
    2 pi is the Brillouin-zone integral that chern_density returns at the
    same energy, from an independent mesh of one reduced zone"""
    from pyqula import topology
    h = geometry.honeycomb_lattice().get_hamiltonian(has_spin=False)
    h.add_haldane(0.2)
    kw = dict(delta=0.3, dk=0.01)
    (es, cs, csi) = topology.chern_density(h, nk=40, es=np.array([e]), **kw)
    for reciprocal in (True, False):
        (kx, ky, b) = topology.dOmega_dE_kmap(h, nk=40, e=e,
                        reciprocal=reciprocal, write=False, **kw)
        assert len(b) == 40*40
        assert abs(np.mean(b)/(2*np.pi) - cs[0]) < 1e-3*abs(cs[0])
