"""Local Chern marker from the KPM projector (topologytk/kpmchern.py)

The marker of a site is built from the projector entries inside a ball of
radius hops around it, with positions relative to the site. Two errors
separate it from the dense Bianco-Resta marker of real_space_chern: the
ball, which the local marker of the exact projector already has, and the
Chebyshev expansion of the projector. Each one is held to its own
reference, the ball to the dense marker and the expansion to the local
marker of the exact projector on the same balls.

Measured on the Haldane flake below (984 sites, t2=0.1, gap 1.04, the 84
sites within a quarter of its radius as the bulk): the largest bulk error
of the ball is 0.30, 0.12, 0.049, 0.020, 0.0079 and 0.0012 at radius 2, 3,
4, 5, 6 and 8, a factor of about 2.5 per hop, and the KPM marker differs
from the exact projector on the same balls of radius 4 by 0.39, 0.020,
2.1e-3, 2.6e-4, 4.2e-5 and 1.4e-5 at npol 10, 25, 50, 100, 200 and 400
(kpm_radius 8), and by 6.8e-3, 7.0e-4 and 3.1e-5 at kpm_radius 2, 4 and
12 (npol 200). The tolerances sit about a factor of two above those.
"""
import numpy as np
import pytest
from scipy.sparse import csr_matrix

from pyqula import islands
from pyqula import topology
from pyqula import densitymatrix
from pyqula.kpmtk import truncation
from pyqula.topologytk import kpmchern
from pyqula.topologytk.realspace import site_areas


def _flake(t2=0.1, mass=0., n=12, spin=False):
    g = islands.get_geometry(name="honeycomb", n=n, nedges=4, rot=0.0,
            clean=False)
    h = g.get_hamiltonian(has_spin=spin)
    if t2 and not spin: h.add_haldane(t2)
    if t2 and spin: h.add_kane_mele(t2)
    if mass: h.add_sublattice_imbalance(mass)
    return h


def _core(r):
    """The sites within a quarter of the radius of the flake"""
    d2 = np.sum(np.array(r)**2, axis=1)
    return d2 < d2.max()/16.


def _exact_local(h, radius, operator=None):
    """The local marker of every site from the exact projector, restricted
    to the same balls as the KPM one"""
    m = h.get_hk_gen()([0., 0., 0.])
    P = csr_matrix(densitymatrix.occupied_projector(m).T)
    nsites = len(h.geometry.r)
    adj = truncation.site_graph([csr_matrix(h.intra)], nsites)
    op = None
    if operator is not None: op = csr_matrix(operator.get_matrix())
    c = kpmchern.local_markers(P, kpmchern._reach(adj, radius),
            np.array(h.geometry.r)[:, 0:2], m.shape[0]//nsites, op=op)
    return c/site_areas(h.geometry)


@pytest.fixture(scope="module")
def haldane():
    h = _flake()
    r, c0 = topology.real_space_chern(h, write=False)
    return h, r, c0


@pytest.fixture(scope="module")
def radius4(haldane):
    """The KPM marker at radius 4 and the exact projector on the same balls"""
    h, r, c0 = haldane
    return h.get_kpm_chern_marker(radius=4)[1], _exact_local(h, 4)


def test_whole_flake_is_the_dense_marker():
    """With the exact projector and balls that cover the flake, the local
    marker is the dense one to roundoff, which pins its sign and its
    prefactor; with operator=sz too, since sz commutes with the Kane-Mele
    Hamiltonian (with Rashba coupling it does not, and the dense marker
    then depends on the origin of the positions)"""
    h = _flake(n=4)
    _, c0 = topology.real_space_chern(h, write=False)
    assert np.allclose(_exact_local(h, 60), c0, atol=1e-10)
    h = _flake(n=4, spin=True)
    sz = h.get_operator("sz")
    _, c0 = topology.real_space_chern(h, operator=sz, write=False)
    assert np.abs(c0).max() > 1. # far from zero
    assert np.allclose(_exact_local(h, 60, operator=sz), c0, atol=1e-10)


def test_ball_converges_to_dense(haldane, radius4):
    """The exact projector on the balls tends to the dense marker in the
    bulk as the radius grows"""
    h, r, c0 = haldane
    core = _core(r)
    err = [np.abs(_exact_local(h, 2) - c0)[core].max(),
            np.abs(radius4[1] - c0)[core].max(),
            np.abs(_exact_local(h, 6) - c0)[core].max()]
    assert err[1] < err[0]/3. and err[2] < err[1]/3.
    assert err[2] < 0.016


def test_expansion_matches_exact_projector(haldane, radius4):
    """At a fixed radius the KPM marker tends to the one of the exact
    projector as the number of polynomials and the truncation grow"""
    h, r, c0 = haldane
    core = _core(r)
    ref = radius4[1]
    err = [np.abs(h.get_kpm_chern_marker(radius=4, npol=npol,
        kpm_radius=12)[1] - ref)[core].max() for npol in (50, 200)]
    assert err[0] < 5e-3 and err[1] < 1e-4
    assert np.abs(radius4[0] - ref)[core].max() < 1.5e-3 # kpm_radius 4


def test_agrees_with_dense_in_the_bulk(haldane):
    h, r, c0 = haldane
    core = _core(r)
    _, c = topology.real_space_chern(h, write=False, mode="kpm", radius=6)
    assert abs(np.mean(c0[core]) - 1.) < 1e-3
    assert np.abs(c - c0)[core].max() < 0.016


def test_sign_flips_with_t2(haldane, radius4):
    """t2 -> -t2 is the complex conjugate of H, which flips every marker"""
    h, r, c0 = haldane
    cp = radius4[0]
    _, cm = _flake(t2=-0.1).get_kpm_chern_marker(radius=4)
    assert np.mean(cp[_core(r)]) > 0.9
    assert np.allclose(cm, -cp, atol=1e-8)


def test_trivial_mass():
    """A sublattice mass above 3*sqrt(3)*t2 closes and reopens the gap,
    and the bulk marker goes to zero"""
    h = _flake(t2=0.1, mass=0.9)
    r, c0 = topology.real_space_chern(h, write=False)
    _, c = h.get_kpm_chern_marker(radius=4, npol=100)
    core = _core(r)
    assert np.abs(c0[core]).max() < 1e-3
    assert np.abs(c[core]).max() < 0.04


def test_sum_over_the_flake(haldane, radius4):
    """The dense marker sums to zero over an open flake, the edge giving
    back what the bulk carries. The local marker does not: the edge states
    are gapless, the projector decays slowly along the edge, and a ball
    misses what compensates the bulk. The KPM marker loses the sum rule as
    the exact projector on the same balls does"""
    h, r, c0 = haldane
    A = site_areas(h.geometry)
    assert abs(np.sum(c0*A)) < 1e-8
    c, ref = radius4
    n = len(c)
    assert np.sum(c*A)/np.sum(A) > 0.5 # far from zero
    assert abs(np.sum(c) - np.sum(ref))/n < 0.01


def test_kane_mele_with_sz():
    """Spinful flake, marker weighted with sz, as real_space_chern does.
    Measured on these 432 sites (dense bulk 1.993): the error against the
    dense marker is 0.24, 0.099 and 0.017 at radius 3, 4 and 6, and
    against the exact projector on the same balls 4.8e-3, 1.8e-3 and
    6.5e-4 at npol=200"""
    h = _flake(t2=0.1, n=8, spin=True)
    sz = h.get_operator("sz")
    r, c0 = topology.real_space_chern(h, operator=sz, write=False)
    _, c = topology.real_space_chern(h, operator=sz, write=False,
            mode="kpm", radius=4, npol=100)
    core = _core(r)
    assert np.mean(np.abs(c0[core])) > 1.5 # far from zero
    assert np.abs(c - _exact_local(h, 4, operator=sz))[core].max() < 4e-3
    assert np.abs(c - c0)[core].max() < 0.2


def test_groups_of_sites(haldane, monkeypatch):
    """A large flake goes in groups of sites, each with its own call of the
    density matrix; with the recursion not truncated the groups change
    nothing"""
    h, r, c0 = haldane
    _, c1 = h.get_kpm_chern_marker(radius=3, npol=50, kpm_radius=100)
    monkeypatch.setattr(kpmchern, "_SITES_PER_CALL", 200)
    _, c2 = h.get_kpm_chern_marker(radius=3, npol=50, kpm_radius=100)
    assert np.allclose(c1, c2, atol=1e-10)


def test_method_matches_function(haldane):
    h, r, c0 = haldane
    _, c1 = topology.real_space_chern(h, write=False, mode="kpm", radius=2,
            npol=50)
    _, c2 = h.get_kpm_chern_marker(radius=2, npol=50)
    assert np.allclose(c1, c2)


def test_guards(haldane):
    h, r, c0 = haldane
    with pytest.raises(ValueError, match="'dense', 'kpm'"):
        topology.real_space_chern(h, write=False, mode="stochastic")
    with pytest.raises(TypeError, match="radius"):
        topology.real_space_chern(h, write=False, radius=3) # a kpm keyword
    with pytest.raises(ValueError, match="radius"):
        h.get_kpm_chern_marker(radius=0)
    with pytest.raises(ValueError, match="npol"):
        h.get_kpm_chern_marker(npol=1.5)
    with pytest.raises(TypeError, match="operator"):
        h.get_kpm_chern_marker(operator="sz")
    from pyqula import geometry
    hp = geometry.honeycomb_lattice().get_hamiltonian(has_spin=False) # 2d
    with pytest.raises(ValueError, match="0d"):
        hp.get_kpm_chern_marker()
