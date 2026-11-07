"""h.get_anomalous_hall_conductivity of 3D crystals, a conductivity per unit
length in units of e^2/h, with the plane chosen by direction=.

The invariants: a stack of decoupled Chern layers with spacing c gives
C/c, and in a metal the 2D result per spacing, exactly on the grid and
exactly with the tetrahedra when the 2D layer is meshed in the basis
(a1,-a2), whose triangles are the projections of the tetrahedra; with an
interlayer hopping the tetrahedra converge at second order to the mean over
kz of the 2D slices; and rotating the crystal moves the result to the
rotated plane.
"""
import numpy as np
import pytest

from pyqula import geometry
from pyqula import multihopping
from pyqula.topologytk import anomaloushall


def _dense(m):
    return np.array(m.todense()) if hasattr(m, "todense") else np.array(m)


def _haldane(t2=0.1, imbalance=0.2):
    """Spinless Haldane model with a staggered onsite, C=1 at half filling,
    gap from -0.25 to 0.25, a metal at fermi=0.5"""
    g = geometry.honeycomb_lattice()
    h = g.get_hamiltonian(has_spin=False)
    h.add_haldane(t2)
    h.add_sublattice_imbalance(imbalance)
    return h


def _mass(tm):
    return np.diag([tm, -tm]).astype(np.complex128)


def _stack(h2, c=2., tm=0.):
    """3D crystal of copies of the 2D h2 stacked along z with spacing c,
    with the interlayer hopping diag(tm,-tm), a mass that depends on kz"""
    h = h2.copy()
    h.geometry.a3 = np.array([0., 0., c])
    h.geometry.dimensionality = 3
    hd = {(R[0], R[1], 0): _dense(m) for R, m in h2.get_hopping_dict().items()}
    if tm != 0.:
        hd[(0, 0, 1)] = _mass(tm)
        hd[(0, 0, -1)] = _mass(tm)
    h.dimensionality = 3
    h.set_multihopping(multihopping.MultiHopping(hd))
    return h


def _slice(h2, kz, tm):
    """The 2D Hamiltonian at fixed kz (reduced) of _stack(h2, tm=tm)"""
    h = h2.copy()
    hd = {R: _dense(m) for R, m in h2.get_hopping_dict().items()}
    hd[(0, 0, 0)] = hd[(0, 0, 0)] + 2.*np.cos(2.*np.pi*kz)*_mass(tm)
    h.set_multihopping(multihopping.MultiHopping(hd))
    return h


def _mirror(h2):
    """The same 2D crystal in the lattice basis (a1,-a2)"""
    h = h2.copy()
    h.geometry.a2 = -np.array(h2.geometry.a2)
    hd = {(R[0], -R[1], 0): _dense(m) for R, m in h2.get_hopping_dict().items()}
    h.set_multihopping(multihopping.MultiHopping(hd))
    return h


def _rotate(h):
    """The same crystal rotated by the proper rotation x->y, y->z, z->x of
    every position and lattice vector; the hoppings do not change"""
    h = h.copy()
    g = h.geometry
    rot = lambda v: np.array([v[2], v[0], v[1]])
    g.r = np.array([rot(r) for r in g.r])
    g.r2xyz()
    g.a1, g.a2, g.a3 = rot(g.a1), rot(g.a2), rot(g.a3)
    return h


def test_decoupled_chern_layers_give_c_over_the_spacing():
    """In the gap, C/c, the same number as the 2D Chern number per spacing
    to rounding, with the Bloechl term or without it"""
    h2 = _haldane()
    h3 = _stack(h2, c=2.)
    for bloechl in (True, False):
        s2 = h2.get_anomalous_hall_conductivity(nk=24, bloechl=bloechl)
        s3 = h3.get_anomalous_hall_conductivity(nk=24, bloechl=bloechl)
        assert abs(s2-1.) < 1e-2, s2
        assert abs(2.*s3-s2) < 1e-12, (s3, s2)


def test_decoupled_metal_layers_give_the_2d_result_per_spacing():
    """In a metal, the grid gives the 2D result per spacing exactly. The
    tetrahedra give exactly the triangles of the 2D layer meshed in the
    basis (a1,-a2), which is where they project, without the Bloechl term,
    whose coefficient depends on the dimension. The 2D results of the two
    bases differ at second order, here 1.3e-2 at nk=24"""
    h2 = _haldane()
    hm = _mirror(h2)
    h3 = _stack(h2, c=2.)
    kw = dict(nk=24, fermi=0.5)
    g2 = h2.get_anomalous_hall_conductivity(integration="grid", T=0.02, **kw)
    g3 = h3.get_anomalous_hall_conductivity(integration="grid", T=0.02, **kw)
    assert abs(2.*g3-g2) < 1e-12, (g3, g2)
    for T in (0., 0.02):
        tm = hm.get_anomalous_hall_conductivity(T=T, bloechl=False, **kw)
        t3 = h3.get_anomalous_hall_conductivity(T=T, bloechl=False, **kw)
        assert abs(tm) > 0.5
        assert abs(2.*t3-tm) < 1e-12, (T, t3, tm)


def test_coupled_layers_converge_to_the_kz_slices():
    """With the kz dependent mass the crystal is a 3D metal at fermi=0.5.
    sigma_xy is the mean over kz of the 2D result of each slice, divided by
    c, since the in-plane velocities do not depend on kz. On 12 slices, each
    with the 2D triangles at nk=128, that reference is within about 3e-4 of
    its limit (the 2D triangles are within 2e-3 at nk=48 and converge at
    second order, and the kz mean of a smooth periodic function converges
    exponentially).
    The tetrahedra at nk=16 and nk=32 approach it at second order"""
    h2 = _haldane()
    tm = 0.05
    h3 = _stack(h2, c=2., tm=tm)
    kzs = np.arange(12)/12.
    ref = np.mean([_slice(h2, kz, tm).get_anomalous_hall_conductivity(
        nk=128, fermi=0.5) for kz in kzs])/2.
    e16 = abs(h3.get_anomalous_hall_conductivity(nk=16, fermi=0.5)-ref)
    e32 = abs(h3.get_anomalous_hall_conductivity(nk=32, fermi=0.5)-ref)
    assert e32 < 1e-2*abs(ref), (e16, e32, ref)
    assert e16/e32 > 3., (e16, e32)


def test_coupled_insulator_is_quantized():
    """With the kz dependent mass and the Fermi level in the gap, every
    slice has C=1, so sigma_xy = 1/c (to 6e-4 at nk=32, where the gap of
    0.3 is resolved), and nothing flows in the other planes, since Omega_x
    and Omega_y are odd in kz"""
    h3 = _stack(_haldane(), c=2., tm=0.05)
    s = h3.get_anomalous_hall_conductivity(nk=32)
    assert abs(2.*s-1.) < 2e-3, s
    for d in ("yz", "zx"):
        assert abs(h3.get_anomalous_hall_conductivity(nk=12,
                direction=d)) < 1e-12


def test_a_rotated_crystal_moves_the_hall_conductivity_to_the_rotated_plane():
    """x->y, y->z, z->x maps the xy plane onto yz, so sigma_yz of the
    rotated metal is sigma_xy of the original one, and the reversed pair
    flips the sign. sigma_xy of the rotated crystal is sigma_zx of the
    original one, which vanishes on the grid since Omega_y is odd in kz;
    the tetrahedra are not symmetric under kz -> -kz, so in a metal they
    give it only to their second order error (6e-5 here)"""
    h3 = _stack(_haldane(), c=2., tm=0.05)
    hr = _rotate(h3)
    kw = dict(nk=12, fermi=0.5, T=0.02)
    sxy = h3.get_anomalous_hall_conductivity(**kw)
    syz = hr.get_anomalous_hall_conductivity(direction="yz", **kw)
    szy = hr.get_anomalous_hall_conductivity(direction="zy", **kw)
    rxy = hr.get_anomalous_hall_conductivity(direction="xy",
            integration="grid", **kw)
    assert abs(sxy) > 0.1
    assert abs(syz-sxy) < 1e-12, (syz, sxy)
    assert abs(szy+sxy) < 1e-12, (szy, sxy)
    assert abs(rxy) < 1e-12, rxy


def test_unknown_direction_lists_the_accepted_ones():
    h3 = _stack(_haldane())
    with pytest.raises(ValueError) as err:
        h3.get_anomalous_hall_conductivity(nk=4, direction="xx")
    for name in anomaloushall.get_direction_names():
        assert name in str(err.value)


def test_a_2d_hamiltonian_refuses_an_out_of_plane_direction():
    h = geometry.honeycomb_lattice().get_hamiltonian(has_spin=False)
    h.add_haldane(0.1)
    with pytest.raises(ValueError, match="in the plane"):
        h.get_anomalous_hall_conductivity(nk=8,direction="yz")


def test_operator_berry_runs_on_a_3d_hamiltonian():
    from pyqula import topology
    h = geometry.cubic_lattice().get_hamiltonian(has_spin=False)
    topology.operator_berry(h,k=[0.1,0.2,0.3])
