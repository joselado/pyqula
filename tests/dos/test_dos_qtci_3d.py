"""integration="qtci" of the DOS in 3D (dostk/simplexdos.py): the tetrahedron
integration on a 2^R x 2^R x 2^R mesh whose band energies come from a
quantics cross interpolation with three quantics coordinates.

The invariants: on the same mesh it is the tetrahedron DOS to the
interpolation tolerance, in either bit ordering; and a stack of decoupled
layers gives the 2D qtci DOS of the layer. The six tetrahedra of a cube
share its main diagonal, so their projections on the layer are the two
triangles of the square cut along (0,0)-(1,1), while the 2D triangles cut it
along (1,0)-(0,1). So the stack is compared with the layer in the lattice
basis (a1,-a2). With a band independent of kz the tetrahedron interpolation
is the triangle one of the layer, and the three tetrahedra above one
triangle fill its prism uniformly, so for the DOS (unit weights) the two
agree exactly, not only to second order.
"""
import numpy as np
import pytest

from pyqula import geometry
from pyqula import multihopping
from pyqula.dostk import simplexdos


def _dense(m):
    return np.array(m.todense()) if hasattr(m, "todense") else np.array(m)


def _layer():
    """gapped graphene, spinless, two bands"""
    h = geometry.honeycomb_lattice().get_hamiltonian(has_spin=False)
    h.add_sublattice_imbalance(0.3)
    return h


def _stack(h2, c=2., tm=0.):
    """3D crystal of copies of the 2D h2 stacked along z with spacing c,
    with the interlayer hopping diag(tm,-tm)"""
    h = h2.copy()
    h.geometry.a3 = np.array([0., 0., c])
    h.geometry.dimensionality = 3
    hd = {(R[0], R[1], 0): _dense(m) for R, m in h2.get_hopping_dict().items()}
    if tm != 0.:
        M = np.diag([tm, -tm]).astype(np.complex128)
        hd[(0, 0, 1)] = M
        hd[(0, 0, -1)] = M
    h.dimensionality = 3
    h.set_multihopping(multihopping.MultiHopping(hd))
    return h


def _mirror(h2):
    """The same 2D crystal in the lattice basis (a1,-a2)"""
    h = h2.copy()
    h.geometry.a2 = -np.array(h2.geometry.a2)
    hd = {(R[0], -R[1], 0): _dense(m) for R, m in h2.get_hopping_dict().items()}
    h.set_multihopping(multihopping.MultiHopping(hd))
    return h


@pytest.mark.parametrize("ordering", ["interleaved", "fused"])
def test_qtci_3d_is_the_tetrahedron_dos_on_the_same_mesh(ordering):
    """On the cubic lattice the qtci DOS is the tetrahedron DOS of the
    same 16^3 mesh to the interpolation tolerance, at finite delta and at
    delta=0, in both bit orderings"""
    h = geometry.cubic_lattice().get_hamiltonian(has_spin=False)
    es = np.linspace(-6.5, 6.5, 131)+0.0037
    for delta in (1e-2, 0.):
        _, d_tet, n_tet = simplexdos.dos_triangle_counted(h, es, nk=16,
                delta=delta)
        _, d_q, n_q, ranks = simplexdos.dos_qtci_counted(h, es, nk=16,
                delta=delta, tolerance=1e-10, ordering=ordering)
        assert np.max(np.abs(d_q-d_tet)) < 1e-6*np.max(d_tet)
        assert n_q <= n_tet


def test_qtci_3d_of_a_layered_metal_is_the_tetrahedron_dos():
    """A layered two-band metal with a kz-dependent mass, at 16^3, through
    h.get_dos"""
    h = _stack(_layer(), tm=0.3)
    es = np.linspace(-3.8, 3.8, 121)+0.0037
    _, d_tet = h.get_dos(integration="triangle", energies=es, nk=16,
            delta=1e-2, write=False)
    _, d_q = h.get_dos(integration="qtci", energies=es, nk=16, delta=1e-2,
            tolerance=1e-10, write=False)
    assert np.max(np.abs(d_q-d_tet)) < 1e-6*np.max(d_tet)


def test_decoupled_layers_give_the_2d_qtci_dos():
    """A stack of decoupled layers has the DOS of one layer: the 3D qtci
    DOS of the stack is the 2D qtci DOS of the layer in the basis (a1,-a2)
    on the same 2^R mesh, to the interpolation tolerance, and both are the
    2D triangle DOS of that layer"""
    h2 = _mirror(_layer())
    h3 = _stack(_layer())
    es = np.linspace(-3.4, 3.4, 121)+0.0037
    _, d3, n3, r3 = simplexdos.dos_qtci_counted(h3, es, nk=32, delta=1e-2,
            tolerance=1e-10)
    _, d2, n2, r2 = simplexdos.dos_qtci_counted(h2, es, nk=32, delta=1e-2,
            tolerance=1e-10)
    _, dt, nt = simplexdos.dos_triangle_counted(h2, es, nk=32, delta=1e-2)
    scale = np.max(dt)
    assert np.max(np.abs(d3-d2)) < 1e-6*scale
    assert np.max(np.abs(d3-dt)) < 1e-6*scale
    # the field does not depend on kz, so the 3D interpolation visits a
    # fraction of the 32^3 mesh (a quarter, at this tight tolerance)
    assert n3 < 0.5*32**3


def test_qtci_ldos_still_refuses_3d():
    h = geometry.cubic_lattice().get_hamiltonian(has_spin=False)
    with pytest.raises(NotImplementedError, match="3D"):
        simplexdos.ldos_qtci_counted(h, np.linspace(-1, 1, 5), nk=8)


def test_qtci_dos_refuses_1d():
    h = geometry.chain().get_hamiltonian(has_spin=False)
    with pytest.raises(NotImplementedError, match="2D and 3D"):
        h.get_dos(integration="qtci", nk=8, write=False)
