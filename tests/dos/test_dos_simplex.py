# Tests of the triangle (segment, tetrahedron) and qtci integrations of the
# density of states (dostk/simplexdos.py). Each one checks the simplex
# integration against an independent path -- the Lorentzian grid sum, the
# exact DOS of the chain and of graphene, a Zeeman-shifted copy of a
# spinless DOS -- or against an exact relation, never a recorded number.
import numpy as np
import pytest
from scipy.special import ellipk

from pyqula import geometry
from pyqula import dos
from pyqula.dostk import simplexdos


def _graphene():
    return geometry.honeycomb_lattice().get_hamiltonian(has_spin=False)


def _graphene_dos(E):
    """Exact DOS per unit cell of first-neighbor graphene with t=1, both
    bands, from the complete elliptic integral"""
    E = np.abs(np.asarray(E,dtype=float))
    out = np.zeros_like(E)
    m1 = (E>0)&(E<1); m2 = (E>1)&(E<3)
    z = lambda e: (1+e)**2-(e**2-1)**2/4
    Z0 = np.where(m1,z(E),4*E); Z1 = np.where(m1,4*E,z(E))
    ok = m1|m2
    m = np.minimum(Z1[ok]/Z0[ok],1.-1e-16)
    out[ok] = (2/np.pi**2)*E[ok]/np.sqrt(Z0[ok])*ellipk(m)
    return out


def test_triangle_agrees_with_the_grid_sum_at_large_delta():
    """At a broadening a fine grid resolves, a coarse triangle mesh gives
    the same DOS, across the Dirac point and the van Hove energies"""
    h = _graphene()
    es = np.linspace(-3.5,3.5,71)
    _,d_tri = h.get_dos(integration="triangle",energies=es,nk=60,
            delta=0.1,write=False)
    _,d_grid = h.get_dos(integration="grid",energies=es,nk=600,delta=0.1,
            write=False)
    assert np.max(np.abs(d_tri-d_grid)) < 3e-3*np.max(d_grid)


def test_triangle_gives_the_exact_dos_of_graphene_at_zero_delta():
    """At delta=0 the triangle DOS converges to the exact DOS of graphene,
    a quantity no Lorentzian sum can give. The energies avoid the
    logarithmic van Hove singularities at +-1 and the Dirac point"""
    h = _graphene()
    es = np.array([-2.5,-1.6,-0.6,0.3,0.8,1.3,2.2,2.9])
    _,d = h.get_dos(integration="triangle",energies=es,nk=301,delta=0.,
            write=False)
    assert np.max(np.abs(d-_graphene_dos(es))) < 1e-3


def test_segment_gives_the_exact_dos_of_the_chain():
    """In 1D the simplices are segments, and the DOS of the chain with t=1
    is 1/(pi sqrt(4-E^2)). A linear band on a segment has a constant
    density, so pointwise the error is first order in 1/nk (it is the
    slope of the band that varies across the segment): it must halve when
    nk doubles"""
    h = geometry.chain().get_hamiltonian(has_spin=False)
    es = np.linspace(-1.9,1.9,39)
    exact = 1./(np.pi*np.sqrt(4.-es**2))
    errs = []
    for nk in [1001,2001,4001]:
        _,d = h.get_dos(integration="triangle",energies=es,nk=nk,delta=0.,
                write=False)
        errs.append(np.max(np.abs(d/exact-1.)))
    assert errs[1] < 1e-2
    assert errs[2] < 0.7*errs[0]


@pytest.mark.parametrize("delta",[0.,1e-2])
def test_triangle_obeys_the_sum_rule(delta):
    """The integral of the DOS is the number of bands. Two things keep it
    from being exact on a grid of energies: a flat triangle (the one
    centered on a K point when 3 does not divide nk, by C3 symmetry) is a
    delta function in energy that no grid point sees, and a triangle whose
    spread is below the energy step is sampled at one point or none. Both
    are of the order of the weight of a few triangles, 1/(2 nk^2) each,
    far below the factors of 2 and pi a normalization error would give.
    At finite delta the Lorentzian tails outside the window carry about
    (2 delta/pi)/5 per band"""
    h = geometry.honeycomb_lattice().get_hamiltonian(has_spin=True)
    h.add_rashba(0.2)
    h.add_sublattice_imbalance(0.3)
    es = np.linspace(-8.,8.,8001)
    _,d = h.get_dos(integration="triangle",energies=es,nk=41,delta=delta,
            write=False)
    assert abs(np.trapezoid(d,es)-4.) < 1e-2


def test_zero_delta_is_the_limit_of_finite_delta():
    """The delta=0 path (bisection over the energies of each triangle) and
    the finite-delta kernel are different code; they meet at delta -> 0"""
    h = _graphene()
    es = np.linspace(-3.2,3.2,161)+0.0037
    _,d0 = h.get_dos(integration="triangle",energies=es,nk=51,delta=0.,
            write=False)
    _,d1 = h.get_dos(integration="triangle",energies=es,nk=51,delta=1e-9,
            write=False)
    assert np.max(np.abs(d0-d1)) < 1e-5*np.max(d0)


def test_operator_weight_of_a_zeeman_split_copy():
    """With an exchange field J along z the spin-up and spin-down copies of
    graphene are the spinless DOS shifted by -+J, so the sz-weighted DOS is
    the difference of two shifted spinless DOS, and the unweighted one
    their sum. J is larger than half the bandwidth so that the two copies
    do not cross: the bands are ordered by energy at each k, and where two
    copies cross that ordering puts a kink inside the triangles, which the
    linear interpolation only resolves as the mesh is refined"""
    J = 3.5
    h0 = _graphene()
    h = geometry.honeycomb_lattice().get_hamiltonian(has_spin=True)
    h.add_exchange([0.,0.,J])
    es = np.linspace(-6.5,6.5,131)+0.0037
    kw = dict(integration="triangle",nk=61,delta=0.,write=False)
    _,dp = h0.get_dos(energies=es+J,**kw)
    _,dm = h0.get_dos(energies=es-J,**kw)
    _,dtot = h.get_dos(energies=es,**kw)
    _,dsz = h.get_dos(energies=es,operator="sz",**kw)
    assert np.max(np.abs(dtot-(dp+dm))) < 1e-10
    err = min(np.max(np.abs(dsz-(dp-dm))),np.max(np.abs(dsz-(dm-dp))))
    assert err < 1e-10
    assert np.max(np.abs(dp-dm)) > 0.1 # the test is not vacuous


def test_multiplet_average_is_basis_independent():
    """In spinful graphene every band is doubly degenerate, and the sx
    weight of one member of a pair depends on the basis eigh picks. The
    multiplet average does not: the sx-weighted DOS vanishes and the
    weight of the identity is the total DOS"""
    h = geometry.honeycomb_lattice().get_hamiltonian(has_spin=True)
    es = np.linspace(-3.,3.,31)+0.0037
    kw = dict(integration="triangle",nk=31,delta=1e-2,write=False)
    _,dsx = h.get_dos(energies=es,operator="sx",**kw)
    _,dtot = h.get_dos(energies=es,**kw)
    _,did = h.get_dos(energies=es,operator=np.identity(4),**kw)
    assert np.max(np.abs(dsx)) < 1e-10
    assert np.max(np.abs(did-dtot)) < 1e-10


def test_qtci_reproduces_the_triangle_on_the_same_mesh():
    """The qtci path runs the triangle integration on a 2^R mesh whose
    band energies come from a cross interpolation, so on the same mesh it
    must agree with the plain triangle integration to the interpolation
    tolerance. On a mesh this small the interpolation visits about every
    point; the saving in diagonalizations appears at larger R"""
    h = _graphene()
    h.add_sublattice_imbalance(0.3)
    es = np.linspace(-3.2,3.2,81)+0.0037
    _,d_tri,n_tri = simplexdos.dos_triangle_counted(h,es,nk=128,delta=1e-2)
    _,d_q,n_q,ranks = simplexdos.dos_qtci_counted(h,es,nk=128,delta=1e-2,
            tolerance=1e-10)
    assert np.max(np.abs(d_q-d_tri)) < 1e-6*np.max(d_tri)
    assert n_q <= n_tri


def test_tetrahedron_agrees_with_the_grid_and_the_sum_rule():
    """In 3D the simplices are tetrahedra: on the cubic lattice the DOS at
    large delta matches a fine grid, and at delta=0 it integrates to one.
    At large delta the grid converges exponentially and the tetrahedra at
    second order, so the comparison is limited by the tetrahedra"""
    h = geometry.cubic_lattice().get_hamiltonian(has_spin=False)
    es = np.linspace(-6.5,6.5,131)+0.0037
    _,d_tet = h.get_dos(integration="triangle",energies=es,nk=30,delta=0.2,
            write=False)
    _,d_grid = h.get_dos(integration="grid",energies=es,nk=60,delta=0.2,
            write=False)
    assert np.max(np.abs(d_tet-d_grid)) < 5e-3*np.max(d_grid)
    es = np.linspace(-6.2,6.2,2481)
    _,d0 = h.get_dos(integration="triangle",energies=es,nk=21,delta=0.,
            write=False)
    assert abs(np.trapezoid(d0,es)-1.) < 1e-3


def test_unknown_names_list_the_registry():
    h = _graphene()
    with pytest.raises(ValueError) as e:
        h.get_dos(integration="tetra",write=False)
    for name in dos.get_integration_names(): assert name in str(e.value)
    with pytest.raises(ValueError) as e:
        h.get_dos(mode="EDD",write=False)
    for name in dos.get_dos_modes(): assert name in str(e.value)
