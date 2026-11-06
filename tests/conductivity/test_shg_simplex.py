# Tests of the triangle and qtci integrations of the second-harmonic
# conductivity (conductivitytk/shgsimplex.py). Each one checks the new
# k-integration against an independent path (the grid sum of shg.py, a
# brute-force quadrature of one simplex) or against an exact relation,
# never against a recorded number.
import numpy as np
import pytest

from pyqula import geometry
from pyqula import conductivity
from pyqula.conductivitytk import shg
from pyqula.conductivitytk import shgsimplex


def _gapped_honeycomb(has_spin=False):
    """Two-band model with broken inversion, gap 0.6 t"""
    h = geometry.honeycomb_lattice().get_hamiltonian(has_spin=has_spin)
    h.add_sublattice_imbalance(0.3)
    return h


def _triangular_three_sublattice():
    """Three bands with no inversion and no degeneracy, so that the
    products of two resonances of three different bands are present"""
    g = geometry.honeycomb_lattice()
    g.x = np.array([-0.5,0.5,-1.0])
    g.y = np.array([0.,0.,np.sqrt(3)/2])
    g.z = np.zeros(3)
    g.xyz2r()
    g.has_sublattice = False
    g.get_fractional()
    h = g.get_hamiltonian(has_spin=False)
    h.add_onsite([-3.,0.5,3.])
    return h


def _quadrature(f, n=400):
    """Average of f(u,v) over the unit triangle u,v >= 0, u+v <= 1, by
    Gauss-Legendre on the square mapped onto the triangle (u = s(1-t),
    v = st, Jacobian s)"""
    x,w = np.polynomial.legendre.leggauss(n)
    x = 0.5*(x+1.) ; w = 0.5*w
    s,t = np.meshgrid(x,x,indexing="ij")
    ws = np.outer(w,w)*s
    return np.sum(ws*f(s*(1.-t),s*t))/np.sum(ws)


def _buffers():
    return np.zeros(4),np.zeros((4,4),dtype=np.complex128),np.zeros(48)


def test_simplex_averages_against_quadrature():
    """The Hermite-Genocchi average of one resonance over a triangle with a
    linear energy is exact. The average of a product of two resonances is
    exact when the two energies differ by a constant on the triangle, and
    the confluent cases (all vertex energies equal) reproduce the value at
    a point"""
    z = 0.7+0.05j
    x,F,hb = _buffers()
    for e in [np.array([0.0,1.0,0.3]),np.array([0.6,0.72,0.69]),
              np.array([0.7,0.7,0.70001])]:
        E = lambda u,v: e[0]+(e[1]-e[0])*u+(e[2]-e[0])*v
        ref = _quadrature(lambda u,v: 1./(z-E(u,v)))
        got = shgsimplex._single_average(e.copy(),3,z,x,F,hb)
        assert abs(got-ref)<1e-5*abs(ref)
        c = e+0.37 # C - A constant: x(k,s) is linear on the prism
        ref = _quadrature(lambda u,v: 1./((z-E(u,v))*(2*z-2*(E(u,v)+0.37))))
        got = shgsimplex._product_average(e.copy(),c.copy(),3,z,x,F,hb)
        assert abs(got-ref)<1e-5*abs(ref)
    a = np.full(3,0.6) ; c = np.full(3,0.9)
    got = shgsimplex._product_average(a,c,3,z,x,F,hb)
    assert abs(got-1./((z-0.6)*(2*z-1.8)))<1e-12
    a = np.full(3,0.6) ; c = np.full(3,0.6) # double resonance at a point
    got = shgsimplex._product_average(a,c,3,z,x,F,hb)
    assert abs(got-1./((z-0.6)*(2*z-1.2)))<1e-10


def test_triangle_agrees_with_the_grid_at_large_broadening():
    """At a broadening the grid resolves, the grid sum is converged and the
    triangle result must approach it at second order in 1/nk for a
    two-band model, where every resonant factor is integrated exactly on
    the interpolated bands"""
    h = _gapped_honeycomb()
    ws = [0.4,0.9,1.5,2.5]
    kw = dict(energies=ws,T=1e-3,delta=0.1,components="xxx")
    _,ref = h.get_shg_conductivity(nk=192,**kw)
    _,ref2 = h.get_shg_conductivity(nk=96,**kw)
    scale = np.max(np.abs(ref))
    assert np.max(np.abs(ref-ref2))<1e-3*scale # the grid is converged
    err = []
    for nk in [64,128]:
        _,s = h.get_shg_conductivity(nk=nk,integration="triangle",**kw)
        err.append(np.max(np.abs(s-ref))/scale)
    assert err[1]<1e-2
    assert 3.<err[0]/err[1]<5.


def test_triangle_agrees_with_the_grid_for_three_bands():
    """The same with three bands, where the products of a resonance at w
    and one at 2w of three different bands are present, at frequencies
    across both resonant shells. A single triangulation of the prism of
    such a product converges only at first order; the average over the
    vertex orderings must bring it to second order"""
    h = _triangular_three_sublattice()
    ws = [1.2,2.0,3.0,4.5]
    kw = dict(energies=ws,T=1e-3,delta=0.15,components=["xxx","yxy"])
    _,ref = h.get_shg_conductivity(nk=120,**kw)
    for c in ["xxx","yxy"]: assert np.max(np.abs(ref[c]))>1e-3
    err = []
    for nk in [32,64]:
        _,s = h.get_shg_conductivity(nk=nk,integration="triangle",**kw)
        err.append(max(np.max(np.abs(s[c]-ref[c]))/np.max(np.abs(ref[c]))
            for c in ["xxx","yxy"]))
    assert err[1]<5e-3
    assert 3.<err[0]/err[1]<5.


def test_triangle_agrees_with_the_grid_in_1d():
    """In 1D the simplices are segments and the prism of a product is a
    square cut into two triangles. A chain with three sites per cell and
    no inversion"""
    g = geometry.chain()
    g.a1 = np.array([3.,0.,0.])
    g.x = np.array([0.0,1.0,2.1]) ; g.y = np.zeros(3) ; g.z = np.zeros(3)
    g.xyz2r()
    g.has_sublattice = False
    g.update_reciprocal() ; g.get_fractional()
    def tij(r1,r2): # first neighbors at distances 1.0, 1.1 and 0.9
        dr = np.linalg.norm(r1-r2)
        return -1. if 0.5<dr<1.2 else 0.
    h = g.get_hamiltonian(has_spin=False,tij=tij)
    h.add_onsite([-1.,0.3,1.2])
    ws = [0.8,1.4,2.2]
    kw = dict(energies=ws,T=1e-3,delta=0.05,components="xxx")
    _,ref = h.get_shg_conductivity(nk=4000,**kw)
    _,s = h.get_shg_conductivity(nk=800,integration="triangle",**kw)
    scale = np.max(np.abs(ref))
    assert scale>1e-3
    assert np.max(np.abs(s-ref))<1e-2*scale


def test_triangle_converges_below_the_mesh_resolution():
    """At a broadening far below the energy spacing of the mesh, two
    triangle meshes agree while the grid on the finer one does not"""
    h = _gapped_honeycomb()
    ws = [0.45,0.8,1.3]
    kw = dict(energies=ws,T=1e-3,delta=1e-3,components="xxx")
    _,s96 = h.get_shg_conductivity(nk=96,integration="triangle",**kw)
    _,s192 = h.get_shg_conductivity(nk=192,integration="triangle",**kw)
    _,g192 = h.get_shg_conductivity(nk=192,**kw)
    scale = np.max(np.abs(s192))
    assert np.max(np.abs(s96-s192))<2e-2*scale
    assert np.max(np.abs(g192-s192))>1e-1*scale


def test_spinful_copy_doubles_the_response():
    """A spinful copy gives exactly twice the response, with both simplex
    integrations: every band is then a degenerate pair, which goes through
    the multiplet averaging of the weights and the per-simplex detection of
    the exact product cases"""
    ws = [0.5,1.0,2.0]
    for integration in ["triangle","qtci"]:
        kw = dict(energies=ws,nk=32,T=1e-3,delta=2e-2,components="xxx",
                integration=integration,tolerance=1e-8)
        _,s1 = _gapped_honeycomb().get_shg_conductivity(**kw)
        _,s2 = _gapped_honeycomb(has_spin=True).get_shg_conductivity(**kw)
        assert np.allclose(s2,2.*s1,atol=1e-6*np.max(np.abs(s1)))


def test_qtci_reproduces_the_triangle_result_on_the_same_mesh():
    """With integration='qtci' the band energies and weights come from a
    quantics tensor cross interpolation; the simplex integral on top is the
    same, so on the same 2^R mesh the two agree to the interpolation
    tolerance"""
    h = _gapped_honeycomb()
    kw = dict(energies=[0.4,0.9,2.0],nk=64,T=1e-3,delta=1e-2,
            components=["xxx","yxy"])
    _,s_tri = h.get_shg_conductivity(integration="triangle",**kw)
    _,s_q = h.get_shg_conductivity(integration="qtci",tolerance=1e-8,**kw)
    for c in ["xxx","yxy"]:
        scale = np.max(np.abs(s_tri[c]))
        assert np.max(np.abs(s_q[c]-s_tri[c]))<1e-4*scale


def test_options_are_checked():
    """An unknown integration lists the accepted ones, the simplex
    integrations refuse the length gauge, and qtci is 2D only"""
    h = _gapped_honeycomb()
    with pytest.raises(ValueError) as err:
        h.get_shg_conductivity(energies=[1.],nk=4,integration="simpson")
    for name in shg.get_integration_names():
        assert name in str(err.value)
    with pytest.raises(NotImplementedError,match="velocity"):
        h.get_shg_conductivity(energies=[1.],nk=4,T=0.,gauge="length",
                integration="triangle")
    hc = geometry.chain().get_hamiltonian(has_spin=False)
    with pytest.raises(NotImplementedError,match="2D"):
        hc.get_shg_conductivity(energies=[1.],nk=8,integration="qtci")
