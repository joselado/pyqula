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
from pyqula.conductivitytk import kubo
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


def _kernels_on_the_same_fields(h,trip,nk,ws,delta):
    """Both simplex kernels, the source kernel (the default) and the
    direct sum over simplices, on the same fields of the uniform mesh"""
    sym = shgsimplex._symmetrized_triplets(trip)
    tms,dirs,d,hkgen,cellvol,scale = shg._derivative_tensors(h)
    tol = 1e-6*scale
    cat = shgsimplex._term_catalogue(tms.shape[1])
    ks = kubo._kmesh(h,nk)
    es,W = shgsimplex._fields(tms,dirs,d,hkgen,ks,1e-3,
            np.array(sym,dtype=np.int64),cat,tol)
    terms,W = shgsimplex._prune(cat[0],W)
    out = [shgsimplex._sigma_from_fields(h.dimensionality,nk,es,terms,W,
            np.array(ws),delta,cellvol,tol,kernel=k)
            for k in ["sources","simplices"]]
    return out,terms


def test_source_kernel_equals_the_direct_sum():
    """The source kernel builds the coefficients of the simplex sum once
    and then sums them per frequency. It decides its Taylor branches on
    delta and the direct sum per frequency, so at this broadening they
    agree to the rounding loss of the source kernel far from resonance, a
    few 1e-11, not to the discretization error. A two-band model (only
    single resonances and the exact product cases) and a three-band one
    (products of three different bands, with the prism pieces), at a
    broadening below the energy spacing of the mesh, where many nodes fall
    in the Taylor clusters"""
    cases = [(_gapped_honeycomb(),[(0,0,0)],40,[0.35,0.8,1.5,2.6]),
            (_triangular_three_sublattice(),[(0,0,0),(1,0,1)],24,
                [1.2,2.0,3.0,4.5])]
    for h,trip,nk,ws in cases:
        for delta in [1e-3,2e-2]:
            (new,old),terms = _kernels_on_the_same_fields(h,trip,nk,ws,delta)
            scale = np.max(np.abs(old))
            assert scale>0.
            assert np.max(np.abs(new-old))<1e-9*scale
    three = (terms[:,0]==shgsimplex._PRODUCT) & (terms[:,3]!=terms[:,4]) \
            & ((terms[:,1]!=terms[:,3]) | (terms[:,2]!=terms[:,4]))
    assert np.any(three) # products of three different bands are there
    with pytest.raises(ValueError) as err:
        shgsimplex._sigma_from_fields(2,4,None,None,None,[1.],1e-2,1.,1e-6,
                kernel="direct")
    for name in shgsimplex._kernels: assert name in str(err.value)


def test_nearly_flat_simplices_far_from_resonance():
    """A simplex whose energies spread by s, with delta/4 < s much smaller
    than its distance abs(x) to the frequency. A switch to the Taylor
    series decided on s compared with delta alone sends it to the plain
    divided difference, which cancels terms of size (abs(x)/s)^r there: at
    omega = 0, delta = 1e-6 and abs(x) = 0.5 it lost 4e-4 of a single
    resonance and more than the whole value of a product of two. The
    direct sum decides per frequency and must give the quadrature value
    to rounding, far from resonance and on it"""
    delta = 1e-6
    x,F,hb = _buffers()
    for shape in [(0.,0.4,1.),(0.,0.02,1.)]:
        for r in [0.3,0.45,1.,3.,10.]:
            s = r*delta
            e = 0.5+s*np.array(shape)
            E = lambda u,v: e[0]+(e[1]-e[0])*u+(e[2]-e[0])*v
            for omega in [0.,0.5+0.5*s]: # far from the triangle, and on it
                z = omega+1j*delta
                ref = _quadrature(lambda u,v: 1./(z-E(u,v)))
                got = shgsimplex._single_average(e.copy(),3,z,x,F,hb)
                assert abs(got-ref)<1e-11*abs(ref)
                c = e+0.3*s # C - A constant, so the prism average is exact
                ref = _quadrature(lambda u,v: 1./((z-E(u,v))
                        *(2*z-2*(E(u,v)+0.3*s))))
                got = shgsimplex._product_average(e.copy(),c.copy(),3,z,
                        x,F,hb)
                assert abs(got-ref)<1e-11*abs(ref)


def _nearly_flat_bands(r,delta):
    """Three nearly flat bands of spread r delta on the 2x2 mesh, with a
    single resonance, a resonance at 2w and a product of three bands whose
    two energies differ by a constant (so that the prism average is
    exact), at frequencies far from the bands and on them. Returns the
    arguments of a simplex kernel and the sum over the triangles of the
    quadrature of each one"""
    nk = 2
    sims,_ = shgsimplex.simplices(2,nk)
    f = np.array([0.,0.4,1.,0.7]) # a profile on the four mesh points
    terms = np.array([(shgsimplex._SINGLE,0,1,0,0),
                      (shgsimplex._DOUBLE,0,2,0,0),
                      (shgsimplex._PRODUCT,0,1,0,2)],dtype=np.int64)
    W = np.ones((nk*nk,len(terms),1),dtype=np.complex128)
    s = r*delta
    es = np.stack([0.*f,0.5+s*f,1.4+2.*s*f],axis=1) # C - A = 0.2
    ws = np.array([0.,0.3,0.5+0.5*s,0.7+0.5*s,2.])
    ref = np.zeros(len(ws),dtype=np.complex128)
    for sim in sims:
        A = es[sim,1] ; B = es[sim,2]
        def lin(e): return lambda u,v: e[0]+(e[1]-e[0])*u+(e[2]-e[0])*v
        EA,EB = lin(A),lin(B)
        for (i,w) in enumerate(ws):
            z = w+1j*delta
            ref[i] += _quadrature(lambda u,v: 1./(z-EA(u,v))
                    + 1./(2*z-EB(u,v))
                    + 1./((z-EA(u,v))*(2*z-EB(u,v))))
    return (2,nk,es,terms,W,ws,delta,1e-12),ref


def test_direct_sum_on_nearly_flat_bands():
    """Three nearly flat bands on the 2x2 mesh against the quadrature of
    every triangle: the direct sum, which decides its branches per
    frequency, agrees with it to 1e-11 of the largest value over
    frequencies far from the bands and on them. With the rule on delta
    alone it lost 6e-10 to 2e-9. The source kernel keeps that rule and its
    loss, 9e-10 to 5e-9 here (see the comment of its module)"""
    for r in [0.45,1.,3.]:
        args,ref = _nearly_flat_bands(r,1e-6)
        got = shgsimplex._kernels["simplices"](*args)
        assert np.max(np.abs(got[:,0]-ref))<1e-11*np.max(np.abs(ref))


@pytest.mark.parametrize("kernel",["sources","simplices"])
def test_bands_flatter_than_the_broadening(kernel):
    """Bands whose spread is below delta/4, and exactly flat ones: every
    triangle is one cluster of nodes that both kernels do by its Taylor
    series at every frequency, so both give the quadrature value to 1e-11
    of the largest one, on resonance too. The series is around the rounded
    mean of the nodes, for which the sum of the deviations is eps abs(m)
    and not zero, also for equal nodes ((0.7 + 0.7 + 0.7)/3 is not 0.7),
    so its first-order term has to be kept: without it the value on
    resonance is off by eps abs(m)/delta, 3e-11 of it at delta = 1e-6 and
    ten times more at the delta = 1e-7 of this test"""
    delta = 1e-7
    for r in [0.,0.1,0.2]:
        args,ref = _nearly_flat_bands(r,delta)
        got = shgsimplex._kernels[kernel](*args)
        assert np.max(np.abs(got[:,0]-ref))<1e-11*np.max(np.abs(ref))
