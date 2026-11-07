import numpy as np
import pytest

from pyqula import geometry
from pyqula import chi
from pyqula.chitk import chiAB
from pyqula.chitk import simplexchi


def _square(mu=0.1,has_spin=False):
    h = geometry.square_lattice().get_hamiltonian(has_spin=has_spin)
    h.add_onsite(mu)
    return h


def _honeycomb(has_spin=False):
    h = geometry.honeycomb_lattice().get_hamiltonian(has_spin=has_spin)
    h.add_onsite(-0.6)
    h.add_sublattice_imbalance(0.2)
    return h


def _chi(h,**kw):
    kw.setdefault("energies",[0.])
    kw.setdefault("mode","matrix")
    return np.array(chi.chiAB(h,**kw)[1])


def test_tiling_covers_the_occupied_part_of_a_simplex():
    """The sub-simplices of the part e < 0 of a segment, a triangle or a
    tetrahedron have the occupied volume and the barycentric weights of
    the step weights of the simplex density matrix, for random vertex
    values"""
    from pyqula.dmtk.simplexdm import _step_weights
    rng = np.random.default_rng(1)
    for p in (2,3,4):
        for trial in range(30):
            e = rng.normal(size=p)
            out = np.zeros((3,p,p))
            n = simplexchi._clip_neg(np.identity(p),e,p,simplexchi._R,
                                     simplexchi._NR,out,0)
            vols = [abs(simplexchi._det(out[i],p)) for i in range(n)]
            w = sum(v*out[i].mean(axis=0) for i,v in enumerate(vols)) \
                if n>0 else np.zeros(p)
            order = np.argsort(e)
            ref = np.zeros(p)
            frac = _step_weights(e[order],p,0.,ref)
            assert abs(sum(vols)-frac) < 1e-12
            assert np.max(np.abs(w[order]-ref)) < 1e-12


def test_triangle_matches_the_converged_grid_at_high_temperature():
    """At T = 0.1 t the uniform mesh converges exponentially, and the
    triangle integration of the same static and dynamic response has to
    reach it at second order in 1/nk"""
    h = _square()
    kw = dict(q=[0.37,0.21,0.],energies=[0.,1.],delta=0.3,T=0.1,
              mode="trace")
    grid = _chi(h,nk=64,**kw)
    tri = [_chi(h,nk=nk,integration="triangle",**kw) for nk in (24,48,96)]
    assert np.max(np.abs(tri[1]-grid)) < 2e-3*np.max(np.abs(grid))
    # second order: successive differences fall by about 4
    d1 = np.max(np.abs(tri[0]-tri[1]))
    d2 = np.max(np.abs(tri[1]-tri[2]))
    assert d1/d2 > 3.


def test_static_response_at_q0_is_the_density_of_states():
    """The static charge response at q = 0 summed over the sites is minus
    the density of states at the Fermi level (the compressibility), here
    of a spinful honeycomb metal at T = 0, with the DOS of the same
    triangle integration. A small q gives the same limit"""
    h = _honeycomb(has_spin=True)
    nk = 30
    c0 = _chi(h,q=[0.,0.,0.],nk=nk,delta=0.,T=0.,integration="triangle")
    dos = h.get_dos(energies=[0.],nk=nk,delta=0.,integration="triangle",
                    write=False)[1][0]
    assert abs(np.sum(c0[0]).real+dos) < 1e-10*dos
    cq = _chi(h,q=[1e-4,3e-5,0.],nk=nk,delta=0.,T=0.,integration="triangle")
    assert abs(np.sum(cq[0])-np.sum(c0[0])) < 1e-3*dos


def test_spin_response_of_a_paramagnet_is_twice_the_spinless_one():
    """For a spin-degenerate band the response of sz (eigenvalues +-1) is
    twice the charge response of one spin species: the degenerate pairs of
    the two spin copies and the basis eigh picks inside them must not
    matter"""
    kw = dict(q=[0.31,0.12,0.],nk=32,delta=0.,T=1e-3,integration="triangle",
              mode="trace")
    c1 = _chi(_square(),**kw)
    cs = _chi(_square(has_spin=True),A="sz",B="sz",**kw)
    assert abs(cs[0]-2.*c1[0]) < 1e-10


def test_response_is_symmetric_under_q_to_minus_q_and_the_mirror():
    """The triangulation of the mesh is invariant under inversion and
    under the mirror kx <-> ky, so for the square lattice with time
    reversal chi(q) = chi(-q) = chi(qy,qx) holds at any nk, and the four
    fold rotation (qx,qy) -> (-qy,qx), which the triangulation breaks,
    holds to the discretization error"""
    h = _square()
    kw = dict(nk=40,delta=0.,T=1e-3,integration="triangle",mode="trace")
    q = np.array([0.23,0.11,0.])
    c = _chi(h,q=q,**kw)[0]
    assert abs(_chi(h,q=-q,**kw)[0]-c) < 1e-12
    assert abs(_chi(h,q=[q[1],q[0],0.],**kw)[0]-c) < 1e-12
    assert abs(_chi(h,q=[-q[1],q[0],0.],**kw)[0]-c) < 2e-3*abs(c)


def test_critical_U_converges_on_a_coarse_mesh():
    """The Stoner criterion at (pi,pi) of the square lattice slightly away
    from half filling, U_c = -1/chi0(Q) per spin, at T = 1e-3 t, where the
    uniform mesh needs nk of order v/T. The triangle values at nk = 48, 96
    and 192 converge (not monotonously: the line where E(k) = E(k+Q) runs
    along mesh edges through the van Hove points), and the RPA response
    through chi_AB_RPA diverges there: for the spinful model, with the
    vertex -U/2 of sz, chi_RPA/chi0 = 1/(1 - U/U_c)"""
    kw = dict(q=[0.5,0.5,0.],energies=[0.],delta=0.,T=1e-3,
              integration="triangle",mode="trace")
    h = _square()
    uc = [-1./_chi(h,nk=nk,**kw)[0].real for nk in (48,96,192)]
    assert abs(uc[0]-uc[2]) < 3e-3*uc[2]
    assert abs(uc[1]-uc[2]) < 1e-4*uc[2]
    from pyqula.chitk.rpa import chi_AB_RPA
    hs = _square(has_spin=True)
    U = 0.9*uc[0]
    kw.pop("mode")
    c0 = chi_AB_RPA(hs,A="sz",B="sz",nk=48,**kw)[1][0]
    cr = chi_AB_RPA(hs,A="sz",B="sz",V=np.array([[-U/2.]]),nk=48,**kw)[1][0]
    assert abs(cr[0,0]/c0[0,0]-1./(1.-U/uc[0])) < 1e-8


def test_dynamic_response_has_an_exact_zero_broadening_limit():
    """The resolvent is integrated exactly, so delta = 0 is the retarded
    limit: it is close to a tiny delta, and the spectral weight of the
    charge response is positive at positive frequency"""
    h = _honeycomb()
    kw = dict(q=[0.21,0.07,0.],energies=np.linspace(0.05,5.,40),nk=24,T=0.,
              integration="triangle",mode="diagonal")
    c0 = _chi(h,delta=0.,**kw)
    c1 = _chi(h,delta=1e-6,**kw)
    assert np.max(np.abs(c0-c1)) < 1e-4*np.max(np.abs(c0))
    assert np.min(c0.imag) > -1e-12


def test_tetrahedra_in_3d():
    """The q = 0 sum rule and the small-q limit on a cubic metal, which
    goes through the tetrahedron tilings"""
    h = geometry.cubic_lattice().get_hamiltonian(has_spin=False)
    h.add_onsite(0.5)
    nk = 12
    kw = dict(nk=nk,delta=0.,T=0.,integration="triangle",mode="trace")
    c0 = _chi(h,q=[0.,0.,0.],**kw)[0]
    dos = h.get_dos(energies=[0.],nk=nk,delta=0.,integration="triangle",
                    write=False)[1][0]
    assert abs(c0.real+dos) < 1e-10*dos
    cq = _chi(h,q=[1e-4,2e-5,3e-5],**kw)[0]
    assert abs(cq-c0) < 1e-3*dos


def test_integration_names_come_from_the_registry():
    h = _square()
    assert chiAB.get_integration_names()==list(chiAB._integrations)
    with pytest.raises(ValueError) as err:
        _chi(h,q=[0.1,0.,0.],nk=4,integration="tetra")
    for name in chiAB.get_integration_names():
        assert name in str(err.value)
    with pytest.raises(NotImplementedError):
        _chi(h,q=[0.1,0.,0.],nk=4,integration="triangle",imode="adaptive")
    with pytest.raises(ValueError):
        _chi(h,q=[0.1,0.,0.],nk=4,integration="triangle",delta=-1.)


def _kernels_agree(h,rtol=1e-10,**kw):
    """the source kernel against the reference kernel, which evaluates
    every piece at every frequency"""
    out = {}
    for kernel in ("simplices","sources"):
        old = simplexchi.default_kernel
        simplexchi.default_kernel = kernel
        try: out[kernel] = _chi(h,integration="triangle",**kw)
        finally: simplexchi.default_kernel = old
    ref = out["simplices"]
    assert np.max(np.abs(out["sources"]-ref)) < rtol*np.max(np.abs(ref))


def test_source_kernel_matches_the_reference_kernel():
    """The source kernel builds the pieces once and sums every frequency
    over the bins of transition energy, by their moments far from the
    frequency and piece by piece near it. It has to give the reference
    kernel to 1e-10 of the largest value: at a finite and at zero
    broadening, at T = 0, at finite T with delta >= T and with delta < T
    (the resonance corrections), on a spinful model with the degenerate
    pairs of the spin copies, in 1D and in 3D"""
    ws = np.linspace(0.02,6.,40)
    h = _square()
    kw = dict(q=[0.3,0.1,0.],energies=ws,nk=40,mode="trace")
    _kernels_agree(h,delta=1e-3,T=0.,**kw)
    _kernels_agree(h,delta=0.,T=0.,**kw)
    _kernels_agree(h,delta=2e-3,T=2e-3,**kw)
    # delta < T, where the pieces of x that hold the resonance of a
    # frequency are corrected for that frequency, and the static response
    _kernels_agree(h,delta=1e-3,T=1e-2,**kw)
    _kernels_agree(h,delta=0.,T=2e-3,**kw)
    _kernels_agree(h,delta=0.,T=1e-3,q=[0.5,0.5,0.],energies=[0.],nk=40,
                   mode="trace")
    _kernels_agree(h,delta=0.,T=0.,q=[0.5,0.5,0.],energies=[0.],nk=40,
                   mode="trace")
    hs = _honeycomb(has_spin=True)
    kw = dict(q=[0.21,0.07,0.],energies=ws,nk=16,A="sz",B="sz",
              mode="matrix")
    _kernels_agree(hs,delta=1e-3,T=0.,**kw)
    _kernels_agree(hs,delta=5e-3,T=5e-3,**kw)
    _kernels_agree(hs,delta=1e-3,T=5e-3,**kw)
    mem = simplexchi.max_memory_gb # one component per block of the sources
    simplexchi.max_memory_gb = 1e-6
    try: _kernels_agree(hs,delta=1e-3,T=0.,**kw)
    finally: simplexchi.max_memory_gb = mem
    _kernels_agree(hs,delta=1e-2,T=0.,q=[0.,0.,0.],energies=ws,nk=16,
                   mode="diagonal")
    hc = geometry.chain().get_hamiltonian(has_spin=False)
    _kernels_agree(hc,delta=1e-3,T=0.,q=[0.13,0.,0.],energies=ws,nk=200,
                   mode="trace")
    h3 = geometry.cubic_lattice().get_hamiltonian(has_spin=False)
    h3.add_onsite(0.5)
    _kernels_agree(h3,delta=1e-2,T=0.,q=[0.11,0.23,0.05],energies=ws,nk=6,
                   mode="trace")
    _kernels_agree(h3,delta=1e-3,T=2e-2,q=[0.11,0.23,0.05],energies=ws,nk=6,
                   mode="trace")


def test_a_piece_of_x_with_a_fixed_tiling_is_evaluated_once():
    """Where no vertex energy lies inside a piece (a,b) of the integral
    over x and every band keeps one side of x on the whole simplex, the
    T=0 weights do not depend on x, and both kernels replace the graded
    nodes of the piece by one evaluation times f(a) - f(b). That has to
    be the node sum to rounding, for triangles and tetrahedra, at a finite
    and at zero broadening"""
    rng = np.random.default_rng(3)
    T = 1e-3
    for p in (3,4):
        ws,kn = simplexchi._workspace(p,len(simplexchi._tbreaks))
        omegas = np.linspace(-0.05,0.05,7)
        for trial in range(20):
            en = -0.01-0.02*rng.random(p) # every E_n below the piece
            em = 0.03+0.02*rng.random(p) # every E_m above it
            if trial%2: en,em = em,-em-0.04 # region B instead of A
            D = en-em
            a,b = sorted(rng.uniform(-0.005,0.02,2)) if trial%2==0 else \
                  sorted(rng.uniform(-0.02,0.005,2))
            kind = simplexchi._interval_kind(en,em,p,a,b)
            assert kind==(1 if trial%2==0 else 2)
            for delta in (1e-3,0.):
                acc = np.zeros(p,dtype=np.complex128)
                nodes = np.zeros((p,len(omegas)),dtype=np.complex128)
                simplexchi._interval_nodes(en,em,D,p,a,b,omegas,0,len(omegas),
                        delta,T,False,1.,nodes,acc,ws)
                once = np.zeros((p,len(omegas)),dtype=np.complex128)
                sg,f1,span = simplexchi._interval(a,b,T)
                simplexchi._kernel0(en,em,D,p,0.5*(a+b),omegas,0,len(omegas),
                        delta,False,span,once,acc,ws)
                assert np.max(np.abs(once)) > 0.
                assert np.max(np.abs(nodes-once)) < 1e-13*np.max(np.abs(once))


def test_kernel_names_come_from_the_registry():
    h = _square()
    with pytest.raises(ValueError) as err:
        simplexchi.chi_triangle(h,q=[0.1,0.,0.],energies=[0.],nk=4,
                pAs=[np.identity(1)],pBs=[np.identity(1)],kernel="pieces")
    for name in simplexchi._kernels:
        assert name in str(err.value)
