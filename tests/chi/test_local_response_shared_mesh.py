"""A q on the mesh of the triangle susceptibility shares the diagonalization
of the k-mesh.

For q with every component times nk an integer, k+q is a point of the
uniform mesh, so the bands and the eigenvectors at k+q are those at k,
reindexed. chi_triangle then diagonalizes the mesh once instead of twice,
and the local response, q=None, diagonalizes it once for all the q of the
mesh. The reference is the two-diagonalization route of a generic q, which
the same code takes when the detection of a mesh q is switched off: both
routes have to give the same response, and the counts of diagonalizations
have to be nk^dim and 2 nk^dim. A generic q next to a mesh q has to give
the response of the mesh q, since the two routes are different code.
"""
import numpy as np
import pytest

from pyqula import geometry
from pyqula.chitk import chiAB
from pyqula.chitk import simplexchi


def _square():
    h = geometry.square_lattice().get_hamiltonian(has_spin=True)
    h.add_onsite(0.4)
    h.add_exchange([0.,0.,0.2]) # no spin degeneracy
    return h


def _honeycomb_supercell():
    g = geometry.honeycomb_lattice().get_supercell(2)
    h = g.get_hamiltonian(has_spin=True) # spin-degenerate multiplets
    h.add_onsite(0.3)
    return h


@pytest.fixture
def counter(monkeypatch):
    """the number of Hamiltonians diagonalized by simplexchi._eigen"""
    count = [0]
    eigen = simplexchi._eigen
    def counted(hkgen,ks,batch,vectors):
        for i0,r in eigen(hkgen,ks,batch,vectors):
            count[0] += len(r[0])
            yield i0,r
    monkeypatch.setattr(simplexchi,"_eigen",counted)
    return count


def _two_diagonalizations(monkeypatch):
    """switch the detection of a mesh q off: every q takes the route of a
    generic q, a diagonalization at k and one at k+q"""
    monkeypatch.setattr(simplexchi,"mesh_q_tol",-1.)


def _chi(h,q,**kw):
    return np.array(chiAB.chiAB_q(h,q=q,integration="triangle",**kw)[1])


def _rel(a,b):
    return np.max(np.abs(a-b))/np.max(np.abs(b))


def test_mesh_shift_is_the_mesh_point_of_k_plus_q():
    from pyqula.dostk.simplexdos import uniform_mesh
    for dim,nk in ((1,7),(2,6),(3,4)):
        ks = uniform_mesh(dim,nk)
        for s in ([0,0,0],[1,0,0],[2,-3,1],[-1,5,2]):
            q = np.zeros(3); q[:dim] = np.array(s[:dim])/nk
            perm = simplexchi.mesh_shift(q,dim,nk)
            d = (ks+q-ks[perm]+0.5)%1.-0.5
            assert np.max(np.abs(d)) < 1e-12
        assert simplexchi.mesh_shift(None,dim,nk) is not None
        q = np.zeros(3); q[0] = 0.37/nk
        assert simplexchi.mesh_shift(q,dim,nk) is None
    # a q along a direction the lattice does not have is not a mesh q
    assert simplexchi.mesh_shift([0.,0.,0.25],2,4) is None


def test_a_mesh_q_is_diagonalized_once(counter):
    h = _square()
    nk = 4
    for q in (None,[0.,0.,0.],[1./nk,2./nk,0.],[-3./nk,1.+1./nk,0.]):
        counter[0] = 0
        out = simplexchi.chi_triangle(h,q=q,counted=True,
                pAs=np.identity(2)[None],pBs=np.identity(2)[None],
                energies=[0.],delta=1e-2,T=0.,nk=nk,mode="trace")
        assert out[2] == nk**2
        assert counter[0] == nk**2
    counter[0] = 0
    out = simplexchi.chi_triangle(h,q=[0.137,0.211,0.],counted=True,
            pAs=np.identity(2)[None],pBs=np.identity(2)[None],
            energies=[0.],delta=1e-2,T=0.,nk=nk,mode="trace")
    assert out[2] == 2*nk**2
    assert counter[0] == 2*nk**2


_cases = [(_square,16,"matrix",False),(_square,16,"trace",False),
          (_square,16,"matrix",True),(_square,16,"trace",True),
          (_honeycomb_supercell,4,"trace",False)]


@pytest.mark.parametrize("model,nk,mode,lowmem",_cases)
def test_a_mesh_q_gives_the_two_diagonalization_response(model,nk,mode,
        lowmem,monkeypatch):
    """static and dynamic, at q = None and two mesh q, with the
    eigenvectors kept and, lowmem, recomputed at every pass"""
    h = model()
    if lowmem: monkeypatch.setattr(simplexchi,"max_memory_gb",1e-9)
    qs = (None,[1./nk,2./nk,0.],[-3./nk,1.+1./nk,0.])
    runs = (dict(energies=[0.,0.5],delta=1e-2),
            dict(energies=[0.,0.5],delta=0.,T=0.))
    new = [[_chi(h,q,nk=nk,mode=mode,**kw) for kw in runs] for q in qs]
    _two_diagonalizations(monkeypatch)
    for q,outs in zip(qs,new):
        for kw,out in zip(runs,outs):
            ref = _chi(h,q,nk=nk,mode=mode,**kw)
            assert np.max(np.abs(ref)) > 1e-3
            assert _rel(out,ref) < 1e-12


@pytest.mark.parametrize("model,nk",[(_square,8),(_honeycomb_supercell,4)])
def test_the_generic_route_joins_the_shared_route_at_a_mesh_q(model,nk,
        counter):
    """a q off a mesh q by 1e-9 is a generic q, two diagonalizations
    of the mesh, and the response moves by an amount of order 1e-9 from
    the one of the mesh q on the shared route: the two routes are two
    pieces of code, and they meet at the mesh q"""
    h = model()
    kw = dict(energies=[0.,0.5],delta=1e-2,nk=nk,mode="matrix")
    qm = np.array([1./nk,2./nk,0.])
    eps = 1e-9
    qg = qm + eps*np.array([1.,-0.7,0.])
    assert simplexchi.mesh_shift(qm,2,nk) is not None
    assert simplexchi.mesh_shift(qg,2,nk) is None
    counter[0] = 0
    mesh = _chi(h,qm,**kw)
    assert counter[0] == nk**2
    counter[0] = 0
    generic = _chi(h,qg,**kw)
    assert counter[0] == 2*nk**2
    assert np.max(np.abs(mesh)) > 1e-3
    assert _rel(generic,mesh) < 1e-6
    # a q within the snap of mesh_q_tol, q*nk an integer to 1e-12, is
    # evaluated as the mesh q itself
    qs = qm + 1e-14*np.array([1.,1.,0.])
    assert simplexchi.mesh_shift(qs,2,nk) is not None
    assert np.array_equal(_chi(h,qs,**kw),mesh)


def _chain():
    h = geometry.chain().get_hamiltonian(has_spin=True)
    h.add_onsite(0.3)
    h.add_exchange([0.,0.,0.2]) # no spin degeneracy
    return h


def _cubic():
    h = geometry.cubic_lattice().get_hamiltonian(has_spin=True)
    h.add_onsite(0.5)
    h.add_exchange([0.3,0.,0.2])
    return h


@pytest.mark.parametrize("model,dim,nk,mode",[(_chain,1,24,"matrix"),
                                              (_chain,1,24,"trace"),
                                              (_cubic,3,5,"trace")])
def test_1d_and_3d_shared_route_against_two_diagonalizations(model,dim,nk,
        mode,counter,monkeypatch):
    """a chain and a cubic metal: a mesh q, q = None, and the local
    response, static and dynamic, against the two-diagonalization route.
    At T = 0, since the finite-T tetrahedra cost about 8 s per q at nk=5"""
    h = model()
    qm = np.zeros(3); qm[:dim] = np.array([2.,-1.,1.])[:dim]/nk
    kw = dict(energies=[0.,0.5],delta=1e-2,T=0.,nk=nk,mode=mode)
    new = [_chi(h,q,**kw) for q in (qm,None)]
    counter[0] = 0
    es,local = chiAB.chiAB(h,q=None,integration="triangle",**kw)
    assert counter[0] == nk**dim
    _two_diagonalizations(monkeypatch)
    for q,out in zip((qm,None),new):
        ref = _chi(h,q,**kw)
        assert np.max(np.abs(ref)) > 1e-3
        assert _rel(out,ref) < 1e-12
    qs = h.geometry.get_kmesh(nk=nk)
    ref = np.mean([_chi(h,q,**kw) for q in qs],axis=0)
    assert _rel(np.array(local),ref) < 1e-12


@pytest.mark.parametrize("model,nk,modes",[(_square,4,("trace","matrix"))])
def test_the_local_response_diagonalizes_the_mesh_once(model,nk,modes,
        counter,monkeypatch):
    """chiAB with q=None, the average over the q-mesh, against the loop
    over the q of the mesh on the two-diagonalization route"""
    h = model()
    kw = dict(energies=[0.,0.5],delta=1e-2,nk=nk,integration="triangle")
    for mode in modes:
        counter[0] = 0
        es,local = chiAB.chiAB(h,q=None,mode=mode,**kw)
        assert counter[0] == nk**2
        with monkeypatch.context() as m:
            m.setattr(simplexchi,"mesh_q_tol",-1.)
            qs = h.geometry.get_kmesh(nk=nk)
            ref = np.mean([chiAB.chiAB_q(h,q=q,mode=mode,**kw)[1]
                           for q in qs],axis=0)
        assert _rel(np.array(local),ref) < 1e-12


def test_the_shared_mesh_is_refused_where_it_does_not_apply():
    h = _square()
    eigen = simplexchi.MeshEigen(h,8)
    with pytest.raises(NotImplementedError,match="integration='triangle'"):
        chiAB.chiAB_q(h,q=[0.,0.,0.],nk=8,energies=[0.],eigen=eigen)
    with pytest.raises(ValueError,match="nk=6"):
        chiAB.chiAB_q(h,q=[0.,0.,0.],nk=6,energies=[0.],eigen=eigen,
                      integration="triangle")


@pytest.mark.parametrize("q",[[0.,0.,0.],[1./8,2./8,0.],[0.137,0.211,0.]])
def test_the_shared_mesh_of_another_hamiltonian_is_refused(q):
    """a MeshEigen of another Hamiltonian with the same nk, dimension and
    number of bands, of a copy of h, or of h changed in place after the
    diagonalization, would give the bands of the wrong Hamiltonian"""
    kw = dict(nk=8,energies=[0.],delta=1e-2,integration="triangle")
    h = _square()
    other = _square()
    other.add_onsite(0.7) # same shape, other bands
    eigen = simplexchi.MeshEigen(other,8)
    with pytest.raises(ValueError,match="Hamiltonian of this call"):
        chiAB.chiAB_q(h,q=q,eigen=eigen,**kw)
    with pytest.raises(ValueError,match="Hamiltonian of this call"):
        chiAB.chiAB_q(h,q=q,eigen=simplexchi.MeshEigen(h.copy(),8),**kw)
    eigen = simplexchi.MeshEigen(h,8)
    chiAB.chiAB_q(h,q=[0.,0.,0.],eigen=eigen,**kw) # diagonalizes the mesh
    h.add_onsite(0.1) # in place, after the diagonalization
    with pytest.raises(ValueError,match="changed in place"):
        chiAB.chiAB_q(h,q=q,eigen=eigen,**kw)


@pytest.mark.parametrize("route",["AB","ops"])
def test_the_local_rpa_response_diagonalizes_the_mesh_once(route,counter,
        monkeypatch):
    """the local dressed response, every q of the mesh dressed on its own
    and averaged, takes one diagonalization of the mesh for all its q, and
    gives the two-diagonalization route"""
    from pyqula.chitk import rpa
    h = _square()
    nk = 4
    kw = dict(energies=[0.,0.5],delta=1e-2,nk=nk,integration="triangle")
    if route=="AB":
        run = lambda: rpa.chi_AB_RPA(h,V=np.array([[0.8]]),q=None,**kw)
    else:
        run = lambda: rpa.chi_ops_RPA(h,ops=["sx","sy","sz"],
                V=0.8*np.identity(3),q=None,**kw)
    counter[0] = 0
    es,out = run()
    assert counter[0] == nk**2
    _two_diagonalizations(monkeypatch)
    counter[0] = 0
    es,ref = run()
    assert counter[0] == 2*nk**4
    assert np.max(np.abs(ref)) > 1e-3
    assert _rel(np.array(out),np.array(ref)) < 1e-12
