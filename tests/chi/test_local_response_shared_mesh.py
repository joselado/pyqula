"""A q on the mesh of the triangle susceptibility shares the diagonalization
of the k-mesh.

For q with every component times nk an integer, k+q is a point of the
uniform mesh, so the bands and the eigenvectors at k+q are those at k,
reindexed. chi_triangle then diagonalizes the mesh once instead of twice,
and the local response, q=None, diagonalizes it once for all the q of the
mesh. The reference is the two-diagonalization route of a generic q, which
the same code takes when the detection of a mesh q is switched off: both
routes have to give the same response, and the counts of diagonalizations
have to be nk^dim and 2 nk^dim.
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
    nk = 8
    for q in (None,[0.,0.,0.],[1./nk,2./nk,0.],[-3./nk,1.+1./nk,0.]):
        counter[0] = 0
        out = simplexchi.chi_triangle(h,q=q,counted=True,
                pAs=np.identity(2)[None],pBs=np.identity(2)[None],
                energies=[0.],delta=1e-2,T=1e-2,nk=nk,mode="trace")
        assert out[2] == nk**2
        assert counter[0] == nk**2
    counter[0] = 0
    out = simplexchi.chi_triangle(h,q=[0.137,0.211,0.],counted=True,
            pAs=np.identity(2)[None],pBs=np.identity(2)[None],
            energies=[0.],delta=1e-2,T=1e-2,nk=nk,mode="trace")
    assert out[2] == 2*nk**2
    assert counter[0] == 2*nk**2


_cases = [(_square,16,"matrix",False),(_square,16,"trace",False),
          (_square,16,"matrix",True),(_square,16,"trace",True),
          (_honeycomb_supercell,4,"matrix",False),
          (_honeycomb_supercell,4,"trace",False),
          (_honeycomb_supercell,4,"trace",True)]


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


def test_a_generic_q_is_unchanged(monkeypatch):
    h = _honeycomb_supercell()
    kw = dict(energies=np.linspace(-1.,1.,3),delta=1e-2,nk=4,mode="matrix")
    q = [0.137,0.211,0.]
    out = _chi(h,q,**kw)
    _two_diagonalizations(monkeypatch)
    assert np.array_equal(out,_chi(h,q,**kw))


@pytest.mark.parametrize("model,nk,modes",[(_square,6,("trace","matrix")),
                                           (_honeycomb_supercell,4,("trace",))])
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
