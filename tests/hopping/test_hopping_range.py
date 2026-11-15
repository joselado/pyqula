"""The range of a hopping given as a function or a HoppingGenerator.

A hopping function on a 1D or 2D lattice used to be stored in the
single-cell form, which keeps only the neighboring cells: on the triangular
lattice 4 of the 6 second-neighbor bonds were lost, and on the chain of one
site the second neighbor. The same shells given as a list were complete.
Function, generator and list must give the same Hamiltonian
(future_development/hopping_range.md).
"""

import numpy as np
import pytest

from pyqula import geometry
from pyqula.specialhopping import NNG


def shells(ds,ts):
    """A hopping function with the value ts[i] at the distance ds[i]"""
    def f(r1,r2):
        d = np.linalg.norm(np.array(r1)-np.array(r2))
        return sum(t*float(abs(d-di)<1e-3) for di,t in zip(ds,ts))
    return f


def bands(h,k):
    return np.linalg.eigvalsh(np.array(h.get_dense().get_hk_gen()(k)))


CASES = [("triangular",[1.,np.sqrt(3.)]), # second shell two cells away
         ("chain",[1.,2.,3.]), # the third neighbor three cells away
         ("square",[1.,np.sqrt(2.),2.]), # the third shell at (2,0)
         ("honeycomb",[1.,np.sqrt(3.),2.,np.sqrt(7.)])] # the fourth shell


@pytest.mark.parametrize("name,ds",CASES)
def test_function_and_generator_give_the_shells_of_a_list(name,ds):
    g = getattr(geometry,{"triangular":"triangular_lattice",
        "chain":"chain","square":"square_lattice",
        "honeycomb":"honeycomb_lattice"}[name])()
    nd = g.neighbor_distances()[0:len(ds)]
    assert np.allclose(nd,ds,atol=1e-6) # the shells of the list
    ts = [1.,0.3,-0.2,0.1][0:len(ds)]
    hl = g.get_hamiltonian(has_spin=False,tij=ts)
    hf = g.get_hamiltonian(has_spin=False,tij=shells(ds,ts))
    hg = g.get_hamiltonian(has_spin=False,tij=NNG(g,ts))
    for k in [np.array([0.13,0.29,0.]),np.array([0.41,0.07,0.])]:
        e = bands(hl,k)
        assert np.max(np.abs(bands(hf,k)-e))<1e-10
        assert np.max(np.abs(bands(hg,k)-e))<1e-10


def test_a_large_cell_still_builds_its_neighbors_only():
    """On a cell larger than the range, the function reaches the
    neighboring cells only, as before"""
    g = geometry.square_lattice().get_supercell(8)
    h = g.get_hamiltonian(has_spin=False,tij=shells([1.,np.sqrt(2.)],[1.,0.3]))
    dirs = [tuple(t.dir) for t in h.hopping]
    assert max(max(abs(x) for x in d) for d in dirs)==1
    h0 = geometry.square_lattice().get_hamiltonian(has_spin=False,tij=[1.,0.3])
    k = np.array([0.,0.,0.])
    e8 = bands(h,k)
    e0 = np.sort(np.concatenate([bands(h0,np.array([i/8,j/8,0.]))
        for i in range(8) for j in range(8)]))
    assert np.max(np.abs(e8-e0))<1e-10


def test_a_function_and_a_generator_reach_the_distance_rcut():
    """A function and a generator reach every pair within rcut=5.0; rcut=
    moves it, and cutoff= (or nc= for a generator) gives a box of cells
    instead"""
    g = geometry.chain()
    ts = [1.,0.3,-0.2,0.1,0.05,0.02] # the sixth neighbor at a distance 6
    f = shells([1.,2.,3.,4.,5.,6.],ts)
    def hg(**kw): return g.get_hamiltonian(has_spin=False,tij=NNG(g,ts),**kw)
    def hf(**kw): return g.get_hamiltonian(has_spin=False,tij=f,**kw)
    h5 = g.get_hamiltonian(has_spin=False,tij=ts[0:5])
    h6 = g.get_hamiltonian(has_spin=False,tij=ts)
    k = np.array([0.13,0.,0.])
    for hi,href in [(hg(),h5),(hg(rcut=6.5),h6),(hg(cutoff=6),h6),
            (hg(nc=6),h6),(hf(rcut=6.5),h6),(hf(cutoff=6,rcut=6.5),h6)]:
        assert np.max(np.abs(bands(hi,k)-bands(href,k)))<1e-10
    with pytest.raises(TypeError,match="nc"):
        hg(nc=6,cutoff=6)


def test_the_reverse_cell_is_the_adjoint():
    """A Hermitian Hamiltonian from a function stores the hopping to -R as
    the adjoint of the hopping to R, also for a function that is not
    periodic in the lattice, which a build of both directions would turn
    into a Bloch Hamiltonian that is not Hermitian"""
    g = geometry.triangular_lattice()
    def f(r1,r2): # reciprocal, but it depends on the absolute position
        d = np.linalg.norm(r1-r2)
        if abs(d-1.)<1e-3: return 1.+0.5*np.tanh((r1[0]+r2[0])/2.)
        return 0.
    for kw in [dict(),dict(is_sparse=True)]:
        h = g.get_hamiltonian(has_spin=False,tij=f,**kw)
        ms = {tuple(int(x) for x in t.dir):t.m for t in h.hopping}
        assert len(ms)>0
        for d,m in ms.items():
            mr = ms[tuple(-x for x in d)]
            assert np.max(np.abs(np.array(mr.todense() if kw else mr)
                - np.conjugate(np.array(m.todense() if kw else m)).T))==0.
        hk = h.get_hk_gen()([0.13,0.29,0.])
        hk = np.array(hk.todense() if kw else hk)
        assert np.max(np.abs(hk-np.conjugate(hk).T))<1e-12


def test_the_heavy_fermion_chain_is_the_single_cell_model():
    """The one-dimensional topological heavy fermion takes the sign of a
    hopping from the absolute position of the bond, so it is defined by
    the hoppings of the cell 0 to the cell +1 and their adjoint"""
    from pyqula import specialhamiltonian
    h = specialhamiltonian.topological_heavy_fermion_1d()
    g = h.geometry
    r = np.array(g.r)
    def tij(ri,rj): # the function of the model, copied
        r0 = (ri+rj)/2. ; dr = ri-rj ; dr2 = dr.dot(dr)
        if 0.9<dr2<1.1:
            if r0[1]<-0.2: return 1.0
            if r0[1]>0.2: return 0.2
        if 1.9<dr2<2.1 and ri[1]*rj[1]<0.:
            return 0.3 if r0[0]>0. else -0.3
        return 0.
    a1 = np.array(g.a1)
    intra = np.array([[tij(ri,rj) for rj in r] for ri in r])
    inter = np.array([[tij(ri,rj+a1) for rj in r] for ri in r])
    for k in [0.,0.13,0.37]:
        ph = np.exp(2j*np.pi*k)
        m = intra + ph*inter + np.conjugate(ph*inter).T
        hk = np.array(h.get_hk_gen()([k,0.,0.]))
        assert np.max(np.abs(hk-np.conjugate(hk).T))<1e-12
        assert np.max(np.abs(np.linalg.eigvalsh(hk)
            -np.linalg.eigvalsh(m)))<1e-10


def test_a_spinful_generator_is_not_doubled():
    """A matrix generator that returns the spin blocks itself
    (spinful_generator=True) gives a spinful Hamiltonian of its own size,
    on a chain: a Rashba-like hopping I + 0.3i sigma_y dx, with the bands
    2 cos(q) +- 0.6 sin(q)"""
    g = geometry.chain()
    sy = np.array([[0.,-1j],[1j,0.]])
    def fm(rs1,rs2):
        m = np.zeros((2*len(rs1),2*len(rs2)),dtype=np.complex128)
        for i,r1 in enumerate(rs1):
            for j,r2 in enumerate(rs2):
                dr = np.array(r2)-np.array(r1)
                if abs(np.linalg.norm(dr)-1.)<1e-3:
                    m[2*i:2*i+2,2*j:2*j+2] = np.identity(2)+0.3j*dr[0]*sy
        return m
    h = g.get_hamiltonian(mgenerator=fm,spinful_generator=True)
    assert h.has_spin and np.array(h.intra).shape==(2,2)
    for k in [0.,0.13,0.37]:
        q = 2.*np.pi*k
        e = np.linalg.eigvalsh(np.array(h.get_hk_gen()([k,0.,0.])))
        exact = np.sort([2.*np.cos(q)+0.6*np.sin(q),
            2.*np.cos(q)-0.6*np.sin(q)])
        assert np.max(np.abs(e-exact))<1e-10


def test_a_ribbon_from_a_function_takes_a_peierls_phase():
    """A ribbon from a hopping function is stored as multicell, and takes
    the same Peierls phase as the ribbon of the first-neighbor default,
    dense or sparse, spinless or spinful"""
    def f(r1,r2):
        return 1. if abs(np.linalg.norm(r1-r2)-1.)<1e-3 else 0.
    g = geometry.honeycomb_zigzag_ribbon(4)
    for kw in [dict(has_spin=False),dict(has_spin=True),
            dict(has_spin=False,is_sparse=True),
            dict(has_spin=True,is_sparse=True)]:
        h0 = g.get_hamiltonian(**kw) ; h0.add_peierls(0.03)
        h1 = g.get_hamiltonian(tij=f,**kw) ; h1.add_peierls(0.03)
        assert h1.is_multicell
        for k in [0.,0.13,0.37]:
            e0 = bands(h0,np.array([k,0.,0.]))
            e1 = bands(h1,np.array([k,0.,0.]))
            assert np.max(np.abs(e0-e1))<1e-10


def _coordination(g,d,nmax=6):
    """Number of sites at the distance d of each site of g, counted by
    brute force over the cells |n_k| <= nmax"""
    r = np.array(g.r)
    out = np.zeros(len(r),dtype=int)
    for n1 in range(-nmax,nmax+1):
        for n2 in range(-nmax,nmax+1):
            R = n1*np.array(g.a1)+n2*np.array(g.a2)
            dist = np.linalg.norm(r[None,:,:]+R-r[:,None,:],axis=2)
            out += np.sum(np.abs(dist-d)<1e-4,axis=1)
    return out


def _bonds(h):
    """Number of nonzero hoppings of each site, over every cell"""
    ms = [np.array(h.intra)]+[np.array(t.m) for t in h.hopping]
    return np.sum([np.sum(np.abs(m)>1e-12,axis=1) for m in ms],axis=0)


def test_a_shell_beyond_rcut_is_cut_whole():
    """rcut is a cut on pairs: on the honeycomb lattice the shell at
    sqrt(27), just beyond rcut=5.0, is dropped whole by default, where
    selecting cells kept a part of it, and kept whole with rcut=5.3"""
    g = geometry.honeycomb_lattice()
    d = np.sqrt(27.)
    f = shells([1.,d],[1.,0.1])
    z1,zd = _coordination(g,1.),_coordination(g,d)
    assert np.all(zd>0)
    h = g.get_hamiltonian(has_spin=False,tij=f)
    assert np.all(_bonds(h)==z1)
    h = g.get_hamiltonian(has_spin=False,tij=f,rcut=5.3)
    assert np.all(_bonds(h)==z1+zd)
    hg = g.get_hamiltonian(has_spin=False,tij=NNG(g,[0.]*10+[1.]),rcut=5.3)
    assert np.all(_bonds(hg)==_coordination(g,g.neighbor_distances(n=11)[10]))


def test_rcut_with_a_list_of_shells_raises():
    """A list of shells reaches its last shell, so rcut with it is two
    ranges for one hopping"""
    g = geometry.chain()
    with pytest.raises(TypeError,match="rcut"):
        g.get_hamiltonian(tij=[1.,0.2],rcut=3.)


def test_a_finite_system_keeps_every_pair():
    """In a finite system the default keeps every pair, also beyond 5.0,
    and rcut= cuts there too when given"""
    g = geometry.chain().get_supercell(12)
    g.set_finite()
    f = shells([1.,8.],[1.,0.3])
    h = g.get_hamiltonian(has_spin=False,tij=f)
    m = np.array(h.intra)
    assert abs(m[0,8]-0.3)<1e-12 and abs(m[8,0]-0.3)<1e-12
    m = np.array(g.get_hamiltonian(has_spin=False,tij=f,rcut=5.).intra)
    assert m[0,8]==0. and abs(m[0,1]-1.)<1e-12


@pytest.mark.parametrize("name,z",[("square_lattice",4),("honeycomb_lattice",3),
        ("triangular_lattice",6)])
def test_the_default_hopping_of_a_sheared_cell_has_every_neighbor(name,z):
    """The default first-neighbor hopping keeps the single-cell storage on
    a reduced cell, and on the sheared cell a2' = a2 + 2 a1 of the same
    lattice, where a first neighbor sits outside the 9 cells, it is built
    as multicell with every neighbor: z per site, and the spectrum at
    Gamma of the reduced cell"""
    g = getattr(geometry,name)()
    h0 = g.get_hamiltonian(has_spin=False)
    assert not h0.is_multicell
    gs = g.get_supercell([[1,0,0],[2,1,0],[0,0,1]]) # the same lattice
    hs = gs.get_hamiltonian(has_spin=False)
    assert hs.is_multicell
    assert np.all(_bonds(hs)==z)
    k = np.zeros(3)
    assert np.max(np.abs(bands(hs,k)-bands(h0,k)))<1e-10
