"""The window (emin,emax) of the ultracell and its frozen states.

The states of the basis below emin are frozen: they are full, they are not
solved, and their density matrix is that of the unit cell tiled on the
supercell. Where the frozen states are eigenstates of the supercell, a band
decoupled from the modulation, the ultracell with a window is the
supercell, for the density matrix, the Fermi level, the band energy, the
expectation values and the mean field. Where they are not, the window is a
variational restriction: a larger window never raises the energy, and no
window goes below the supercell.
"""

import numpy as np
import pytest
from scipy.linalg import eigh
from scipy.sparse import coo_matrix, csr_matrix

from pyqula import geometry
from pyqula.klist import kmesh
from pyqula.ultracell import Ultracell
from pyqula.ultracelltk.basis import FoldedBasis, dense
from pyqula.ultracelltk.meanfield import (SparseBackend, MeshStates,
        frozen_entries)


def fractional(g):
    B = np.linalg.inv(np.array([g.a1,g.a2,g.a3]).T)
    return lambda r: B@np.array(r)


def same_state(u2,h3,k=(0.21,0.33,0.)):
    m2 = dense(u2.hs.get_hk_gen()(np.array(k)))
    m3 = dense(h3.get_hk_gen()(np.array(k)))
    return np.max(np.abs(m2-m3))


def all_pairs(n,ds):
    """Every entry of an n x n matrix in each direction of ds"""
    r,c = [x.ravel() for x in np.indices((n,n))]
    return {d: (r,c) for d in ds}


def two_legs(spin=False,lower=-6.):
    """Two chains along x, 8 apart, so that the first-neighbor hopping
    makes no rung between them, the lower one shifted by lower"""
    g = geometry.ladder()
    g.r = np.array(g.r)*np.array([1.,8.,1.])
    g.r2xyz()
    h = g.get_hamiltonian(has_spin=spin)
    h.add_onsite(lambda r: lower if r[1]<0. else 0.)
    return h


def upper_wave(n,v=0.4):
    """A potential on the upper leg only, periodic on n cells"""
    def f(r):
        if r[1]<0.: return 0.
        x = 2*np.pi*r[0]/n
        return v*np.cos(x+0.3) + 0.5*v*np.sin(2*x)
    return f


def modulated_cell():
    """Spinful graphene with Rashba coupling on a 3x2 cell, under an
    onsite potential and an exchange field periodic on the cell"""
    g = geometry.honeycomb_lattice()
    h = g.get_hamiltonian()
    h.add_rashba(0.2)
    hs = h.get_supercell([3,2])
    x = fractional(hs.geometry)
    hs.add_onsite(lambda r: 0.3*np.cos(2*np.pi*x(r)[0]))
    hs.add_exchange(lambda r: [0.2*np.sin(2*np.pi*x(r)[0]),0.,0.1])
    return h,hs


def test_window_arguments():
    h,hs = modulated_cell()
    u = Ultracell(h,hs,window=(-2.,None))
    assert u.window==(-2.,None) and u.emin==-2. and u.emax is None
    u = Ultracell(h,hs,emax=1.)
    assert u.window==(None,1.) and u.emin is None and u.emax==1.
    with pytest.raises(ValueError): # both at once
        Ultracell(h,hs,emax=1.,window=(-1.,1.))
    with pytest.raises(ValueError): # an empty window
        Ultracell(h,hs,window=(1.,-1.))
    with pytest.raises(TypeError):
        Ultracell(h,hs,window=("low",1.))
    with pytest.raises(TypeError): # not a pair
        Ultracell(h,hs,window=0.5)


def test_frozen_entries_are_the_frozen_basis_vectors():
    """The frozen density matrix, the unit-cell one tiled on the supercell,
    is the sum over the frozen basis vectors of every kS of the mesh with
    the Bloch phase of the direction, at D and -D, on the Kekule cell of
    graphene, whose diagonal form has P and Q not the identity"""
    g = geometry.honeycomb_lattice()
    gs = g.get_supercell([[2,1,0],[-1,1,0],[0,0,1]],store_primal=True)
    h = g.get_hamiltonian()
    h.add_rashba(0.3)
    hs = gs.get_hamiltonian()
    hs.add_rashba(0.3)
    b = FoldedBasis(h,hs,window=(-1.,None))
    assert np.any(b.P!=np.identity(3)) or np.any(b.Q!=np.identity(3))
    ks = kmesh(2,nk=2)
    n = hs.intra.shape[0]
    ds = [(0,0,0),(1,0,0),(-1,0,0),(1,-1,0),(-1,1,0)]
    frozen = [b.states(k,frozen=True)[4] for k in ks]
    fz = frozen_entries(b,ks,frozen,all_pairs(n,ds))
    for d in ds:
        ref = np.zeros((n,n),dtype=np.complex128)
        for k,Uf in zip(ks,frozen):
            B = np.hstack([b.basis_block(k,Uf,iq) for iq in range(b.N)])
            ref += np.exp(2j*np.pi*np.dot(k,d))*(np.conjugate(B)@B.T)
        ref /= len(ks)
        assert np.max(np.abs(fz[d].reshape(n,n)-ref))<1e-12
    m = fz[(1,0,0)].reshape(n,n) # the opposite direction is the adjoint
    assert np.max(np.abs(fz[(-1,0,0)].reshape(n,n)-m.conj().T))<1e-12
    assert np.max(np.abs(m))>1e-2 # bonds between supercells
    assert np.max(np.abs(np.diag(fz[(0,0,0)].reshape(n,n))))>0.1


@pytest.mark.parametrize("rebuild",[False,True])
def test_frozen_band_of_decoupled_legs_is_the_supercell(rebuild):
    """The lower leg lies below emin and is frozen, and the upper one, which
    carries the modulation, is kept whole. The frozen states are then
    eigenstates of the supercell, so the density matrix at every entry and
    direction, the Fermi level and the band energy are those of "ed" on
    the supercell"""
    n,nk,filling = 6,2,17./24
    h = two_legs()
    hs = h.get_supercell(n)
    hs.add_onsite(upper_wave(n))
    hs.ultracell_basis = FoldedBasis(h,hs,window=(-3.,None))
    hs.ultracell_basis.rebuild = rebuild
    norb = hs.intra.shape[0]
    ds = [(0,0,0),(1,0,0),(-1,0,0)]
    b = SparseBackend(nk=nk,T=1e-7)
    dm,fermi = b.dm_with_fermi(hs,all_pairs(norb,ds),filling)
    assert b.states.nfrozen==nk*n # the whole lower leg
    h3 = hs.copy() ; h3.turn_dense()
    fermi3 = h3.get_fermi4filling(filling,nk=nk,T=1e-7)
    assert abs(fermi-fermi3)<1e-8
    h3.shift_fermi(-fermi3)
    eds = h3.get_density_matrix(ds=ds,nk=nk,T=1e-7)
    for d in ds:
        assert np.max(np.abs(dm[d].toarray()-dense(eds[d])))<1e-8
    hs.shift_fermi(-fermi) # the backend kept the states of hs
    assert abs(b.band_energy(hs)-h3.get_total_energy(nk=nk))<1e-8


@pytest.mark.parametrize("rebuild",[False,True])
def test_band_energy_is_the_trace_with_the_density_matrix(rebuild):
    """With frozen states, kept ones and dropped ones, the band energy is
    sum_D sum_ij H_D[i,j] dm[D][i,j] with the density matrix the backend
    returns, Tr[rho H] for rho the frozen projector plus the occupied
    ultracell states, for the clean basis and the rebuilt one; a backend
    that solves the shifted Hamiltonian again gives the same energy"""
    h,hs = modulated_cell()
    hs.turn_sparse()
    hs.ultracell_basis = FoldedBasis(h,hs,window=(-2.,1.))
    hs.ultracell_basis.rebuild = rebuild
    n = hs.intra.shape[0]
    needed = {}
    for d,m in hs.get_multihopping().get_dict().items():
        m = coo_matrix(m)
        r,c = m.row.astype(np.int64),m.col.astype(np.int64)
        if d==(0,0,0): # the shift by the Fermi level is on the diagonal
            r = np.concatenate([r,np.arange(n)])
            c = np.concatenate([c,np.arange(n)])
        key = np.unique(r*n+c) # each entry once
        needed[tuple(d)] = (key//n,key%n)
    b = SparseBackend(nk=2,T=1e-7)
    dm,fermi = b.dm_with_fermi(hs,needed,0.5)
    states = b.states
    assert states.nfrozen>0 and sum(len(e) for e in states.es)>0
    assert sum(len(e) for e in states.es)+states.nfrozen<n*len(states.ks)
    hs.shift_fermi(-fermi)
    e1 = b.band_energy(hs)
    e2 = 0.
    for d,m in hs.get_multihopping().get_dict().items():
        m = coo_matrix(m)
        x = dm[tuple(d)].toarray()[m.row,m.col]
        e2 += np.real(np.sum(m.data*x))
    assert abs(e1-e2)<1e-8
    h2 = hs.copy() # solved again, from the shifted Hamiltonian
    assert abs(SparseBackend(nk=2,T=1e-7).band_energy(h2)-e1)<1e-10


def test_window_holding_every_state_is_the_supercell():
    """A window below and above every level keeps every state and freezes
    none, so the ultracell is the supercell"""
    h,hs = modulated_cell()
    kS = [0.231,0.417,0.]
    e2 = eigh(dense(hs.get_hk_gen()(kS)),eigvals_only=True)
    for window in [(-10.,10.),(-10.,None),(None,10.)]:
        u = Ultracell(h,hs,window=window)
        e1 = eigh(u.get_hk_gen()(kS),eigvals_only=True)
        assert np.max(np.abs(e1-e2))<1e-12
        assert np.sum(u.basis.states(kS,frozen=True)[5])==0 # none frozen


def test_nested_windows_are_variational():
    """At a fixed filling the one-shot band energy of the ultracell is that
    of a Slater determinant in a restricted space: the frozen states full
    and the lowest ultracell states of the kept ones. A larger window
    holds the state of a smaller one, so the energy never increases when
    emin goes down or emax up, and it is never below the supercell"""
    g = geometry.honeycomb_lattice()
    h = g.get_hamiltonian(has_spin=False)
    hs = h.get_supercell([3,2])
    x = fractional(hs.geometry)
    hs.add_onsite(lambda r: 0.4*np.cos(2*np.pi*x(r)[0]+0.2)
            + 0.2*np.sin(2*np.pi*(x(r)[0]+x(r)[1])))
    n,nk,filling = hs.intra.shape[0],2,0.5
    h3 = hs.copy() ; h3.turn_dense()
    fermi3 = h3.get_fermi4filling(filling,nk=nk,T=1e-7)
    h3.shift_fermi(-fermi3)
    e0 = h3.get_total_energy(nk=nk) + fermi3*n*filling # the supercell
    es = []
    for window in [(-2.5,1.2),(-2.5,2.),(-2.8,2.),(None,2.),(None,2.5)]:
        hc = hs.copy()
        hc.ultracell_basis = FoldedBasis(h,hc,window=window)
        b = SparseBackend(nk=nk,T=1e-7)
        fermi = b.fermi(b.solve(hc),filling)
        hc.shift_fermi(-fermi) # the backend kept the states of hc
        es.append(b.band_energy(hc) + fermi*n*filling)
    assert es[0]>e0+1e-6 # the smallest window misses something
    for e in es: assert e>=e0-1e-10
    for e1,e2 in zip(es[0:-1],es[1:]): assert e2<=e1+1e-10


def test_expectation_values_with_frozen_states():
    """The lower leg carries a strong exchange field along a generic
    direction, and only its lower spin band is below emin, so the frozen
    states carry a moment; the upper leg carries a modulated exchange
    field and is kept whole. The expectation values per site are those
    of the supercell for the occupation and every spin component, sy
    included, whose sign the transpose convention of dm decides"""
    n = 6
    h = two_legs(spin=True,lower=-3.1) # no level of the mesh at zero
    m = 4.*np.array([0.3,-0.5,0.6])/np.sqrt(0.7)
    h.add_exchange(lambda r: m if r[1]<0. else np.zeros(3))
    h.add_onsite(lambda r: 0.13 if r[1]>0. else 0.)
    hs = h.get_supercell(n)
    hs.add_onsite(upper_wave(n,v=0.3))
    hs.add_exchange(lambda r: [0.3*np.cos(2*np.pi*r[0]/n),
        0.2*np.sin(2*np.pi*r[0]/n),0.1] if r[1]>0. else [0.,0.,0.])
    for rebuild in [False,True]:
        u = Ultracell(h,hs,window=(-4.5,None),rebuild_basis=rebuild)
        nf = sum(np.sum(u.basis.states(k,frozen=True)[5])
                for k in kmesh(1,nk=2))
        assert nf==2*n # the lower spin band of the lower leg
        for op in [None,"sx","sy","sz"]:
            v1 = u.get_vev(operator=op,nk=2)
            v2 = hs.get_vev(operator=op,nk=2)
            assert np.max(np.abs(v1-v2))<1e-10
        lower = np.array(hs.geometry.r)[:,1]<0.
        sy = u.get_vev(operator="sy",nk=2)
        assert np.min(np.abs(sy[lower]))>0.1 # the frozen states hold it


@pytest.mark.parametrize("rebuild",[False,True])
def test_window_mean_field_of_decoupled_legs_is_the_supercell(rebuild):
    """A first-neighbor V1 on two legs without rungs, the lower one frozen
    and the upper one modulated and kept whole. A full decoupled band has
    the identity as density matrix whatever its Hamiltonian, and the trace
    of its block as band energy, so every iteration of the spinless loop
    is that of "ed" on the supercell from the same guess, which has no
    rung either"""
    from pyqula.scftk.densitydensity import random_hermitian_guess
    n = 6
    h = two_legs()
    hs = h.get_supercell(n)
    hs.add_onsite(upper_wave(n,v=0.3))
    norb = hs.intra.shape[0]
    np.random.seed(3)
    mf = random_hermitian_guess({(0,0,0):0,(1,0,0):0,(-1,0,0):0},
            hs.intra.shape,scale=0.3)
    hop = hs.get_multihopping().get_dict()
    for d in mf: # on the bonds of the legs and the sites only
        mask = np.abs(dense(hop[d]))>0. if d in hop else \
                np.zeros((norb,norb),dtype=bool)
        if d==(0,0,0): mask = mask | np.identity(norb,dtype=bool)
        mf[d] = mf[d]*mask
    kw = dict(V1=1.0,filling=17./24,nk=2,maxerror=1e-11,verbose=0,
            write=False)
    u = Ultracell(h,hs,window=(-3.,None))
    u2,e2 = u.get_mean_field_hamiltonian(return_total_energy=True,
            rebuild_basis=rebuild,mf={d: csr_matrix(m) for d,m in mf.items()},
            **kw)
    h3,e3 = hs.get_mean_field_hamiltonian(return_total_energy=True,
            integration="ed",mf=mf,**kw)
    assert abs(e2-e3)<1e-8
    assert same_state(u2,h3,k=(0.37,0.,0.))<1e-7
    assert u2.window==(-3.,None)


def test_window_uniform_mean_field_is_the_supercell():
    """At zero modulation the mean field from a uniform guess stays
    uniform, and the rebuilt basis is made of the eigenstates of the whole
    Hamiltonian at every iteration: the frozen states below emin are full
    eigenstates, the dropped ones above emax empty ones, so the window is
    exact and the antiferromagnet of graphene is that of "ed" on the
    supercell"""
    g = geometry.honeycomb_lattice()
    h = g.get_hamiltonian()
    hs = h.get_supercell([3,2])
    u = Ultracell(h,hs,window=(-2.,1.5))
    kw = dict(U=3.0,filling=0.5,mf="antiferro",nk=2,maxerror=1e-11,verbose=0,
            write=False)
    u2,e2 = u.get_mean_field_hamiltonian(return_total_energy=True,**kw)
    h3,e3 = hs.get_mean_field_hamiltonian(return_total_energy=True,
            integration="ed",**kw)
    assert abs(e2-e3)<1e-8
    assert same_state(u2,h3)<1e-7
    m = u2.get_vev(operator="sz",nk=2)
    assert np.min(np.abs(m))>0.3 # an antiferromagnet
    hc = u2.hs.copy()
    hc.ultracell_basis = u2.basis
    assert MeshStates(hc,nk=2).nfrozen>0 # the window froze states


def test_frozen_states_must_be_below_the_fermi_level():
    """The frozen states are full: a filling that does not reach above them,
    a chemical potential among them and expectation values with a frozen
    state above zero energy raise, and name emin"""
    n,nk = 6,2
    h = two_legs()
    hs = h.get_supercell(n)
    hs.add_onsite(upper_wave(n))
    hs.ultracell_basis = FoldedBasis(h,hs,window=(-3.,None))
    norb = hs.intra.shape[0]
    needed = all_pairs(norb,[(0,0,0)])
    with pytest.raises(ValueError,match="lower emin"): # 10 of 12 frozen
        SparseBackend(nk=nk,T=1e-7).dm_with_fermi(hs,needed,0.4)
    h2 = hs.copy() ; h2.shift_fermi(5.) # the lower leg from -3 to 1
    with pytest.raises(ValueError,match="lower emin"):
        SparseBackend(nk=nk,T=1e-7).dm(h2,needed)
    u = Ultracell(h,h2,window=(-3.,None))
    with pytest.raises(ValueError,match="lower emin"):
        u.get_vev(nk=nk)
    # and the frozen part of the density matrix is the identity on the
    # lower leg when the Fermi level is above it
    dm,fermi = SparseBackend(nk=nk,T=1e-7).dm_with_fermi(hs,needed,17./24)
    m = dm[(0,0,0)].toarray()
    lower = np.array(hs.geometry.r)[:,1]<0.
    assert np.max(np.abs(m[np.ix_(lower,lower)]-np.identity(n)))<1e-12
