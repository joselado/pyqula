"""The Nambu (BdG) ultracell and its mean field.

The cut of a BdG Hamiltonian is |E|<=emax from its zero, the window
(-emax,emax) on the BdG levels, and the BdG states below -emax are frozen,
with their anomalous part. Particle-hole symmetry maps the frozen states at
kS to the dropped ones at -kS, so the kept space is closed under it over a
mesh closed under -k. With every state kept the mean field is that of "ed"
on the supercell, and a uniform paired state is exact for any emax, since
the rebuilt basis is then made of the eigenstates of the whole Hamiltonian.
"""

import numpy as np
import pytest
from scipy.linalg import eigh
from scipy.sparse import coo_matrix

from pyqula import geometry
from pyqula.klist import kmesh
from pyqula.ultracell import Ultracell
from pyqula.ultracelltk.basis import FoldedBasis, dense
from pyqula.ultracelltk.meanfield import SparseBackend, frozen_entries


def fractional(g):
    B = np.linalg.inv(np.array([g.a1,g.a2,g.a3]).T)
    return lambda r: B@np.array(r)


def same_state(u2,h3,k=(0.37,0.,0.)):
    m2 = dense(u2.hs.get_hk_gen()(np.array(k)))
    m3 = dense(h3.get_hk_gen()(np.array(k)))
    return np.max(np.abs(m2-m3))


def nambu_chain(n=6,v=0.3):
    """A spinful chain with the Nambu degree of freedom and its supercell
    of n cells, under a potential periodic on it"""
    h = geometry.chain().get_hamiltonian()
    h.setup_nambu_spinor()
    hs = h.get_supercell(n)
    if v!=0.: hs.add_onsite(lambda r: v*np.cos(2*np.pi*r[0]/n+0.3))
    return h,hs


def pairing_model():
    """Graphene with Haldane and Rashba couplings, an exchange field and a
    sublattice imbalance, so with neither inversion nor time reversal,
    under a modulated s-wave pairing and potential on a 3x2 cell"""
    g = geometry.honeycomb_lattice()
    h = g.get_hamiltonian()
    h.add_haldane(0.1)
    h.add_rashba(0.2)
    h.add_exchange([0.1,0.2,0.3])
    h.add_sublattice_imbalance(0.15)
    h.setup_nambu_spinor()
    hs = h.get_supercell([3,2])
    x = fractional(hs.geometry)
    hs.add_swave(lambda r: 0.2 + 0.1*np.cos(2*np.pi*x(r)[0]))
    hs.add_onsite(lambda r: 0.2*np.sin(2*np.pi*x(r)[1]))
    return h,hs


def entries_of(hs):
    """Every stored entry of the hoppings of hs and the diagonal, once"""
    n = hs.intra.shape[0]
    out = {}
    for d,m in hs.get_multihopping().get_dict().items():
        m = coo_matrix(m)
        r,c = m.row.astype(np.int64),m.col.astype(np.int64)
        if tuple(d)==(0,0,0):
            r = np.concatenate([r,np.arange(n)])
            c = np.concatenate([c,np.arange(n)])
        key = np.unique(r*n+c)
        out[tuple(d)] = (key//n,key%n)
    return out


def test_nambu_cut_is_symmetric():
    """emax=x is the window (-x,x), and an asymmetric Nambu window raises;
    the Nambu mean field with a cut needs the rebuilt basis, and a Nambu
    Hamiltonian cannot be shifted after the search"""
    h,hs = nambu_chain()
    u = Ultracell(h,hs,emax=1.)
    assert u.window==(-1.,1.) and u.emin==-1. and u.emax==1.
    assert Ultracell(h,hs,window=(-1.,1.)).window==(-1.,1.)
    assert Ultracell(h,hs).window==(None,None)
    for window in [(-1.,2.),(None,1.),(-1.,None)]:
        with pytest.raises(ValueError,match="symmetric"):
            Ultracell(h,hs,window=window)
    with pytest.raises(ValueError):
        Ultracell(h,hs,emax=-0.5)
    with pytest.raises(NotImplementedError):
        u.get_mean_field_hamiltonian(U=-2.,rebuild_basis=False,verbose=0,
                write=False)
    hc = hs.copy()
    hc.ultracell_basis = FoldedBasis(h,hc,emax=1.)
    n = hc.intra.shape[0]
    with pytest.raises(NotImplementedError,match="rigidly"):
        SparseBackend(nk=1).dm_with_fermi(hc,
                {(0,0,0): (np.arange(n),np.arange(n))},0.3)


@pytest.mark.parametrize("rebuild",[False,True])
def test_nambu_window_holding_every_state_is_the_supercell(rebuild):
    h,hs = pairing_model()
    u = Ultracell(h,hs,emax=20.,rebuild_basis=rebuild)
    kS = [0.231,0.417,0.]
    e1 = eigh(u.get_hk_gen()(kS),eigvals_only=True)
    e2 = eigh(dense(hs.get_hk_gen()(kS)),eigvals_only=True)
    assert np.max(np.abs(e1-e2))<1e-11
    assert np.sum(u.basis.states(kS,frozen=True)[5])==0 # none frozen


@pytest.mark.parametrize("rebuild",[False,True])
def test_nambu_cut_is_closed_under_particle_hole_symmetry(rebuild):
    """On a model without inversion, the ultracell spectrum at kS is minus
    the one at -kS, with the same number of kept states, and the states
    frozen at kS are as many as those dropped at -kS. Over a mesh closed
    under -k, the electron and hole occupations of every site add up to
    two, one per Nambu pair, with the cut and a chemical potential"""
    h,hs = pairing_model()
    u = Ultracell(h,hs,emax=1.,rebuild_basis=rebuild)
    n = hs.intra.shape[0]
    kS = np.array([0.23,0.41,0.])
    e1 = eigh(u.get_hk_gen()(kS),eigvals_only=True)
    e2 = eigh(u.get_hk_gen()(-kS),eigvals_only=True)
    assert 0<len(e1)<n and len(e1)==len(e2)
    assert np.max(np.abs(e1+e2[::-1]))<1e-10
    f1 = np.sum(u.basis.states(kS,frozen=True)[5])
    f2 = np.sum(u.basis.states(-kS,frozen=True)[5])
    assert f1>0 and f1==n-len(e2)-f2 # frozen at kS, dropped at -kS
    hc = hs.copy() ; hc.shift_fermi(-0.1) # -mu on electrons, +mu on holes
    hc.turn_sparse()
    hc.ultracell_basis = FoldedBasis(h,hc,emax=1.)
    hc.ultracell_basis.rebuild = rebuild
    dm = SparseBackend(nk=2,T=1e-7).dm(hc,
            {(0,0,0): (np.arange(n),np.arange(n))})
    occ = np.real(dm[(0,0,0)].diagonal())
    assert np.max(np.abs(occ.reshape(-1,4).sum(axis=1)-2.))<1e-10


@pytest.mark.parametrize("rebuild",[False,True])
def test_nambu_band_energy_is_the_trace_with_the_density_matrix(rebuild):
    """The Nambu band energy is (Tr[rho H] + Tr h_e)/2, with rho the BdG
    density matrix the backend returns, frozen states included, and Tr h_e
    the electron diagonal of the Bloch Hamiltonian on the mesh, as
    spectrum.total_energy for "ed" """
    h,hs = pairing_model()
    hs.shift_fermi(-0.1)
    hs.turn_sparse()
    hs.ultracell_basis = FoldedBasis(h,hs,emax=1.)
    hs.ultracell_basis.rebuild = rebuild
    n = hs.intra.shape[0]
    b = SparseBackend(nk=2,T=1e-7)
    dm = b.dm(hs,entries_of(hs))
    assert b.states.nfrozen>0
    e1 = b.band_energy(hs)
    tr = 0.
    for d,m in hs.get_multihopping().get_dict().items():
        m = coo_matrix(m)
        tr += np.real(np.sum(m.data*dm[tuple(d)].toarray()[m.row,m.col]))
    hk = hs.get_hk_gen()
    the = np.mean([np.sum(np.real(np.diag(dense(hk(k))))[np.arange(n)%4<2])
        for k in kmesh(2,nk=2)])
    assert abs(e1-(tr+the)/2.)<1e-8


@pytest.mark.parametrize("rebuild",[False,True])
def test_nambu_nested_cuts_are_variational(rebuild):
    """At a fixed chemical potential the one-shot grand potential of the
    BdG ultracell is that of a Bogoliubov vacuum in a restricted space: the
    frozen states full and the negative ultracell states of the kept ones.
    It is never below the supercell, and a larger emax holds the state of a
    smaller one, so it never increases with emax"""
    h,hs = pairing_model()
    hs.shift_fermi(-0.1)
    h3 = hs.copy() ; h3.turn_dense()
    e0 = h3.get_total_energy(nk=2)
    es = []
    for emax in [0.6,1.0,1.6,2.4]:
        hc = hs.copy() ; hc.turn_sparse()
        hc.ultracell_basis = FoldedBasis(h,hc,emax=emax)
        hc.ultracell_basis.rebuild = rebuild
        b = SparseBackend(nk=2,T=1e-7)
        b.solve(hc)
        es.append(b.band_energy(hc))
    assert es[0]>e0+1e-6 # the smallest cut misses something
    for e in es: assert e>=e0-1e-10
    for e1,e2 in zip(es[0:-1],es[1:]): assert e2<=e1+1e-10


@pytest.mark.parametrize("fixed_mu",[False,True])
def test_nambu_mean_field_every_state_is_the_supercell(fixed_mu):
    """An attractive U on a chain under a modulated potential, with every
    state kept: the BdG mean field of the ultracell, at a filling (the
    Fermi level of the paired state, from the electrons in their own basis)
    and at a fixed chemical potential, is that of "ed" on the supercell
    from the same guess"""
    from pyqula.scftk.superscf import pairing_amplitude
    h,hs = nambu_chain(v=0.3)
    kw = dict(U=-2.5,mf="swave",nk=3,maxerror=1e-11,verbose=0,write=False)
    if fixed_mu: kw["mu"] = -1.2
    else: kw["filling"] = 0.28 # 10 of 36 states, in a gap of the chain
    u2,e2 = Ultracell(h,hs).get_mean_field_hamiltonian(
            return_total_energy=True,**kw)
    h3,e3 = hs.get_mean_field_hamiltonian(return_total_energy=True,
            integration="ed",**kw)
    assert abs(e2-e3)<1e-8
    assert same_state(u2,h3)<1e-7
    assert pairing_amplitude(u2.hs)>0.05 # a paired state


@pytest.mark.parametrize("fixed_mu",[False,True])
def test_uniform_pairing_with_a_cut_is_the_supercell(fixed_mu):
    """A uniform attractive U at zero modulation, cut at |E|<=1.5, which
    freezes the BdG states of the top of the band. The rebuilt basis is
    made of the eigenstates of the whole Hamiltonian, so the cut is exact,
    and the pairing and the energy are those of "ed" on the supercell. The
    frozen states carry a sizable anomalous part, so the agreement needs
    it"""
    h,hs = nambu_chain(v=0.)
    kw = dict(U=-2.5,mf="swave",nk=3,maxerror=1e-11,verbose=0,write=False)
    if fixed_mu: kw["mu"] = -1.2
    else: kw["filling"] = 0.28
    u2,e2 = Ultracell(h,hs,emax=1.5).get_mean_field_hamiltonian(
            return_total_energy=True,**kw)
    h3,e3 = hs.get_mean_field_hamiltonian(return_total_energy=True,
            integration="ed",**kw)
    assert abs(e2-e3)<1e-8
    assert same_state(u2,h3)<1e-7
    assert u2.window==(-1.5,1.5)
    b = u2.basis # the BdG states of the converged periodic part
    ks = kmesh(1,nk=3)
    frozen = [b.states(k,frozen=True)[4] for k in ks]
    kept = sum(np.sum(b.states(k)[2]) for k in ks)
    assert sum(f.shape[2] for f in frozen)>0 and kept>0
    n = u2.hs.intra.shape[0]
    i = np.arange(n//4) # the anomalous entries of every site
    r = np.concatenate([4*i,4*i,4*i+1,4*i+1])
    c = np.concatenate([4*i+2,4*i+3,4*i+2,4*i+3])
    fz = frozen_entries(b,ks,frozen,{(0,0,0): (r,c)})[(0,0,0)]
    assert np.max(np.abs(fz))>1e-3
