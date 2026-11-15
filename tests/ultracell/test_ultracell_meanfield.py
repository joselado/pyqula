"""The mean field of the ultracell, integration="ultracell" of scftk.

With every state kept, the loop sees the same supercell Hamiltonian in
another basis, so it must converge to the mean field of the supercell
itself, integration="ed", from the same initial guess. Its response to a
weak modulation must be the RPA response about the converged mean-field
state, which chitk computes on the unit cell alone. With a cut, the basis
rebuilt from the lattice-periodic part of the mean field must converge to
the full result as emax grows, and do better than the basis of the clean
unit cell when the mean field has a uniform part.
"""

import numpy as np
import pytest

from pyqula import geometry
from pyqula.ultracelltk.basis import dense


def fractional(g):
    B = np.linalg.inv(np.array([g.a1,g.a2,g.a3]).T)
    return lambda r: B@np.array(r)


def same_state(u2,h3,k=(0.21,0.33,0.)):
    m2 = dense(u2.hs.get_hk_gen()(np.array(k)))
    m3 = dense(h3.get_hk_gen()(np.array(k)))
    return np.max(np.abs(m2-m3))


@pytest.mark.parametrize("rebuild",[True,False])
def test_hubbard_every_state_is_the_supercell(rebuild):
    g = geometry.honeycomb_lattice()
    h = g.get_hamiltonian() # spinful
    hs = h.get_supercell([3,2])
    x = fractional(hs.geometry)
    hs.add_onsite(lambda r: 0.2*np.cos(2*np.pi*x(r)[0]))
    u = h.get_ultracell(hs)
    kw = dict(U=3.0,filling=0.5,mf="antiferro",nk=2,maxerror=1e-11,verbose=0,
            write=False) # well below the 1e-8 the two engines are held to
    u2,e2 = u.get_mean_field_hamiltonian(return_total_energy=True,
            rebuild_basis=rebuild,**kw)
    h3,e3 = hs.get_mean_field_hamiltonian(return_total_energy=True,
            integration="ed",**kw)
    assert abs(e2-e3)<1e-8
    assert same_state(u2,h3)<1e-7
    m = u2.get_vev(operator="sz",nk=2)
    assert np.min(np.abs(m))>0.3 # an antiferromagnet, not the trivial state
    assert np.max(np.abs(m-h3.get_vev(operator="sz",nk=2)))<1e-7


def test_bond_interaction_every_state_is_the_supercell():
    """A first-neighbor V1 has Fock terms on the bonds, so the mean field
    modulates the hoppings and not only the onsite energies"""
    g = geometry.chain()
    h = g.get_hamiltonian(has_spin=False)
    n = 6
    hs = h.get_supercell(n)
    hs.add_onsite(lambda r: 0.3*np.cos(2*np.pi*r[0]/n))
    u = h.get_ultracell(hs)
    # one random guess for both, since the sparse loop of the ultracell
    # draws a random guess on the pairs the interaction couples only
    from scipy.sparse import csr_matrix
    from pyqula.scftk.densitydensity import random_hermitian_guess
    np.random.seed(3)
    mf = random_hermitian_guess({(0,0,0):0,(1,0,0):0,(-1,0,0):0},
            hs.intra.shape,scale=0.3)
    kw = dict(V1=1.5,filling=0.5,nk=4,maxerror=1e-11,verbose=0,write=False)
    u2,e2 = u.get_mean_field_hamiltonian(return_total_energy=True,
            mf={d: csr_matrix(m) for d,m in mf.items()},**kw)
    h3,e3 = hs.get_mean_field_hamiltonian(return_total_energy=True,
            integration="ed",mf=mf,**kw)
    assert abs(e2-e3)<1e-8
    assert same_state(u2,h3,k=(0.37,0.,0.))<1e-7


def canted_guess(hs,theta=0.4):
    """A staggered moment tilted by theta from z towards x, as the onsite
    matrix of the mean field, so that the x and y channels of an isotropic
    exchange are active from the first iteration"""
    from pyqula.ultracelltk.basis import dense
    hg = hs.copy() ; hg.turn_dense()
    m0 = dense(hg.intra).copy()
    r0 = np.array(hs.geometry.r)
    sub = np.array(hs.geometry.sublattice)
    def ex(r):
        s = sub[np.argmin(np.sum((r0-np.array(r))**2,axis=1))]
        return [0.5*s*np.sin(theta),0.,0.5*s*np.cos(theta)]
    hg.add_exchange(ex)
    return {(0,0,0): dense(hg.intra)-m0}


def test_exchange_every_state_is_the_supercell():
    """The spin-spin exchange J1 has Fock terms on the bonds in every spin
    channel; with every state kept the sparse engine on the ultracell
    states must converge to the mean field of "ed" on the supercell from
    the same tilted guess. Without J1 the same guess relaxes slowly in
    both engines alike, by 0.5% per iteration, so it is not used"""
    g = geometry.honeycomb_lattice()
    h = g.get_hamiltonian()
    hs = h.get_supercell([3,2])
    x = fractional(hs.geometry)
    hs.add_onsite(lambda r: 0.2*np.cos(2*np.pi*x(r)[0]))
    u = h.get_ultracell(hs)
    kw = dict(U=2.5,J1=0.4,filling=0.5,nk=2,maxerror=1e-11,verbose=0,
            write=False)
    u2,e2 = u.get_mean_field_hamiltonian(return_total_energy=True,
            mf=canted_guess(hs),**kw)
    h3,e3 = hs.get_mean_field_hamiltonian(return_total_energy=True,
            integration="ed",mf=canted_guess(hs),**kw)
    assert abs(e2-e3)<1e-8
    assert same_state(u2,h3)<1e-7
    mx = u2.get_vev(operator="sx",nk=2)
    assert np.max(np.abs(mx))>0.05 # a non-collinear guess stays tilted


def test_exchange_cut_converges():
    """With U and J1, the rebuilt basis converges to the full result as
    emax grows, from above"""
    g = geometry.honeycomb_lattice()
    h = g.get_hamiltonian()
    n = 4
    hs = h.get_supercell([n,1])
    x = fractional(g)
    hs.add_onsite(lambda r: 0.3*np.cos(2*np.pi*x(r)[0]/n))
    kw = dict(U=2.5,J1=0.4,filling=0.5,mf="antiferro",nk=2,maxerror=1e-8,
            verbose=0,return_total_energy=True,write=False)
    u0,e0 = h.get_ultracell(hs).get_mean_field_hamiltonian(**kw)
    err = []
    for emax in [1.0,1.5,2.5]:
        u2,e = h.get_ultracell(hs,emax=emax).get_mean_field_hamiltonian(**kw)
        err.append(abs(e-e0))
        assert e>=e0-1e-8 # the cut cannot go below the full basis
    assert err[0]>err[1]>err[2]
    assert err[2]<1e-3


def test_linear_response_is_the_rpa_about_the_mean_field():
    """A weak potential eps cos(2 pi x1/n) on a gapped honeycomb lattice:
    the change of the occupations, Fourier transformed over the cells, is
    chi V at q=Q, with chi the bare response at U=0 and the RPA response
    about the mean-field state at finite U, on the same k-mesh"""
    from pyqula.chitk.chiAB import chiAB_q
    from pyqula.chitk.densitychi import densitychi_RPA
    n,nk,eps = 4,2,1e-4
    g = geometry.honeycomb_lattice()
    x = fractional(g)
    V = lambda r: eps*np.cos(2*np.pi*x(r)[0]/n)
    for U in [0.,1.0]:
        h = g.get_hamiltonian()
        h.add_sublattice_imbalance(1.0)
        hs = h.get_supercell([n,n])
        hs1 = hs.copy() ; hs1.add_onsite(V)
        u0,u1 = h.get_ultracell(hs),h.get_ultracell(hs1)
        if U==0.:
            dn = u1.get_vev(nk=nk) - u0.get_vev(nk=nk)
            es,chis = chiAB_q(h,energies=[0.],q=[1./n,0.,0.],nk=n*nk,
                    delta=1e-7)
        else:
            kw = dict(U=U,filling=0.5,nk=nk,maxerror=1e-12,verbose=0,
                    mf="random",write=False)
            np.random.seed(1)
            m1 = u1.get_mean_field_hamiltonian(**kw).get_vev(nk=nk)
            np.random.seed(2)
            m0 = u0.get_mean_field_hamiltonian(**kw).get_vev(nk=nk)
            dn = m1 - m0
            np.random.seed(3)
            hmf = h.get_mean_field_hamiltonian(U=U,filling=0.5,nk=n*nk,
                    maxerror=1e-12,verbose=0,mf="random",integration="ed",
                    write=False)
            es,chis = densitychi_RPA(hmf,U=U,energies=[0.],q=[1./n,0.,0.],
                    nk=n*nk,delta=1e-7)
        v = np.array([V(r) for r in hs.geometry.r])
        Q,f = u1.get_harmonics(dn)
        Q,fv = u1.get_harmonics(v) # the potential in the same cell gauge
        i = np.where((np.round(Q[:,0]*n)==1)&(np.round(Q[:,1]*n)==0))[0][0]
        pred = np.array(chis[0])@fv[i]
        assert np.max(np.abs(f[i]-pred))<1e-3*np.max(np.abs(pred))
        assert np.max(np.abs(pred))>1e-7 # the response is not empty


def test_cut_rebuilt_basis_converges():
    """The antiferromagnet of graphene under a slow potential, cut at emax:
    the rebuilt basis converges to the full result as emax grows, and does
    better than the basis of the clean unit cell, which misses the uniform
    moment"""
    g = geometry.honeycomb_lattice()
    h = g.get_hamiltonian()
    n = 4
    hs = h.get_supercell([n,1])
    x = fractional(g)
    hs.add_onsite(lambda r: 0.3*np.cos(2*np.pi*x(r)[0]/n))
    kw = dict(U=3.0,filling=0.5,mf="antiferro",nk=2,maxerror=1e-8,verbose=0,
            return_total_energy=True,write=False)
    u0,e0 = h.get_ultracell(hs).get_mean_field_hamiltonian(**kw)
    m0 = u0.get_vev(operator="sz",nk=2)
    err = []
    for emax in [1.0,1.5,2.5]:
        u2,e = h.get_ultracell(hs,emax=emax).get_mean_field_hamiltonian(**kw)
        err.append(abs(e-e0))
        assert e>=e0-1e-8 # the cut cannot go below the full basis
    assert err[0]>err[1]>err[2]
    assert err[2]<1e-3
    u3,e3 = h.get_ultracell(hs,emax=1.0).get_mean_field_hamiltonian(
            rebuild_basis=False,**kw)
    assert abs(e3-e0)>10*err[0]
    m3 = u3.get_vev(operator="sz",nk=2)
    assert np.max(np.abs(m3-m0))>0.1 # the clean basis loses the moment


def test_mean_field_wrong_inputs_raise():
    g = geometry.honeycomb_lattice()
    h = g.get_hamiltonian()
    hs = h.get_supercell([2,2])
    u = h.get_ultracell(hs,emax=0.)
    with pytest.raises(ValueError): # the cut leaves no state above zero
        u.get_mean_field_hamiltonian(U=1.,filling=0.5,nk=1,verbose=0,
                write=False)
    with pytest.raises(ValueError):
        h.get_ultracell(hs).get_mean_field_hamiltonian(U=1.,integration="ed")
    with pytest.raises(ValueError): # a plain supercell has no ultracell basis
        hs.get_mean_field_hamiltonian(U=1.,integration="ultracell",verbose=0,
                write=False)
    hl = geometry.chain().get_hamiltonian(has_spin=False)
    with pytest.raises(ValueError): # the exchange needs spin
        hl.get_ultracell(hl.get_supercell(3)).get_mean_field_hamiltonian(
                J1=1.,write=False)
    hn = h.copy() ; hn.setup_nambu_spinor()
    with pytest.raises(NotImplementedError):
        hn.get_ultracell(hn.get_supercell([2,2])).get_mean_field_hamiltonian(
                U=-1.)


@pytest.mark.parametrize("emax",[None,0.5])
def test_sparse_backend_reads_the_density_matrix(emax):
    """The sparse engine of the spinful mean field takes the density matrix
    at the entries it reads, at d and at -d, as csr matrices. They must be
    those of the dense density matrix of the same ultracell states, and,
    with every state kept, those of "ed" on the supercell, with the Fermi
    level of the same filling"""
    from pyqula.scftk import sparsemeanfield
    from pyqula.ultracelltk.basis import FoldedBasis
    from pyqula.ultracelltk.meanfield import (SparseBackend, MeshStates,
            dm_full, smearing)
    g = geometry.honeycomb_lattice()
    h = g.get_hamiltonian()
    h.add_rashba(0.2)
    hs = h.get_supercell([3,2])
    x = fractional(hs.geometry)
    hs.add_onsite(lambda r: 0.3*np.cos(2*np.pi*x(r)[0]))
    hs.add_exchange(lambda r: [0.2*np.sin(2*np.pi*x(r)[0]),0.,0.1])
    hs.turn_sparse()
    hs.ultracell_basis = FoldedBasis(h,hs,emax=emax)
    n = hs.intra.shape[0]
    v = sparsemeanfield.interaction(hs,V1=1.0,
            U=np.ones(len(hs.geometry.r)))
    needed = sparsemeanfield.needed_entries([v],n,blocks=True)
    assert any(d!=(0,0,0) for d in needed) # bonds between supercells
    b = SparseBackend(nk=3,T=1e-7)
    dm,fermi = b.dm_with_fermi(hs,needed,0.5)
    h2 = hs.copy() ; h2.shift_fermi(-fermi)
    full = dm_full(MeshStates(h2,nk=3),list(needed),smearing(1e-7))
    for d,(r,c) in needed.items():
        assert np.max(np.abs(dm[d].toarray()[r,c]-full[d][r,c]))<1e-12
    if emax is None: # the supercell itself
        h3 = hs.copy() ; h3.turn_dense()
        fermi3 = h3.get_fermi4filling(0.5,nk=3,T=1e-7)
        assert abs(fermi-fermi3)<1e-9
        h3.shift_fermi(-fermi3)
        eds = h3.get_density_matrix(ds=list(needed),nk=3,T=1e-7)
        for d,(r,c) in needed.items():
            ed = dense(eds[d])
            assert np.max(np.abs(dm[d].toarray()[r,c]-ed[r,c]))<1e-8
    # the band energy of the shifted Hamiltonian, from the kept states
    hs.shift_fermi(-fermi) # the backend kept the states of hs
    e1 = b.band_energy(hs)
    e2 = SparseBackend(nk=3).band_energy(h2) # recomputed
    assert abs(e1-e2)<1e-10


def test_full_filling_without_a_cut():
    """With every state kept a filling of one occupies all of them, as for
    "ed"; with a cut it needs the states above emax and raises"""
    from pyqula.ultracelltk.meanfield import fermi_level
    es = [np.array([-1.,0.,1.]),np.array([-0.5,0.5,2.])]
    assert fermi_level(es,3,1.0)>=2.
    with pytest.raises(ValueError): # 6 states needed, 5 kept
        fermi_level([es[0],es[1][0:2]],3,1.0)


def test_nondiagonal_mean_field_every_state():
    """The Kekule cell of graphene, sqrt3 x sqrt3, a supercell along other
    vectors than those of the unit cell, with a first-neighbor V1 whose
    Fock terms modulate the bonds: with every state kept the mean field is
    that of "ed" on the supercell, from the same guess"""
    from scipy.sparse import csr_matrix
    from pyqula.scftk.densitydensity import random_hermitian_guess
    g = geometry.honeycomb_lattice()
    gs = g.get_supercell([[2,1,0],[-1,1,0],[0,0,1]],store_primal=True)
    h = g.get_hamiltonian(has_spin=False)
    hs = gs.get_hamiltonian(has_spin=False)
    x = fractional(gs)
    hs.add_onsite(lambda r: 0.2*np.cos(2*np.pi*(x(r)[0]+x(r)[1])))
    u = h.get_ultracell(hs)
    np.random.seed(5)
    mf = random_hermitian_guess(hs.get_multihopping().get_dict(),
            hs.intra.shape,scale=0.3)
    kw = dict(V1=2.0,filling=0.5,nk=3,maxerror=1e-11,verbose=0,write=False)
    u2,e2 = u.get_mean_field_hamiltonian(return_total_energy=True,
            mf={d: csr_matrix(m) for d,m in mf.items()},**kw)
    h3,e3 = hs.get_mean_field_hamiltonian(return_total_energy=True,
            integration="ed",mf=mf,**kw)
    assert abs(e2-e3)<1e-8
    assert same_state(u2,h3)<1e-7


@pytest.mark.parametrize("spin",[True,False])
def test_fixed_chemical_potential_every_state_is_the_supercell(spin):
    """With mu= instead of a filling the loop shifts the Hamiltonian by mu
    and solves it, and the energy has no Fermi term; with every state kept
    it must give the mean field and the energy of "ed" on the supercell,
    spinful (the sparse engine of the exchange) and spinless (the sparse
    density-density loop)"""
    from scipy.sparse import csr_matrix
    from pyqula.scftk.densitydensity import random_hermitian_guess
    if spin:
        g = geometry.honeycomb_lattice()
        h = g.get_hamiltonian()
        hs = h.get_supercell([3,2])
        kw = dict(U=2.5,J1=0.3,mu=0.2,mf="antiferro")
    else:
        g = geometry.chain()
        h = g.get_hamiltonian(has_spin=False)
        hs = h.get_supercell(6)
        np.random.seed(7)
        mf = random_hermitian_guess({(0,0,0):0,(1,0,0):0,(-1,0,0):0},
                hs.intra.shape,scale=0.3)
        kw = dict(V1=1.5,mu=0.1,mf=mf)
    x = fractional(hs.geometry)
    hs.add_onsite(lambda r: 0.2*np.cos(2*np.pi*x(r)[0]))
    kw.update(nk=3,maxerror=1e-11,verbose=0,write=False)
    kw2 = dict(kw)
    if not spin: kw2["mf"] = {d: csr_matrix(m) for d,m in kw["mf"].items()}
    u2,e2 = h.get_ultracell(hs).get_mean_field_hamiltonian(
            return_total_energy=True,**kw2)
    h3,e3 = hs.get_mean_field_hamiltonian(return_total_energy=True,
            integration="ed",**kw)
    assert abs(e2-e3)<1e-8
    k = (0.21,0.33,0.) if spin else (0.37,0.,0.)
    assert same_state(u2,h3,k=k)<1e-7


@pytest.mark.parametrize("rebuild",[False,True])
def test_kpoint_without_kept_states_in_the_mean_field(rebuild):
    """Spinless graphene on a 2x2 cell with emax=-2.7 keeps Gamma (E=-3)
    and the six momenta next to it (E=-2.80) on the folded 8x8 mesh, and
    no state at kS=(1/2,0) of the 4x4 mesh. The backend must skip that
    kS: one electron on the mesh fills Gamma alone, whatever the empty
    kpoints, so the trace of the density matrix is one electron"""
    from pyqula.ultracelltk.basis import FoldedBasis
    from pyqula.ultracelltk.meanfield import (SparseBackend, MeshStates,
            fermi_level)
    g = geometry.honeycomb_lattice()
    h = g.get_hamiltonian(has_spin=False)
    hs = h.get_supercell([2,2])
    hs.ultracell_basis = FoldedBasis(h,hs,emax=-2.7)
    hs.ultracell_basis.rebuild = rebuild
    n = hs.intra.shape[0]
    states = MeshStates(hs,nk=4)
    kept = [len(e) for e in states.eigenvalues()]
    assert min(kept)==0 and sum(kept)==7 # Gamma and its six neighbors
    needed = {(0,0,0): (np.arange(n),np.arange(n))}
    dm,fermi = SparseBackend(nk=4,T=1e-7).dm_with_fermi(hs,needed,
            1./(16*n))
    assert -3.<fermi<-2.8
    assert abs(16*np.real(np.sum(dm[(0,0,0)].diagonal()))-1.)<1e-8
    # a kpoint without states in the T=0 cut of the Fermi level
    es = [np.array([-1.,0.]),np.zeros(0)]
    assert abs(fermi_level(es,3,1./6)+0.5)<1e-12


def test_fixed_chemical_potential_needs_the_states_below_it():
    """With a cut at emax=0.5 the states of graphene between 0.5 and mu
    are not in the basis. A fixed mu above every kept state would leave
    them out of the density matrix, the mean field and the band energy
    without a message, while a filling that needs them raises. The fixed
    mu must raise the same way, and a mu below the highest kept state
    must not"""
    from pyqula.scftk import sparsemeanfield
    from pyqula.ultracelltk.basis import FoldedBasis
    from pyqula.ultracelltk.meanfield import SparseBackend
    g = geometry.honeycomb_lattice()
    h = g.get_hamiltonian()
    hs = h.get_supercell([3,2])
    hs.ultracell_basis = FoldedBasis(h,hs,emax=0.5)
    n = hs.intra.shape[0]
    v = sparsemeanfield.interaction(hs,U=np.ones(len(hs.geometry.r)))
    needed = sparsemeanfield.needed_entries([v],n)
    for mu,raises in [(1.0,True),(-2.0,False)]:
        h2 = hs.copy() ; h2.shift_fermi(-mu)
        b = SparseBackend(nk=2,T=1e-7)
        if raises:
            with pytest.raises(ValueError,match="chemical potential"):
                b.dm(h2,needed)
        else: b.dm(h2,needed)
    # the loops, spinful and spinless, with mu far above the kept states
    u = h.get_ultracell(hs,emax=0.5)
    with pytest.raises(ValueError,match="chemical potential"):
        u.get_mean_field_hamiltonian(U=1.0,mu=3.0,mf="antiferro",nk=2,
                verbose=0,write=False)
    from scipy.sparse import csr_matrix
    from pyqula.scftk.densitydensity import random_hermitian_guess
    hl = geometry.chain().get_hamiltonian(has_spin=False)
    hc = hl.get_supercell(6)
    np.random.seed(7)
    mf = random_hermitian_guess({(0,0,0):0,(1,0,0):0,(-1,0,0):0},
            hc.intra.shape,scale=0.3)
    with pytest.raises(ValueError,match="chemical potential"):
        hl.get_ultracell(hc,emax=0.5).get_mean_field_hamiltonian(V1=1.0,
                mu=3.0,mf={d: csr_matrix(m) for d,m in mf.items()},nk=2,
                verbose=0,write=False)
