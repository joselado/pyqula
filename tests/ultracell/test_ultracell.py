"""The ultracell (ultra long-range method) against the supercell it solves.

With every state kept the ultracell is the supercell written in the folded
Bloch basis of the unit cell, so the spectrum, the local density of states
and the expectation values per site must be those of the supercell to
rounding, for any modulation, and at zero modulation the unfolded spectral
function and the density of states must be those of the unit cell. With a
cut of the basis at emax the ultracell is a Rayleigh-Ritz compression of
the supercell, so its i-th lowest level can never go below the i-th lowest
level of the supercell, and it can only come down as emax grows.
"""

import numpy as np
import pytest
from scipy.linalg import eigh

from pyqula import geometry
from pyqula.multihopping import MultiHopping
from pyqula.unfolding import bloch_projector
from pyqula.ultracelltk.basis import dense


def fractional(hs):
    """Function giving the reduced coordinates of a position in the
    lattice of the supercell, so that a modulation periodic in them is
    periodic on the supercell"""
    d = hs.dimensionality
    A = np.array([hs.geometry.a1,hs.geometry.a2,hs.geometry.a3])[0:d,0:d]
    B = np.linalg.inv(A.T)
    return lambda r: B@np.array(r)[0:d]


def modulate_hoppings(hs,f):
    """Multiply every hopping of hs by 1+f at the middle of its bond; f
    periodic on the supercell makes the result Hermitian whichever end of
    the bond the convention measures it from"""
    hs.turn_multicell()
    d = {}
    a = [np.array(hs.geometry.a1),np.array(hs.geometry.a2),
            np.array(hs.geometry.a3)]
    fo = hs.intra.shape[0]//len(hs.geometry.r) # orbitals per site
    for key,M in hs.get_multihopping().dict.items():
        M = dense(M).copy()
        dv = sum(key[i]*a[i] for i in range(3))
        for i,j in zip(*np.nonzero(M)):
            si,sj = i//fo,j//fo
            if si==sj and tuple(key)==(0,0,0): continue # onsite block
            M[i,j] *= 1. + f(0.5*(hs.geometry.r[si]+hs.geometry.r[sj]+dv))
        d[key] = M
    hs.set_multihopping(MultiHopping(d))
    return hs


def modulated_honeycomb(nsuper=(5,3),spin=True,rashba=0.3):
    g = geometry.honeycomb_lattice()
    h = g.get_hamiltonian(has_spin=spin)
    if spin: h.add_rashba(rashba)
    hs = h.get_supercell(list(nsuper))
    x = fractional(hs)
    hs.add_onsite(lambda r: 0.3*np.cos(2*np.pi*x(r)[0]+0.2)
            + 0.1*np.sin(2*np.pi*(x(r)[0]+x(r)[1])))
    if spin:
        hs.add_exchange(lambda r: [0.4*np.cos(2*np.pi*x(r)[0]),
            0.3*np.sin(2*np.pi*x(r)[1]),0.2])
    modulate_hoppings(hs,lambda r: 0.2*np.cos(2*np.pi*x(r)[1]+0.5))
    return h,hs


def spectra(u,hs,kS):
    e1 = eigh(u.get_hk_gen()(kS),eigvals_only=True)
    e2 = eigh(dense(hs.get_hk_gen()(kS)),eigvals_only=True)
    return e1,e2


@pytest.mark.parametrize("n",[6,7])
def test_chain_every_state_is_the_supercell(n):
    g = geometry.chain()
    h = g.get_hamiltonian() # spinful
    h.add_onsite(0.1)
    hs = h.get_supercell(n)
    hs.add_onsite(lambda r: 0.5*np.cos(2*np.pi*r[0]/n+0.3))
    hs.add_exchange(lambda r: [0.3*np.sin(2*np.pi*r[0]/n),0.,
        0.2*np.cos(4*np.pi*r[0]/n)])
    u = h.get_ultracell(hs)
    e1,e2 = spectra(u,hs,[0.37,0.,0.])
    assert np.max(np.abs(e1-e2))<1e-12


@pytest.mark.parametrize("nsuper,spin",[((5,3),True),((4,4),False),
    ((3,2),True)])
def test_honeycomb_every_state_is_the_supercell(nsuper,spin):
    h,hs = modulated_honeycomb(nsuper,spin=spin)
    u = h.get_ultracell(hs)
    for kS in ([0.231,0.417,0.],[0.,0.,0.],[0.5,0.5,0.]):
        e1,e2 = spectra(u,hs,kS)
        assert np.max(np.abs(e1-e2))<1e-12


def test_cubic_every_state_is_the_supercell():
    g = geometry.cubic_lattice()
    h = g.get_hamiltonian(has_spin=False)
    hs = h.get_supercell([2,2,3])
    x = fractional(hs)
    hs.add_onsite(lambda r: 0.4*np.cos(2*np.pi*x(r)[2]) +
            0.2*np.sin(2*np.pi*(x(r)[0]+x(r)[1])))
    u = h.get_ultracell(hs)
    e1,e2 = spectra(u,hs,[0.11,0.23,0.37])
    assert np.max(np.abs(e1-e2))<1e-12


def test_modulated_pairing_every_state_is_the_supercell():
    g = geometry.honeycomb_lattice()
    h = g.get_hamiltonian()
    h.add_rashba(0.2)
    h.setup_nambu_spinor()
    hs = h.get_supercell([4,3])
    x = fractional(hs)
    hs.add_swave(lambda r: 0.2 + 0.1*np.cos(2*np.pi*x(r)[0]))
    hs.add_exchange(lambda r: [0.,0.,0.3*np.sin(2*np.pi*x(r)[1])])
    u = h.get_ultracell(hs)
    e1,e2 = spectra(u,hs,[0.19,0.43,0.])
    assert np.max(np.abs(e1-e2))<1e-12


def test_unfolded_weight_is_the_unfolding_operator():
    """sum_m |c_{q=0,m}|^2 is the weight per unit cell; the unfolding
    operator of the supercell gives N times it"""
    h,hs = modulated_honeycomb((4,3))
    u = h.get_ultracell(hs)
    N = 12
    p = np.array([0.13,0.29,0.])
    es = np.linspace(-3.,3.,60)
    a1 = u.get_kdos_bands(kpath=[p],energies=es,delta=0.05,write=False)[2]
    op = bloch_projector(hs,g0=h.geometry)
    kS = p*np.array([4,3,1])
    a2 = hs.get_kdos_bands(kpath=[kS],operator=op,energies=es,delta=0.05,
            write=False)[2]
    assert np.max(np.abs(N*a1-a2))<1e-10*np.max(np.abs(a2))
    assert np.max(a2)>1. # the spectral function is not empty here


def test_zero_modulation_gives_the_unit_cell():
    g = geometry.honeycomb_lattice()
    h = g.get_hamiltonian()
    h.add_rashba(0.3)
    h.add_exchange([0.1,0.2,0.3])
    u = h.get_ultracell(h.get_supercell([5,3]))
    kp = [[0.1,0.2,0.],[0.33,0.1,0.],[0.5,0.,0.]]
    es = np.linspace(-3.,3.,80)
    r1 = u.get_kdos_bands(kpath=kp,energies=es,delta=0.05,write=False)
    r2 = h.get_kdos_bands(kpath=kp,energies=es,delta=0.05,write=False)
    assert np.max(np.abs(np.array(r1)-np.array(r2)))<1e-10
    d1 = u.get_dos(nk=2,delta=0.1,energies=es,write=False)[1]
    d2 = h.get_dos(nk=[10,6],delta=0.1,energies=es,write=False)[1]
    assert np.max(np.abs(d1-d2))<1e-10


def test_ldos_and_expectation_values_are_the_supercell():
    h,hs = modulated_honeycomb((4,3))
    u = h.get_ultracell(hs)
    l1 = u.get_ldos(e=0.4,delta=0.1,nk=2,write=False)[2]
    l2 = hs.get_ldos(e=0.4,delta=0.1,nk=2,mode="diagonalization",nrep=1,
            write=False)[2]
    assert np.max(np.abs(l1-l2))<1e-10
    assert np.std(l1)>1e-3 # the modulation shows in the map
    for op in [None,"sz","sx"]:
        v1 = u.get_vev(operator=op,nk=2)
        v2 = hs.get_vev(operator=op,nk=2)
        assert np.max(np.abs(v1-v2))<1e-10


def test_cut_basis_is_a_rayleigh_ritz_compression():
    """The i-th lowest level of the cut basis is never below the i-th
    lowest level of the supercell, and it comes down as emax grows"""
    g = geometry.honeycomb_lattice()
    h = g.get_hamiltonian(has_spin=False)
    n = 15
    hs = h.get_supercell([n,1])
    x = fractional(hs)
    hs.add_onsite(lambda r: 0.4*np.cos(2*np.pi*x(r)[0]))
    kS = [0.2,0.37,0.]
    e0 = eigh(dense(hs.get_hk_gen()(kS)),eigvals_only=True)
    last = None
    for emax in [-1.5,-1.,-0.5,0.,1.]:
        u = h.get_ultracell(hs,emax=emax)
        e = eigh(u.get_hk_gen()(kS),eigvals_only=True)
        assert len(e)<len(e0)
        assert np.all(e>=e0[0:len(e)]-1e-10)
        if last is not None: # nested bases
            assert np.all(e[0:len(last)]<=last+1e-10)
        last = e
    # and the low-lying levels converge: with emax=1 the levels below -1.5
    # are within 1e-3 of the supercell
    m = e0<-1.5
    assert np.max(np.abs(last[0:np.sum(m)]-e0[m]))<1e-3


def test_cut_states_in_real_space_are_the_rayleigh_ritz_states():
    """With a cut the blocks of the folded momenta have different sizes. The
    states put back on the supercell orbitals must be orthonormal, and their
    energy in the supercell Hamiltonian must be their ultracell eigenvalue,
    which a wrong scatter of the coefficients breaks"""
    from pyqula.ultracelltk.matrix import eigenstates, real_space_states
    h,hs = modulated_honeycomb((4,3))
    u = h.get_ultracell(hs,emax=0.8)
    kS = np.array([0.31,0.17,0.])
    es,vs,U,valid = eigenstates(u.basis,u.hk(kS),kS)
    assert len(set(np.sum(valid,axis=1)))>1 # the blocks differ in size
    Phi = real_space_states(u.basis,kS,vs,U,valid)
    HS = dense(hs.get_hk_gen()(kS))
    assert np.max(np.abs(Phi.conj().T@Phi-np.identity(len(es))))<1e-10
    assert np.max(np.abs(np.real(np.sum(Phi.conj()*(HS@Phi),axis=0))-es))<1e-10


@pytest.mark.parametrize("model",["honeycomb","chain","cubic"])
def test_rebuilt_basis_diagonalizes_the_block_of_each_momentum(model):
    """The rebuilt basis comes from the hoppings of the supercell averaged
    over its cells. The block of each folded momentum on itself, the
    supercell projected on the orbital Bloch sums of that momentum, must
    have the same lowest levels and the same states, with modulated
    hoppings, an exchange field and spin-orbit coupling"""
    if model=="honeycomb": h,hs = modulated_honeycomb((4,3))
    elif model=="chain":
        h = geometry.chain().get_hamiltonian(has_spin=False)
        hs = h.get_supercell(5)
        hs.add_onsite(lambda r: 0.4*np.cos(2*np.pi*r[0]/5)+0.1)
        modulate_hoppings(hs,lambda r: 0.3*np.sin(2*np.pi*r[0]/5))
    else:
        h = geometry.cubic_lattice().get_hamiltonian(has_spin=False)
        hs = h.get_supercell([3,2,2])
        x = fractional(hs)
        hs.add_onsite(lambda r: 0.3*np.sin(2*np.pi*(x(r)[0]+x(r)[2])))
        modulate_hoppings(hs,lambda r: 0.2*np.cos(2*np.pi*x(r)[1]))
    u = h.get_ultracell(hs,emax=0.3,rebuild_basis=True)
    b = u.basis
    kS = np.array([0.31,0.17,0.43])
    E,U,valid = b.states(kS)
    assert 0<np.sum(valid)<b.N*b.norb # a cut
    hk = u.hk(kS)
    I = [np.identity(b.norb,dtype=np.complex128)]*b.N # every orbital
    for iq in range(b.N):
        B = b.basis_block(kS,I,iq) # orbital Bloch sums of this momentum
        heff = B.conj().T@np.asarray(hk@B)
        e = eigh(heff,eigvals_only=True)
        m = np.sum(valid[iq])
        if m==0: continue # nothing kept at this momentum
        assert np.max(np.abs(E[iq,0:m]-e[0:m]))<1e-12
        w = U[iq][:,0:m] # the kept states diagonalize the block
        assert np.max(np.abs(w.conj().T@heff@w-np.diag(E[iq,0:m])))<1e-12


def test_cut_observables_keep_their_sum_rules():
    """With a cut and a modulation, the LDOS summed over the sites is N
    times the DOS per unit cell; at zero modulation, the unfolded spectral
    function is the Lorentzians of the levels of the unit cell up to emax"""
    h,hs = modulated_honeycomb((4,3))
    u = h.get_ultracell(hs,emax=0.8)
    e,delta = 0.3,0.05
    l = u.get_ldos(e=e,delta=delta,nk=2,write=False)[2]
    d = u.get_dos(energies=np.array([e]),delta=delta,nk=2,write=False)[1][0]
    assert abs(np.sum(l)-12*d)<1e-10*abs(12*d)
    hc = h.get_supercell([4,3])
    u0 = h.get_ultracell(hc,emax=0.8)
    es = np.linspace(-3.,3.,80)
    kp = [[0.1,0.2,0.],[0.33,0.1,0.]]
    a = u0.get_kdos_bands(kpath=kp,energies=es,delta=0.05,write=False)[2]
    ref = []
    for k in kp:
        ek = np.linalg.eigvalsh(dense(h.get_hk_gen()(k)))
        ek = ek[ek<=0.8]
        ref.append(np.sum(0.05/((es[:,None]-ek[None,:])**2+0.05**2),axis=1)/np.pi)
    assert np.max(np.abs(a-np.concatenate(ref)))<1e-10


def test_expectation_values_need_the_occupied_states():
    h,hs = modulated_honeycomb((2,2))
    with pytest.raises(ValueError):
        h.get_ultracell(hs,emax=-0.5).get_vev(nk=1)


def test_helical_exchange_is_the_spin_spiral():
    """A helical exchange field of period n on the supercell against the
    spin-spiral (generalized Bloch) Hamiltonian of the unit cell: the
    spectrum at kS is the union of the spiral bands at the folded momenta
    shifted by half the spiral wavevector"""
    n,J = 7,0.6
    g = geometry.chain()
    h = g.get_hamiltonian()
    hs = h.get_supercell(n)
    hs.add_exchange(lambda r: [J*np.cos(2*np.pi*r[0]/n),
        J*np.sin(2*np.pi*r[0]/n),0.])
    u = h.get_ultracell(hs)
    hsp = h.copy()
    hsp.add_exchange([J,0.,0.])
    hsp.generate_spin_spiral(vector=[0.,0.,1.],qspiral=[1./n,0.,0.])
    hk = hsp.get_hk_gen()
    kS = 0.37
    e1 = eigh(u.get_hk_gen()([kS,0.,0.]),eigvals_only=True)
    e2 = np.sort(np.concatenate([np.linalg.eigvalsh(dense(hk(
        [(kS+q)/n+0.5/n,0.,0.]))) for q in range(n)]))
    assert np.max(np.abs(e1-e2))<1e-12


def test_harmonics_of_a_cosine():
    g = geometry.chain()
    h = g.get_hamiltonian(has_spin=False)
    n = 9
    hs = h.get_supercell(n)
    u = h.get_ultracell(hs)
    v = np.array([0.3*np.cos(2*np.pi*r[0]/n) + 0.1 for r in hs.geometry.r])
    Q,f = u.get_harmonics(v)
    q = np.round(Q[:,0]*n).astype(int)%n
    assert abs(f[q==0,0][0]-0.1)<1e-12
    # the supercell builder centers the positions, so the phase of a
    # harmonic is measured from the cell it calls R=0; the modulus is not
    assert abs(abs(f[q==1,0][0])-0.15)<1e-12
    assert abs(f[q==n-1,0][0]-np.conjugate(f[q==1,0][0]))<1e-12 # real f
    assert np.max(np.abs(f[(q!=0)&(q!=1)&(q!=n-1)]))<1e-12


def test_wrong_inputs_raise():
    g = geometry.honeycomb_lattice()
    h = g.get_hamiltonian()
    with pytest.raises(ValueError):
        h.get_ultracell(g.get_hamiltonian(has_spin=False).get_supercell(2))
    hs = h.get_supercell(2)
    with pytest.raises(TypeError):
        h.get_ultracell(hs,emax="high")
    gs = g.get_supercell([[2,1,0],[-1,1,0],[0,0,1]]) # rotated supercell
    hs = gs.get_hamiltonian() ; hs.remove_spin()
    with pytest.raises(ValueError): # not the same Hilbert space
        h.get_ultracell(hs)


@pytest.mark.parametrize("size",[3,8])
def test_states_in_chunks_give_the_same_observables(monkeypatch,size):
    """The states go to the supercell orbitals a chunk at a time, and the
    matrix is built for a group of momenta at a time, to bound the memory;
    with chunks of a few states, and groups of one or a few momenta, the
    spectrum, the local density of states, the expectation values and the
    density matrix must not change"""
    from pyqula.ultracelltk import matrix
    from pyqula.ultracelltk.meanfield import MeshStates, dm_full
    h,hs = modulated_honeycomb((3,2))
    u = h.get_ultracell(hs,emax=1.0)
    kS = np.array([0.31,0.17,0.])
    def compute():
        hm = u.hs.copy()
        hm.ultracell_basis = u.basis
        return (np.concatenate([eigh(u.get_hk_gen()(kS),eigvals_only=True),
                u.get_ldos(e=0.3,delta=0.05,nk=2,write=False)[2]]),
                u.get_vev(operator="sz",nk=2),
                dm_full(MeshStates(hm,nk=2),[(0,0,0),(1,0,0)],1e-7))
    l1,v1,d1 = compute()
    monkeypatch.setattr(matrix,"chunk_entries",size*hs.intra.shape[0])
    assert len(matrix.state_chunks(u.basis,10))==-(-10//size) # chunks
    E,U,valid = u.basis.states(kS)
    groups = matrix.momentum_groups(u.basis,np.sum(valid,axis=1))
    assert 1<len(groups)<u.basis.N or size==3 # groups of a few momenta
    l2,v2,d2 = compute()
    assert np.max(np.abs(l1-l2))<1e-12
    assert np.max(np.abs(v1-v2))<1e-12
    for d in d1: assert np.max(np.abs(d1[d]-d2[d]))<1e-12


NONDIAGONAL = [
    ("honeycomb",[[2,1,0],[-1,1,0],[0,0,1]]), # sqrt3 x sqrt3, the Kekule cell
    ("honeycomb",[[1,2,0],[-2,3,0],[0,0,1]]), # 7 cells, rotated
    ("triangular",[[3,1,0],[-1,2,0],[0,0,1]]), # 7 cells, P and Q not trivial
    ("triangular",[[2,0,0],[1,3,0],[0,0,1]]), # 6 cells, sheared
    ("chain",[[3,0,0],[0,1,0],[0,0,1]]),
    ]


def nondiagonal_model(name,M,spin=True):
    """A unit cell h and its supercell of matrix M, built from the geometry
    of the supercell, with the same Rashba coupling on both and a
    modulation on the supercell: an onsite potential and an exchange field
    periodic on it, and modulated hoppings"""
    g = {"honeycomb": geometry.honeycomb_lattice,
         "triangular": geometry.triangular_lattice,
         "chain": geometry.chain}[name]()
    gs = g.get_supercell(M,store_primal=True)
    h = g.get_hamiltonian(has_spin=spin)
    hs = gs.get_hamiltonian(has_spin=spin)
    if spin and g.dimensionality==2:
        h.add_rashba(0.3) ; hs.add_rashba(0.3)
    x = fractional(hs)
    d = hs.dimensionality
    def wave(r,a=0.):
        return np.cos(2*np.pi*np.sum(x(r)[0:d])+a)
    hs.add_onsite(lambda r: 0.3*wave(r) + 0.1*wave(r,1.))
    if spin: hs.add_exchange(lambda r: [0.2*wave(r,0.4),0.,0.1])
    modulate_hoppings(hs,lambda r: 0.2*wave(r,0.7))
    return h,hs


@pytest.mark.parametrize("name,M",NONDIAGONAL)
def test_nondiagonal_supercell_every_state(name,M):
    """A supercell along other vectors than those of the unit cell, of
    matrix M: with every state kept the ultracell is the supercell, at a
    generic kS, and with a cut a Rayleigh-Ritz compression of it"""
    h,hs = nondiagonal_model(name,M)
    u = h.get_ultracell(hs)
    assert u.basis.N==round(abs(np.linalg.det(np.array(M))))
    for kS in [np.array([0.13,0.37,0.]),np.array([0.,0.,0.])]:
        e1,e2 = spectra(u,hs,kS)
        assert np.max(np.abs(e1-e2))<1e-11
    u = h.get_ultracell(hs,emax=0.,rebuild_basis=True)
    e1 = eigh(u.get_hk_gen()(kS),eigvals_only=True)
    assert np.all(e1>=e2[0:len(e1)]-1e-10)


@pytest.mark.parametrize("name,M",NONDIAGONAL[0:3])
def test_nondiagonal_unfolded_weight_and_harmonics(name,M):
    """The unfolded weight of each state is that of operator="unfold" on
    the supercell over N, and a plane wave cos(2 pi Q.R) on the cells has
    its harmonics at Q and -Q only, for each folded momentum Q"""
    h,hs = nondiagonal_model(name,M,spin=False)
    u = h.get_ultracell(hs)
    b = u.basis
    p = np.array([0.21,0.08,0.])
    es = np.linspace(-4.,4.,80)
    a1 = u.get_kdos_bands(kpath=[p],energies=es,delta=0.05,write=False)[2]
    op = bloch_projector(hs,g0=h.geometry)
    kS = b.M@p # the momentum of the supercell that p folds onto
    a2 = hs.get_kdos_bands(kpath=[kS],operator=op,energies=es,delta=0.05,
            write=False)[2]
    assert np.max(np.abs(b.N*a1-a2))<1e-10*np.max(np.abs(a2))
    assert np.max(a2)>1. # the spectral function is not empty here
    Qs = b.folded_momenta(np.zeros(3))
    rep = np.array(hs.geometry.supercell_replica,dtype=float)
    for Q in Qs[1:]:
        f = np.cos(2*np.pi*rep@Q)
        Qs2,F = u.get_harmonics(f)
        same = [np.allclose(np.mod(Qp-Q+0.5,1.)-0.5,0.,atol=1e-9)
                for Qp in Qs2]
        opposite = [np.allclose(np.mod(Qp+Q+0.5,1.)-0.5,0.,atol=1e-9)
                for Qp in Qs2]
        mask = np.array(same)|np.array(opposite)
        assert np.max(np.abs(F[~mask]))<1e-12
        assert np.min(np.abs(F[np.array(same)]))>0.2 # half of cos


@pytest.mark.parametrize("name,M",NONDIAGONAL[0:3])
def test_nondiagonal_rebuilt_basis(name,M):
    """The rebuilt basis of a non-diagonal supercell diagonalizes the block
    of each folded momentum on itself, as in the diagonal case"""
    h,hs = nondiagonal_model(name,M)
    u = h.get_ultracell(hs,emax=0.3,rebuild_basis=True)
    b = u.basis
    kS = np.array([0.31,0.17,0.])
    E,U,valid = b.states(kS)
    hk = u.hk(kS)
    I = [np.identity(b.norb,dtype=np.complex128)]*b.N
    for iq in range(b.N):
        B = b.basis_block(kS,I,iq)
        heff = B.conj().T@np.asarray(hk@B)
        m = np.sum(valid[iq])
        if m==0: continue
        e = eigh(heff,eigvals_only=True)
        assert np.max(np.abs(E[iq,0:m]-e[0:m]))<1e-12
