# The LDOS weighted by an operator O is the local matrix element: state n
# puts Re[conj(u_n(a)) (O u_n)(a)] on orbital a, summed over the orbitals
# of each site, in every mode and integration of h.get_ldos() and
# h.get_multildos(). The diagonalization modes used the product
# <u_n|O|u_n> |u_n(i)|^2 instead, which is quadratic in the projector of
# one state, so inside a degenerate multiplet it depended on the basis the
# diagonalization returned: on a Neel state a seed field of 1e-10 along z
# gave a staggered sz map of 0.27 and the same seed along x gave 7e-5.
# Every test here asserts an invariant (a basis rotation, a seed field in
# two directions, the Green's function route, a numpy reference, a sum
# rule), never a recorded number.
import numpy as np
import pytest

from pyqula import geometry
from pyqula import algebra
from pyqula import klist
from pyqula.increase_hilbert import full2profile
from pyqula.ldostk.ldoswaves import ldos_waves_from_eigsystem


def _neel():
    h = geometry.honeycomb_lattice().get_hamiltonian(has_spin=True)
    h.add_antiferromagnetism(0.5) # PT symmetric, every band is a pair
    return h


def _rashba():
    """spin-orbit coupling and a tilted exchange field, so that no
    eigenstate is an eigenstate of sx, sy or sz"""
    h = geometry.honeycomb_lattice().get_hamiltonian(has_spin=True)
    h.add_rashba(0.3)
    h.add_exchange([0.1,0.2,0.3])
    h.add_sublattice_imbalance(0.2)
    return h


def _hermitian(n,seed):
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(n,n)) + 1j*rng.normal(size=(n,n))
    return a + a.T.conj()


def _ldos(h,**kwargs):
    return np.array(h.get_ldos(nrep=1,write=False,**kwargs)[2])


def _lorentzian(e,es,delta):
    return delta/((e-es)**2+delta**2)/np.pi


def test_the_weight_does_not_depend_on_the_basis_of_a_multiplet():
    """Rotating each degenerate pair of the Neel state with a random U(2)
    gives another eigenbasis of the same H(k), and the map stays the
    same, for sz and for a random Hermitian matrix"""
    h = _neel()
    k = np.array([0.137,0.291,0.])
    es,vs = np.linalg.eigh(algebra.todense(h.get_hk_gen()(k)))
    assert np.max(np.abs(es[1::2]-es[0::2])) < 1e-10 # degenerate pairs
    rng = np.random.default_rng(7)
    ws = vs.copy()
    for i in range(0,len(es),2):
        q,_ = np.linalg.qr(rng.normal(size=(2,2))+1j*rng.normal(size=(2,2)))
        ws[:,i:i+2] = vs[:,i:i+2]@q
    energies = np.array([-1.3,0.9,1.6])
    for op in ["sz",_hermitian(4,seed=1)]:
        op = h.get_operator(op)
        d1 = ldos_waves_from_eigsystem(es,vs.T,energies,0.1,operator=op,k=k)
        d2 = ldos_waves_from_eigsystem(es,ws.T,energies,0.1,operator=op,k=k)
        assert np.max(np.abs(d1)) > 1e-2
        assert np.max(np.abs(d1-d2)) < 1e-12*np.max(np.abs(d1))
    # the sz map is staggered and not zero, so the check is not vacuous
    d = ldos_waves_from_eigsystem(es,ws.T,energies,0.1,
            operator=h.get_operator("sz"),k=k)
    d = np.array([full2profile(h,x) for x in d]) # (energies, sites)
    assert np.max(np.abs(d)) > 1e-2
    assert np.max(np.abs(d[:,0]+d[:,1])) < 1e-12*np.max(np.abs(d))


def _seeded(d):
    h = _neel()
    h.add_exchange(1e-10*np.array(d)) # far below _degeneracy_tol
    return h


def _check_seed_independent(a,b,tol):
    assert np.max(np.abs(a)) > 0.1 # a staggered spin polarization
    assert abs(a[0]+a[1]) < tol*np.max(np.abs(a))
    assert np.max(np.abs(a-b)) < tol*np.max(np.abs(a))


@pytest.mark.parametrize("integration",["grid","triangle"])
def test_a_seed_field_does_not_pick_the_map(integration):
    """A seed exchange of 1e-10 along z or along x only picks the basis
    inside each degenerate pair of the Neel state, so the sz map is the
    same with both"""
    a,b = [_ldos(_seeded(d),e=1.2,delta=0.05,nk=30,integration=integration,
            operator="sz") for d in ([0.,0.,1.],[1.,0.,0.])]
    _check_seed_independent(a,b,1e-7)


def test_a_seed_field_does_not_pick_the_qtci_map():
    a,b = [_ldos(_seeded(d),e=1.2,delta=0.05,nk=32,integration="qtci",
            operator="sz",tolerance=1e-10)
            for d in ([0.,0.,1.],[1.,0.,0.])]
    _check_seed_independent(a,b,1e-6)


def test_a_seed_field_does_not_pick_the_multildos_map():
    a,b = [_seeded(d).get_multildos(energies=np.array([1.2]),delta=0.05,
            nk=30,operator="sz",nrep=1,write=False)[3][0]
            for d in ([0.,0.,1.],[1.,0.,0.])]
    _check_seed_independent(np.array(a),np.array(b),1e-7)


def test_the_diagonalization_agrees_with_the_green_function():
    """mode='green' is -Im diag(G O + O G)/(2 pi), a property of the
    Green's function, and mode='arpack' on the same k-mesh gives the same
    map for operators that do not commute with H"""
    h = _rashba()
    args = dict(e=0.4,delta=0.1,nk=30)
    for op in ["sx","sz",_hermitian(4,seed=2)]:
        g = _ldos(h,mode="green",operator=op,**args)
        a = _ldos(h,mode="arpack",operator=op,**args)
        assert np.max(np.abs(g)) > 1e-3
        assert np.max(np.abs(g-a)) < 1e-6*np.max(np.abs(g))


def _spin_maps(h,e,delta,nk):
    """the local spin density sum_n L(e-E_n) <u_n|P_i sz|u_n> and the
    product sum_n L(e-E_n) <u_n|sz|u_n> |u_n(i)|^2, from numpy"""
    sz = np.diag(algebra.todense(h.get_operator("sz").get_matrix())).real
    local,product = 0.,0.
    ks = klist.kmesh(h.dimensionality,nk=nk)
    for k in ks:
        es,vs = np.linalg.eigh(algebra.todense(h.get_hk_gen()(k)))
        rho = np.abs(vs)**2 # (orbital, state)
        L = _lorentzian(e,es,delta)
        local = local + (rho*sz[:,None])@L
        product = product + rho@(L*(sz@rho))
    return full2profile(h,local/len(ks)),full2profile(h,product/len(ks))


def test_the_sz_map_is_the_local_spin_density():
    """With spin-orbit coupling the sz map is the local spin density
    |u(i,up)|^2-|u(i,dn)|^2, which differs from the old product"""
    h = _rashba()
    got = _ldos(h,e=0.4,delta=0.1,nk=12,mode="arpack",operator="sz")
    local,product = _spin_maps(h,0.4,0.1,12)
    assert np.max(np.abs(local)) > 1e-3
    assert np.max(np.abs(got-local)) < 1e-10*np.max(np.abs(local))
    assert np.max(np.abs(local-product)) > 1e-3*np.max(np.abs(local))


def test_the_old_value_is_kept_for_eigenstates_of_the_operator():
    """On a lattice polarized along z the states are eigenstates of sz,
    and the local spin density is the old product"""
    h = geometry.square_lattice().get_hamiltonian(has_spin=True)
    h.add_zeeman([0.,0.,0.4])
    got = _ldos(h,e=0.3,delta=0.1,nk=12,mode="arpack",operator="sz")
    local,product = _spin_maps(h,0.3,0.1,12)
    assert np.max(np.abs(local)) > 1e-3
    assert np.max(np.abs(local-product)) < 1e-10*np.max(np.abs(local))
    assert np.max(np.abs(got-local)) < 1e-10*np.max(np.abs(local))


@pytest.mark.parametrize("delta",[0.,0.05])
def test_the_triangle_map_sums_to_the_operator_resolved_dos(delta):
    """The local elements of a state add up to <u_n|O|u_n> over the sites,
    so the triangle map summed over the sites is the triangle DOS
    weighted by the operator, on the same mesh, to rounding"""
    h = _rashba()
    m = _hermitian(4,seed=3)
    for e in [-1.1,0.35,1.7]:
        d = _ldos(h,e=e,delta=delta,nk=30,integration="triangle",operator=m)
        _,ref = h.get_dos(energies=[e],delta=delta,nk=30,
                integration="triangle",operator=m,write=False)
        assert abs(ref[0]) > 1e-3
        assert abs(np.sum(d)-ref[0]) < 1e-10*np.max(np.abs(d))


def test_the_valley_map_sums_to_the_valley_dos():
    """In grid mode a k-dependent operator works through its action on a
    state, and the map summed over the sites is sum_n L(E-E_n)
    <u_n|O(k)|u_n>, for the valley operator of a zigzag ribbon. The
    sublattice imbalance and the magnetic field break the symmetries that
    exchange the two valleys, so that sum is not zero"""
    h = geometry.honeycomb_zigzag_ribbon(4).get_hamiltonian()
    h.add_peierls(0.05)
    h.add_sublattice_imbalance(0.2)
    op = h.get_operator("valley")
    e,delta,nk = 1.0,0.05,6
    got = _ldos(h,e=e,delta=delta,nk=nk,mode="arpack",operator=op)
    ref = 0.
    ks = klist.kmesh(1,nk=nk)
    for k in ks:
        es,vs = np.linalg.eigh(algebra.todense(h.get_hk_gen()(k)))
        w = np.array([np.real(op.braket(v,k=k)) for v in vs.T])
        ref += np.sum(_lorentzian(e,es,delta)*w)
    ref /= len(ks)
    assert abs(ref) > 1e-3
    assert abs(np.sum(got)-ref) < 1e-10*abs(ref)


def test_the_sparse_multildos_uses_the_local_element():
    """get_multildos of a sparse Hamiltonian keeps the num_bands states
    closest to the energy at each k, and weights them with the local
    element too: the sz map is the local spin density of those states,
    computed here from a dense diagonalization"""
    h = geometry.honeycomb_zigzag_ribbon(4).get_hamiltonian(has_spin=True)
    h.add_rashba(0.3)
    h.add_exchange([0.1,0.2,0.3])
    h.turn_sparse()
    e,delta,nk,nb = 0.3,0.1,4,8
    got = h.get_multildos(energies=np.array([e]),delta=delta,nk=nk,
            num_bands=nb,operator="sz",nrep=1,write=False)[3][0]
    sz = np.diag(algebra.todense(h.get_operator("sz").get_matrix())).real
    ref = 0.
    ks = klist.kmesh(1,nk=nk)
    for k in ks:
        es,vs = np.linalg.eigh(algebra.todense(h.get_hk_gen()(k)))
        keep = np.argsort(np.abs(es-e))[:nb] # the states ARPACK returns
        rho = np.abs(vs[:,keep])**2
        ref = ref + (rho*sz[:,None])@_lorentzian(e,es[keep],delta)
    ref = full2profile(h,ref/len(ks))
    assert np.max(np.abs(ref)) > 1e-3
    assert np.max(np.abs(np.array(got)-ref)) < 1e-8*np.max(np.abs(ref))
