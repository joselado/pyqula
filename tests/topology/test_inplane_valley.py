import numpy as np

from pyqula import geometry
from pyqula.operatortk.inplane_valley import _calibrate_taus


def _folded_dirac_subspace(h0):
    """Return an orthonormal basis for the 4-dimensional zero-energy
    subspace of a Kekule-commensurate honeycomb Hamiltonian at Gamma (the
    folded K,K' Dirac points)"""
    hk0 = h0.get_hk_gen()([0.,0.,0.])
    (es,vs) = np.linalg.eigh(hk0)
    inds = np.where(np.abs(es)<1e-6)[0]
    assert len(inds)==4
    return vs[:,inds]


def test_inplane_valley_pseudospin_algebra():
    """tau_x, tau_y (get_operator("valley_x"/"valley_y")) and tau_z
    (get_operator("valley")), projected onto the low-energy (folded
    K,K' Dirac point) subspace of a Kekule-commensurate honeycomb cell,
    must satisfy the expected Pauli pseudospin algebra: each squares to
    the identity, and any two of them anticommute."""
    g3 = geometry.honeycomb_lattice().supercell(3)
    h0 = g3.get_hamiltonian(has_spin=False)
    h0.turn_multicell()
    V = _folded_dirac_subspace(h0)

    opx = h0.get_operator("valley_x")
    opy = h0.get_operator("valley_y")
    opz = h0.get_operator("valley")
    k = [0.,0.,0.]
    Mx = V.conj().T@opx.m(None,k=k)@V
    My = V.conj().T@opy.m(None,k=k)@V
    Mz = V.conj().T@opz.m(None,k=k)@V

    I4 = np.eye(4)
    assert np.max(np.abs(Mx@Mx-I4))<1e-8
    assert np.max(np.abs(My@My-I4))<1e-8
    assert np.max(np.abs(Mz@Mz-I4))<1e-8
    assert np.max(np.abs(Mx@Mz+Mz@Mx))<1e-8
    assert np.max(np.abs(My@Mz+Mz@My))<1e-8
    assert np.max(np.abs(Mx@My+My@Mx))<1e-8


def test_kekule_gapped_states_are_valley_x_eigenstates():
    """Gapping the Dirac points (exactly at the folded K,K' point, k=Gamma
    of the tripled cell) with a chiral Kekule mass matching the tau_x
    calibration must produce the gap-edge eigenstates as exact +-1
    eigenvectors of the tau_x ("valley_x") operator."""
    (t1x,t2x,t1y,t2y) = _calibrate_taus()
    g3 = geometry.honeycomb_lattice().supercell(3)
    h = g3.get_hamiltonian(has_spin=False)
    h = h.get_multicell()
    h.add_chiral_kekule(t1=0.4*t1x,t2=0.4*t2x)
    hk = h.get_hk_gen()([0.,0.,0.])
    (es,vs) = np.linalg.eigh(hk)
    near_gap = np.abs(np.abs(es)-0.4)<1e-6 # the induced gap is exactly 0.4
    assert np.sum(near_gap)==4 # the 4 folded Dirac states

    opx = h.get_operator("valley_x")
    Mx = opx.m(None,k=[0.,0.,0.])
    for i in np.where(near_gap)[0]:
        w = vs[:,i]
        ev = np.real(np.conj(w)@Mx@w) # expectation value, should be +-1
        assert np.isclose(abs(ev),1.0,atol=1e-6)


def test_add_valley_exchange_matches_operator_convention():
    """add_valley_exchange([vx,vy,vz]) must open a gap of exactly
    |v|=sqrt(vx^2+vy^2+vz^2) at the folded Dirac point, with the same
    (tau_x,tau_y,tau_z) normalization as get_operator("valley_x"/
    "valley_y"/"valley"), for an arbitrary (not just axis-aligned)
    direction in valley pseudospin space."""
    v = np.array([0.1,0.15,0.25])
    g3 = geometry.honeycomb_lattice().supercell(3)
    h = g3.get_hamiltonian(has_spin=False)
    h.add_valley_exchange(list(v))
    h.turn_multicell()
    es = np.linalg.eigvalsh(h.get_hk_gen()([0.,0.,0.]))
    gap = np.linalg.norm(v)
    near_gap = np.abs(np.abs(es)-gap)<1e-6
    assert np.sum(near_gap)==4 # the 4 folded Dirac states
