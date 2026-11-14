import numpy as np

from pyqula import geometry
from pyqula.kondolattice import KondoLatticeHamiltonian
from pyqula.multihopping import MultiHopping

# The mean-field term is V_j f^dag_j c_j + h.c., so the self-consistency
# equation is V_j = -(J/2) sum_s <c^dag_js f_js>, the stationary point of
# the energy in conj(V_j). The loop used the element of the density matrix
# with the two indices exchanged, whose target is conj(V_j). For a real V
# that is the same number, so every test with a real seed passed. Two cases
# tell them apart: a complex seed, whose phase is the phase of the f
# fermion and must stay, and a complex hopping between the moments, where
# the phases of V cannot be removed and conj(V) is not a solution.


def order_parameter(h, J, nk, T):
    """-(J/2) sum_s <c^dag_js f_js> of every Kondo pair, from the
    eigenstates of h on the k-mesh, with no density matrix routine"""
    hk, ks = h.get_hk_gen(), h.geometry.get_kmesh(nk=nk)
    out = np.zeros(len(h._kondo_pairs), dtype=np.complex128)
    for k in ks:
        e, u = np.linalg.eigh(np.array(hk(k)))
        occ = 1./(np.exp(e/T) + 1.)
        for ip, (ci, fi) in enumerate(h._kondo_pairs):
            for s in (0, 1):
                out[ip] += np.sum(occ*np.conj(u[2*ci+s])*u[2*fi+s])
    return -(J/2.)*out/len(ks)


def test_complex_seed_keeps_its_phase():
    """A phase on the f fermion of a site, f -> e^{-i a} f, turns V into
    e^{i a} V and changes nothing else. So a seed with a phase must
    converge to the solution of the real seed times that phase, with the
    same energy."""
    h = KondoLatticeHamiltonian(geometry.chain().get_hamiltonian(
            has_spin=True))
    kw = dict(J=1.5, filling=0.15, nk=40, T=0.02, return_total_energy=True)
    h0, e0 = h.get_mean_field_hamiltonian(mf=([0.3], [0.0]), **kw)
    phase = np.exp(0.7j)
    h1, e1 = h.get_mean_field_hamiltonian(mf=([0.3*phase], [0.0]), **kw)
    V0, V1 = h0.hybridization[0], h1.hybridization[0]
    assert abs(V0.imag) < 1e-10 and V0.real > 0.3, V0
    assert abs(V1 - phase*V0) < 1e-6, (V1, phase*V0)
    assert abs(e1 - e0) < 1e-8, (e0, e1)
    # the converged V is the order parameter of the converged state
    op = order_parameter(h1, 1.5, 40, 0.02)
    assert abs(op[0] - V1) < 1e-4, (op, V1)


def moments_with_flux(phi, tf=0.2):
    """A chain of two Kondo pairs per cell with a hopping between the
    moments, tf e^{i phi} inside the cell and tf between cells. The two
    plaquettes c0-c1-f1-f0 of the chain then hold different fluxes, which
    no phase of the f fermions removes."""
    hc = geometry.chain().get_supercell(2).get_hamiltonian(has_spin=True)
    h = KondoLatticeHamiltonian(hc)
    d = {k: np.array(m, dtype=np.complex128)
         for k, m in h.get_multihopping().get_dict().items()}
    (c0, f0), (c1, f1) = h._kondo_pairs
    for s in (0, 1):
        d[(0, 0, 0)][2*f0+s, 2*f1+s] += tf*np.exp(1j*phi)
        d[(0, 0, 0)][2*f1+s, 2*f0+s] += tf*np.exp(-1j*phi)
        d[(1, 0, 0)][2*f1+s, 2*f0+s] += tf
        d[(-1, 0, 0)][2*f0+s, 2*f1+s] += tf
    h.set_multihopping(MultiHopping(d))
    return h


def test_hybridization_with_a_complex_hopping_between_the_moments():
    """With a flux between the moments the relative phase of the two V is
    physical. The converged V must be the order parameter of the converged
    state, and two seeds with different phases must agree on the energy
    and on V_1 conj(V_0), which no phase of the f fermions changes."""
    J, nk, T = 2.0, 30, 0.02
    h = moments_with_flux(0.9)
    hk = np.array(h.get_hk_gen()([0.2, 0., 0.]))
    assert np.max(np.abs(hk - hk.conj().T)) < 1e-14
    out = []
    for seed in ([0.5, 0.5], [0.5*np.exp(0.4j), 0.5*np.exp(-0.2j)]):
        h2, e = h.get_mean_field_hamiltonian(J=J, filling=0.15, nk=nk, T=T,
                mf=(seed, [0.0, 0.0]), return_total_energy=True)
        V = h2.hybridization
        assert np.max(np.abs(h2.local_occupation - 1.)) < 1e-4
        op = order_parameter(h2, J, nk, T)
        assert np.max(np.abs(op - V)) < 1e-4, (op, V)
        out.append((V[1]*np.conj(V[0]), e))
    (p0, e0), (p1, e1) = out
    assert abs(p0 - p1) < 1e-4, (p0, p1)
    assert abs(e0 - e1) < 1e-4, (e0, e1)
    # the relative phase is neither 0 nor pi: V cannot be made real
    assert abs(p0.imag) > 0.05, p0
