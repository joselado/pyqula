import numpy as np

from pyqula import geometry
from pyqula.filling import eigenvalues
from pyqula.kondolattice import KondoLatticeHamiltonian
from pyqula.multihopping import MultiHopping
from pyqula.scftk.kondolattice import hs_constant

# see test_kondolattice.py's header for why filling=0.15

J, FILLING, NK, T, Q = 1.5, 0.15, 120, 2e-2, 1.0
SEED = (np.array([0.3+0.0j]), np.array([0.0]))


def kondo_chain():
    gc = geometry.chain()
    return KondoLatticeHamiltonian(gc.get_hamiltonian(has_spin=True))


def grand_potential(K, v, lam, mu):
    """The mean-field grand potential per cell of the onsite Kondo chain K,

        Omega(V) = -T (1/Nk) sum_kn ln(1+exp(-(e_kn(V)-mu)/T))
                   + N|V|^2/J - sum_j lam_j Q ,

    at the fields v, the multipliers lam and the chemical potential mu,
    written from the definition.

    The band part is the finite-T free energy, not a sum over the states
    below mu: the SCF's own fixed point is defined by a Fermi-Dirac-weighted
    density matrix at this T, and it is the free energy whose V-derivative
    is that same weighted <dH/dV>."""
    pairs = K._kondo_pairs
    h1 = K.get_dense()
    hop0 = h1.get_dict()
    m = np.zeros(h1.intra.shape, dtype=np.complex128)
    for idx, (ci, fi) in enumerate(pairs):
        for s in (0, 1): # spin components
            cc, ff = 2*ci+s, 2*fi+s
            m[ff, ff] += lam[idx]
            m[cc, ff] += np.conjugate(v[idx])
            m[ff, cc] += v[idx]
    h = h1.copy()
    h.set_multihopping(MultiHopping(hop0)+MultiHopping({(0, 0, 0): m}))
    h.shift_fermi(-mu)
    es = eigenvalues(h, nk=NK) # every eigenvalue of the k-mesh
    nkp = len(es)//h.intra.shape[0] # number of kpoints
    band = -T*np.sum(np.logaddexp(0., -es/T))/nkp
    return band + hs_constant(v, J) - np.sum(lam)*Q


def test_grand_potential_is_stationary_in_the_hybridization():
    """Hellmann-Feynman: the mean-field grand potential must be
    STATIONARY in V at the fixed point the SCF's own update equation
    converges to -- that is what makes the fixed point a saddle point of
    the Hubbard-Stratonovich action rather than an arbitrary
    self-consistency. It is an oracle independent of this code: it fixes
    the constant that multiplies |V|^2 without reference to how the
    constant is written down anywhere.

    The energy the SCF reports carried |V|^2/J instead of N|V|^2/J, which
    is stationary nowhere."""
    K = kondo_chain()
    scf = K.get_mean_field_hamiltonian(J=J, filling=FILLING, nk=NK,
            mf=SEED, mix=0.3, maxerror=1e-8, maxite=5000)
    assert scf is not None, "SCF did not converge"
    V = scf.hybridization.real.copy()
    lam = scf.constraint_lambda.copy()
    assert abs(V[0]) > 1e-2, V # genuinely on the Kondo branch, not V=0
    mu = K.get_dense().get_fermi4filling(FILLING, nk=NK) # as the SCF fixes it
    d = 1e-3
    op, o0, om = [grand_potential(K, V+x, lam, mu) for x in (d, 0., -d)]
    first = (op-om)/(2*d)
    second = (op-2*o0+om)/d**2
    # the probe is sensitive to V at all (otherwise the test is vacuous)
    assert abs(second) > 1e-2, second
    assert abs(first) < 1e-3, first


def test_total_energy_is_the_grand_potential():
    """The total energy of the loop is the grand potential above, at the
    smearing T and the chemical potential of the loop and at the
    converged fields. It used to be the sum of the eigenvalues below mu, a
    band energy at T=0, plus mu times the electron count at T, which is
    neither the energy at T=0 nor the grand potential"""
    K = kondo_chain()
    h2, e = K.get_mean_field_hamiltonian(J=J, filling=FILLING, nk=NK, T=T,
            mf=SEED, mix=0.3, maxerror=1e-10, maxite=5000,
            return_total_energy=True)
    assert abs(h2.hybridization[0]) > 0.1 # the Kondo branch
    mu = K.get_dense().get_fermi4filling(FILLING, nk=NK) # as the SCF fixes it
    o = grand_potential(K, h2.hybridization, h2.constraint_lambda, mu)
    assert abs(e - o) < 1e-10, (e, o)


def fixed_mu(mu):
    """A get_fermi4filling that gives mu whatever the filling"""
    def f(self, filling, **kwargs): return mu
    return f


def test_grand_potential_falls_with_mu_by_the_electron_count(monkeypatch):
    """Thermodynamics: at fixed T the grand potential of the converged
    state changes with the chemical potential as dOmega/dmu = -N, with N
    the electron count per cell, c and f together. Omega is stationary in
    V at the fixed point, and in lam once n_f = Q, so along the converged
    solutions only the explicit mu of the bands counts, and it gives
    -(1/Nk) sum_kn f(e_kn - mu). The loop takes mu from the filling of the
    bare bands, which moves in steps of a level spacing, so the test sets
    mu directly. The old energy, a T=0 band energy plus mu times N, does
    not change with mu like this"""
    K = kondo_chain()
    mu0 = K.get_dense().get_fermi4filling(FILLING, nk=NK)
    d = 1e-5
    seed, out = SEED, []
    for mu in (mu0 - d, mu0 + d):
        monkeypatch.setattr(KondoLatticeHamiltonian, "get_fermi4filling",
                fixed_mu(mu))
        h2, e = K.get_mean_field_hamiltonian(J=J, filling=FILLING, nk=NK,
                T=T, mf=seed, mix=0.3, maxerror=1e-10, maxite=5000,
                return_total_energy=True)
        assert abs(h2.hybridization[0]) > 0.1 # the Kondo branch
        n = np.trace(h2.get_density_matrix(nk=NK, T=T)).real
        out.append((e, n))
        seed = (h2.hybridization, h2.constraint_lambda) # the same branch
    monkeypatch.undo()
    (e0, n0), (e1, n1) = out
    slope = (e1 - e0)/(2*d)
    assert abs(slope + (n0 + n1)/2.) < 1e-6, (slope, n0, n1)
