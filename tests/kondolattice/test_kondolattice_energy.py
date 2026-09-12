import numpy as np

from pyqula import geometry
from pyqula.filling import eigenvalues
from pyqula.kondolattice import KondoLatticeHamiltonian
from pyqula.multihopping import MultiHopping
from pyqula.scftk.kondolattice import hs_constant

# see test_kondolattice.py's header for why filling=0.15

J, FILLING, NK, T, Q = 1.5, 0.15, 120, 2e-2, 1.0


def test_grand_potential_is_stationary_in_the_hybridization():
    """Hellmann-Feynman: the mean-field grand potential

        Omega(V) = -T sum_n ln(1+exp(-(e_n(V)-mu)/T)) + N|V|^2/J
                   - sum_j lam_j Q

    must be STATIONARY in V at the fixed point the SCF's own update
    equation converges to -- that is what makes the fixed point a saddle
    point of the Hubbard-Stratonovich action rather than an arbitrary
    self-consistency. It is an oracle independent of this code: it fixes
    the constant that multiplies |V|^2 without reference to how the
    constant is written down anywhere.

    The energy the SCF reports carried |V|^2/J instead of N|V|^2/J, which
    is stationary nowhere."""
    gc = geometry.chain()
    K = KondoLatticeHamiltonian(gc.get_hamiltonian(has_spin=True))
    scf = K.get_mean_field_hamiltonian(J=J, filling=FILLING, nk=NK,
            mf=(np.array([0.3+0.0j]), np.array([0.0])), mix=0.3,
            maxerror=1e-8, maxite=5000)
    assert scf is not None, "SCF did not converge"
    V = scf.hybridization.real.copy()
    lam = scf.constraint_lambda.copy()
    assert abs(V[0]) > 1e-2, V # genuinely on the Kondo branch, not V=0
    pairs = K._kondo_pairs
    h1 = K.get_dense()
    hop0 = h1.get_dict()
    mu = h1.get_fermi4filling(FILLING, nk=NK) # as the SCF fixes it

    def omega(v):
        """The same grand potential at an arbitrary V, same lam and mu.

        The band part is the finite-T free energy, not a sum over the
        states below mu: the SCF's own fixed point is defined by a
        Fermi-Dirac-weighted density matrix at this T, and it is the free
        energy whose V-derivative is that same weighted <dH/dV>."""
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

    d = 1e-3
    op, o0, om = omega(V+d), omega(V), omega(V-d)
    first = (op-om)/(2*d)
    second = (op-2*o0+om)/d**2
    # the probe is sensitive to V at all (otherwise the test is vacuous)
    assert abs(second) > 1e-2, second
    assert abs(first) < 1e-3, first
