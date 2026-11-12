import numpy as np
import pytest

from pyqula import geometry
from pyqula.spinon import SpinonHamiltonian, get_decoupling_names

# The bond decoupling of SpinonHamiltonian (decoupling="bond") and the
# spin-channel RPA on top of it follow Willsher and Knolle,
# arXiv:2503.13831: the spinon hopping is t_ij = J_ij chi_ij / 2 with
# chi_ij = sum_s <f^dag_is f_js>, there is no magnetic mean field, and the
# spin response is chi0 (1 + J(q) chi0)^-1. These tests check the mean
# field against closed forms, the response against the same formula
# written by hand, and both against the numbers of that paper for the
# pi-flux state of the triangular lattice.


def bond_field(g, **kwargs):
    kw = dict(decoupling="bond", mix=0.5, maxerror=1e-7, T=0.01)
    kw.update(kwargs)
    return SpinonHamiltonian(g).get_mean_field_hamiltonian(**kw)


def hoppings(h, tol=1e-6):
    """Moduli of the nonzero hoppings between different sites"""
    out = []
    for d, m in h.get_multihopping().get_dict().items():
        m = np.array(m)[0::2, 0::2]
        if tuple(int(i) for i in d) == (0, 0, 0):
            m = m - np.diag(np.diag(m))
        out += [abs(x) for x in m.ravel() if abs(x) > tol]
    return np.array(out)


def pi_flux_ansatz(g0, g):
    """The pi-flux pattern of the triangular lattice g0 on its (2,1)
    supercell g: flux pi through the up triangles and zero through the
    down ones. With p the coordinate of a site along a1, a bond along a1
    is +1 and a bond that changes the coordinate along a2 is (-1)^p of
    its end of lower a2."""
    A = np.array([g0.a1[:2], g0.a2[:2]]).T
    def pq(r): return np.round(np.linalg.solve(A, (r - g.r[0])[:2])).astype(int)
    out = {}
    for n1 in (-1, 0, 1):
        for n2 in (-1, 0, 1):
            m = np.zeros((4, 4), dtype=np.complex128)
            for i in range(2):
                for j in range(2):
                    rj = g.r[j] + n1*g.a1 + n2*g.a2
                    if abs(np.linalg.norm(rj - g.r[i]) - 1.) > 1e-6: continue
                    (pi, qi), (pj, qj) = pq(g.r[i]), pq(rj)
                    if qi == qj: s = 1.
                    else: s = (-1.)**(pi if qi < qj else pj)
                    m[2*i, 2*j] = m[2*i+1, 2*j+1] = s
            out[(n1, n2, 0)] = m
    return out


@pytest.fixture(scope="module")
def triangular():
    g0 = geometry.triangular_lattice()
    g = g0.get_supercell((2, 1))
    return g, pi_flux_ansatz(g0, g)


_dirac = {} # the converged states, shared by the tests below


def dirac(triangular, J2=0.0, gutzwiller=1.0):
    """The pi-flux state held by ansatz=. Its temperature keeps the two
    sites equivalent on a mesh that contains the Dirac points."""
    g, ansatz = triangular
    if (J2, gutzwiller) not in _dirac:
        _dirac[(J2, gutzwiller)] = bond_field(g, J1=1.0, J2=J2,
                ansatz=ansatz, nk=20, maxerror=1e-6, gutzwiller=gutzwiller,
                T=0.01*gutzwiller, return_total_energy=True)
    return _dirac[(J2, gutzwiller)]


K, M = [2./3., 1./3., 0.], [0., 0.5, 0.] # of the lattice, in the supercell


def test_chain_hopping_is_J_over_pi():
    """On the chain chi = 2/pi, so the bond decoupling gives a hopping
    J/pi, where the Hartree-Fock one gives 3J/(4 pi)"""
    np.random.seed(0)
    h = bond_field(geometry.chain(), J1=1.0, nk=400)
    assert np.allclose(hoppings(h), 1./np.pi, atol=1e-3), hoppings(h)
    assert np.allclose(h.local_occupation, 1.0, atol=1e-5)
    np.random.seed(0)
    hf = SpinonHamiltonian(geometry.chain()).get_mean_field_hamiltonian(
            J1=1.0, nk=24, mix=0.3, maxerror=1e-5, maxite=2000)
    assert np.allclose(hoppings(hf), 3./(4.*np.pi), atol=2e-3), hoppings(hf)


def test_gutzwiller_scales_the_hopping():
    np.random.seed(0)
    h = bond_field(geometry.chain(), J1=1.0, nk=400, gutzwiller=2.0, T=0.02)
    assert np.allclose(hoppings(h), 2./np.pi, atol=2e-3), hoppings(h)
    assert h.gutzwiller == 2.0


def test_bond_decoupling_has_no_magnetic_channel():
    """With two sites in the cell the Hartree-Fock loop goes to the Neel
    state. The bond loop cannot: it stays a singlet, and from a random
    guess it finds the dimerized state, the mean-field ground state of the
    bond functional, with energy -J/4 per site."""
    g = geometry.chain().get_supercell(2)
    np.random.seed(1)
    h, e = bond_field(g, J1=1.0, nk=100, maxerror=1e-6,
                      return_total_energy=True)
    assert np.max(np.abs(h.get_magnetization())) < 1e-8
    assert abs(e/2. + 0.25) < 1e-4, e/2.
    np.random.seed(1)
    hf = SpinonHamiltonian(g).get_mean_field_hamiltonian(J1=1.0, nk=100,
            mix=0.3, maxerror=1e-6)
    assert np.max(np.abs(hf.get_magnetization())) > 0.5


def test_ansatz_holds_the_uniform_chain():
    """The same cell with the uniform pattern as ansatz keeps it, with the
    hopping and the energy -2J/pi^2 per site of the one-site cell"""
    g = geometry.chain().get_supercell(2)
    h, e = bond_field(g, J1=1.0, nk=200, return_total_energy=True,
                      ansatz=g.get_hamiltonian(has_spin=True))
    assert np.allclose(hoppings(h), 1./np.pi, atol=1e-3), hoppings(h)
    assert abs(e/2. + 2./np.pi**2) < 1e-4, e/2.


def test_rpa_is_the_bubble_dressed_with_Jq():
    """On the chain every matrix is a number, so the dressed response can
    be written by hand: chi0/(1 - J(q) chi0) with J(q) = 2 g J cos(q), in
    the sign convention of the library. The three components are equal
    and uncoupled in a singlet."""
    np.random.seed(0)
    gz, q = 2.0, 0.2
    h = bond_field(geometry.chain(), J1=0.7, nk=200, gutzwiller=gz)
    kw = dict(q=[q, 0., 0.], energies=np.linspace(0., 2., 9), delta=0.05,
              nk=61)
    es, c0 = h.get_spinchi_full(RPA=False, **kw)
    es, c = h.get_spinchi_full(**kw)
    Jq = 2.*gz*0.7*np.cos(2.*np.pi*q)
    assert np.max(np.abs(c0[:, 2, 2])) > 0.1
    assert np.allclose(c[:, 2, 2], c0[:, 2, 2]/(1. - Jq*c0[:, 2, 2]),
                       atol=1e-10)
    assert np.allclose(c[:, 0, 0], c[:, 2, 2], atol=1e-12)
    assert np.allclose(c[:, 0, 1], 0., atol=1e-12)
    # the transverse response S+ S- is twice the longitudinal one
    es, cl = h.get_spinchi_ladder(**kw)
    assert np.allclose(np.array(cl).reshape(len(es)), 2.*c[:, 2, 2],
                       atol=1e-10)


def test_local_response_does_not_depend_on_the_cell():
    """The dressed local response of the uniform chain is the same in the
    one-site and in the two-site cell on equivalent meshes, which checks
    the Fourier transform of the vertex"""
    g1 = geometry.chain()
    g2 = g1.get_supercell(2)
    h1 = bond_field(g1, J1=1.0, nk=200, T=0.1,
                    ansatz=g1.get_hamiltonian(has_spin=True))
    h2 = bond_field(g2, J1=1.0, nk=100, T=0.1,
                    ansatz=g2.get_hamiltonian(has_spin=True))
    kw = dict(energies=np.linspace(0.3, 2., 5), delta=0.1)
    c1 = h1.get_spinchi_full(nk=20, **kw)[1]
    c2 = h2.get_spinchi_full(nk=10, **kw)[1]
    assert np.max(np.abs(c1[:, 2, 2])) > 0.05
    for i in (0, 1):
        assert np.allclose(c2[:, 4+i, 4+i], c1[:, 2, 2], atol=1e-4)


def test_pi_flux_state_against_the_paper(triangular):
    """The held pi-flux state of the triangular lattice: a real, uniform
    hopping t/J = 0.198 (arXiv:2503.13831), and a stiffness at K of
    1 - 3 Pi0 with their Pi0 = 0.389/J"""
    h, e = dirac(triangular)
    hop = h.get_multihopping().get_dict()
    assert max(np.max(np.abs(np.array(m).imag)) for m in hop.values()) < 1e-10
    t = hoppings(h)
    assert len(t) == 12 and np.ptp(t) < 1e-3, t
    assert abs(np.mean(t) - 0.198) < 2e-3, np.mean(t)
    r = h.get_rpa_stiffness(K, nk=61, delta=1e-3, T=1e-3)
    assert abs(r - (1. - 3.*0.389)) < 5e-3, r


def test_critical_couplings_of_the_paper(triangular):
    """The stiffness vanishes at K at J2/J1 = 0.0716, the transition to
    120 degree order, and at M at J2/J1 = 0.1758, the one to stripe order
    (arXiv:2503.13831), and it has the sign of a stable liquid in
    between. An odd nk keeps the Dirac points off the mesh."""
    kw = dict(nk=61, delta=1e-3, T=1e-3)
    def stiffness(J2, q): return dirac(triangular, J2=J2)[0].get_rpa_stiffness(q, **kw)
    assert abs(stiffness(0.0716, K)) < 3e-3
    assert abs(stiffness(0.1758, M)) < 5e-3
    assert stiffness(0.125, K) > 0.05 and stiffness(0.125, M) > 0.02
    assert stiffness(0.0, K) < -0.05


def test_gutzwiller_leaves_the_stiffness_unchanged(triangular):
    """J -> gJ doubles the hoppings and halves the bubble, so the static
    kernel is the same once the temperature and the broadening scale too"""
    r1 = dirac(triangular, J2=0.125)[0].get_rpa_stiffness(K, nk=31,
            delta=1e-3, T=1e-3)
    h = dirac(triangular, J2=0.125, gutzwiller=2.0)[0]
    r2 = h.get_rpa_stiffness(K, nk=31, delta=2e-3, T=2e-3)
    assert abs(r1 - r2) < 1e-5, (r1, r2)
    assert abs(np.mean(hoppings(h)) - 2.*0.198) < 4e-3


def test_spinon_exciton_pole(triangular):
    """get_magnon_bands goes to the same site-basis vertex: at K it finds
    one pole below the continuum, three times degenerate (a triplet), at
    the peak of the dressed response, and the pole softens as J2 goes
    down towards the transition at 0.0716"""
    es = np.linspace(0.01, 0.8, 80)
    def pole(J2):
        h = dirac(triangular, J2=J2)[0]
        qs, ws, gammas = h.get_magnon_bands(qpath=[K], energies=es,
                                            delta=0.01, nk=31)
        assert len(ws) == 3 and np.ptp(ws) < 1e-6, ws
        return h, ws[0]
    h, w = pole(0.125)
    chi = h.get_spinchi_full(q=K, energies=es, delta=0.01, nk=31)[1]
    peak = es[np.argmax([np.trace(c).imag for c in chi])]
    assert abs(w - peak) < 2.*(es[1] - es[0]), (w, peak)
    assert 0.05 < pole(0.09)[1] < w - 0.05


def twisted(mf, eps=0.1):
    """The guess mf with a phase exp(+-i eps) on every hopping, which
    breaks time reversal and keeps it Hermitian"""
    out = {}
    for d, m in mf.items():
        n = m.shape[0]//2
        f = np.ones((n, n), dtype=np.complex128)
        for i in range(n):
            for j in range(n):
                if d == (0, 0, 0): forward = i < j
                else: forward = d > tuple(-x for x in d)
                f[i, j] = np.exp(1j*eps*(1. if forward else -1.))
        out[d] = 0.2*m*np.kron(f, np.ones((2, 2)))
    return out


def test_ansatz_holds_the_pi_flux_state_and_the_free_loop_does_not(triangular):
    """From a guess that breaks time reversal slightly, the free loop
    goes to a state of lower bond energy with complex hoppings (the chiral
    one), and the loop with ansatz= comes back to the pi-flux state. This
    is why ansatz= exists."""
    g, ansatz = triangular
    guess = twisted(ansatz)
    for d in guess:
        assert np.allclose(guess[d], guess[tuple(-x for x in d)].conj().T)
    eh = dirac(triangular)[1]
    def imag(h):
        return max(np.max(np.abs(np.array(m).imag))
                   for m in h.get_multihopping().get_dict().values())
    kw = dict(J1=1.0, nk=20, maxerror=1e-6, mf=guess,
              return_total_energy=True)
    free, ef = bond_field(g, **kw)
    assert ef/2. < eh/2. - 3e-3, (ef, eh)
    assert imag(free) > 0.05
    held, e = bond_field(g, ansatz=ansatz, **kw)
    assert abs(e - eh) < 1e-4 and imag(held) < 1e-10


def test_decoupling_names_and_guards(triangular):
    g = geometry.chain()
    assert get_decoupling_names() == ["hartree-fock", "bond"]
    h = SpinonHamiltonian(g)
    with pytest.raises(ValueError, match="'hartree-fock', 'bond'"):
        h.get_mean_field_hamiltonian(J1=1.0, decoupling="fock")
    with pytest.raises(ValueError, match="gutzwiller"):
        h.get_mean_field_hamiltonian(J1=1.0, gutzwiller=0.0)
    with pytest.raises(NotImplementedError, match="isotropic"):
        h.get_mean_field_hamiltonian(J1=1.0, J1z=0.2, decoupling="bond")
    with pytest.raises(NotImplementedError, match="density-density"):
        h.get_mean_field_hamiltonian(J1=1.0, V1=0.2, decoupling="bond")
    with pytest.raises(NotImplementedError, match="integration"):
        h.get_mean_field_hamiltonian(J1=1.0, decoupling="bond",
                                     integration="kpm")
    with pytest.raises(NotImplementedError, match="ansatz"):
        h.get_mean_field_hamiltonian(J1=1.0,
                ansatz=g.get_hamiltonian(has_spin=True))
    with pytest.raises(ValueError, match="has_spin=True"):
        h.get_mean_field_hamiltonian(J1=1.0, decoupling="bond",
                ansatz=g.get_hamiltonian(has_spin=False))
    hz = SpinonHamiltonian(g)
    hz.add_zeeman([0., 0., 0.2])
    with pytest.raises(NotImplementedError, match="Zeeman"):
        hz.get_mean_field_hamiltonian(J1=1.0, decoupling="bond")


def test_pair_ladder_is_refused_on_a_bond_state():
    np.random.seed(0)
    g = geometry.chain()
    h = bond_field(g, J1=1.0, nk=100)
    for call in (lambda: h.get_transverse_spinchi(q=[0.1, 0., 0.]),
                 lambda: h.get_magnon_bands(method="pair"),
                 lambda: h.get_magnon_bands(method="tdhf"),
                 lambda: h.get_magnon_energies(),
                 lambda: h.get_goldstone_residual()):
        with pytest.raises(NotImplementedError, match="decoupling='bond'"):
            call()
    np.random.seed(0)
    hf = SpinonHamiltonian(g).get_mean_field_hamiltonian(J1=1.0, nk=24,
            mix=0.3, maxerror=1e-5, maxite=2000)
    with pytest.raises(ValueError, match="decoupling='bond'"):
        hf.get_rpa_stiffness([0.1, 0., 0.])
