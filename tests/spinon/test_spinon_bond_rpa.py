import numpy as np
import pytest

from pyqula import geometry
from pyqula.spinon import SpinonHamiltonian, get_decoupling_names
from pyqula.spinon import get_hold_names

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
    assert get_hold_names() == ["phases", "pattern"]
    ansatz = g.get_hamiltonian(has_spin=True)
    with pytest.raises(ValueError, match="'phases', 'pattern'"):
        h.get_mean_field_hamiltonian(J1=1.0, decoupling="bond",
                                     ansatz=ansatz, hold="amplitudes")
    with pytest.raises(ValueError, match="no ansatz"):
        h.get_mean_field_hamiltonian(J1=1.0, decoupling="bond",
                                     hold="pattern")
    with pytest.raises(ValueError, match="J_ij"):
        h.get_mean_field_hamiltonian(J2=1.0, decoupling="bond",
                                     ansatz=ansatz, hold="pattern")
    with pytest.raises(NotImplementedError, match="weiss"):
        h.get_mean_field_hamiltonian(J1=1.0, weiss=True)
    hz = SpinonHamiltonian(g)
    hz.add_zeeman([0., 0., 0.2])
    for kw in (dict(), dict(weiss=False)): # a Zeeman field, a Weiss field
        with pytest.raises(NotImplementedError, match="return_total_energy"):
            hz.get_mean_field_hamiltonian(J1=1.0, decoupling="bond",
                    return_total_energy=True, **kw)
    with pytest.raises(NotImplementedError, match="return_total_energy"):
        h.get_mean_field_hamiltonian(J1=1.0, decoupling="bond", weiss=True,
                                     return_total_energy=True)


def test_pair_ladder_is_refused_on_a_bond_state(monkeypatch):
    np.random.seed(0)
    g = geometry.chain()
    h = bond_field(g, J1=1.0, nk=100)
    for call in (lambda: h.get_transverse_spinchi(q=[0.1, 0., 0.]),
                 lambda: h.get_magnon_bands(method="pair"),
                 lambda: h.get_magnon_bands(method="tdhf"),
                 lambda: h.get_magnon_energies(),
                 lambda: h.get_goldstone_residual(),
                 lambda: h.get_magnon_chern()):
        with pytest.raises(NotImplementedError, match="decoupling='bond'"):
            call()
    np.random.seed(0)
    hf = SpinonHamiltonian(g).get_mean_field_hamiltonian(J1=1.0, nk=24,
            mix=0.3, maxerror=1e-5, maxite=2000)
    with pytest.raises(ValueError, match="decoupling='bond'"):
        hf.get_rpa_stiffness([0.1, 0., 0.])
    # on a state that is not a bond state the method is the one of
    # Hamiltonian, which returns what bse.magnon_chern returns
    from pyqula import bse
    monkeypatch.setattr(bse, "magnon_chern", lambda h, **kw: ("sentinel", h, kw))
    assert hf.get_magnon_chern(nq=4) == ("sentinel", hf, {"nq": 4})


def test_pattern_hold_agrees_with_phases_hold_for_equivalent_bonds(triangular):
    """With all the bonds of the ansatz equivalent, the single amplitude
    of hold="pattern" is the one every bond converges to on its own"""
    g = geometry.chain()
    ansatz = g.get_hamiltonian(has_spin=True)
    h = bond_field(g, J1=1.0, nk=400, ansatz=ansatz, hold="pattern")
    assert np.allclose(hoppings(h), 1./np.pi, atol=1e-3), hoppings(h)
    g, ansatz = triangular
    h, e = bond_field(g, J1=1.0, J2=0.125, ansatz=ansatz, nk=20,
                      maxerror=1e-6, hold="pattern", return_total_energy=True)
    t = hoppings(h)
    assert len(t) == 12 and np.ptp(t) < 1e-10
    href, eref = dirac(triangular, J2=0.125)
    assert abs(t[0] - np.mean(hoppings(href))) < 2e-4
    assert abs(e - eref) < 1e-5, (e, eref) # the same state, the same energy


def test_pattern_hold_with_inequivalent_bonds():
    """A chain of four sites with a modulated exchange. With
    hold="phases" the amplitudes are free and the loop goes to dimers,
    one bond in two close to zero. With hold="pattern" every hopping
    stays J_ij times one amplitude, with one fermion on every site."""
    g = geometry.chain().get_supercell(4)
    x0, L = np.mean(g.r[:, 0]), np.linalg.norm(g.a1)
    def J(r1, r2):
        d = np.linalg.norm(np.array(r1) - np.array(r2))
        if abs(d - 1.) > 1e-3: return 0.
        return 1. + 0.3*np.cos(2.*np.pi*((r1[0] + r2[0])/2. - x0)/L)
    def ratios(h): # t_ij / J_ij on every bond
        out = []
        for d, m in h.get_multihopping().get_dict().items():
            m = np.array(m)[0::2, 0::2]
            for i in range(4):
                for j in range(4):
                    Jij = J(g.r[i], g.r[j] + d[0]*g.a1)
                    if Jij != 0.: out.append(m[i, j]/Jij)
        return np.array(out)
    kw = dict(Jr=J, rcut=1.5, ansatz=g.get_hamiltonian(has_spin=True),
              nk=50, maxerror=1e-6, T=0.02)
    h = bond_field(g, hold="pattern", **kw)
    r = ratios(h)
    assert len(r) == 8 and np.ptp(r.real) < 1e-10, r
    assert 0.2 < abs(r[0]) < 0.4 and np.max(np.abs(r.imag)) < 1e-12
    assert np.allclose(h.local_occupation, 1.0, atol=1e-4)
    # the amplitude is the stationary point of the bond functional in the
    # family: with F(c) = c E1 + 2 c^2 sum_bonds J_ij, c^2 is minus the
    # hopping energy of the converged state over 4 sum_bonds J_ij
    hk, ks = h.get_hk_gen(), h.geometry.get_kmesh(nk=50)
    onsite = np.diag(np.diag(np.array(h.get_multihopping().get_dict()[(0, 0, 0)])))
    ehop = 0.
    for k in ks:
        m = np.array(hk(k))
        e, u = np.linalg.eigh(m)
        rho = (u/(np.exp(e/0.02) + 1.))@u.conj().T
        ehop += np.trace(rho@(m - onsite)).real/len(ks)
    sumJ = sum(J(g.r[i], g.r[(i+1) % 4] + (g.a1 if i == 3 else 0.))
               for i in range(4))
    assert abs(r[0].real**2 + ehop/(4.*sumJ)) < 1e-7, (r[0], ehop, sumJ)
    free = np.abs(ratios(bond_field(g, hold="phases", **kw)))
    assert free.min() < 0.05 and free.max() > 0.4, free


def test_non_reciprocal_exchange_raises():
    """S_i.S_j is symmetric, so the exchange of a bond cannot depend on the
    end it is seen from. A ramp J = 1 + 0.1 x on a cell of four sites is
    symmetric in its two arguments but not periodic: the bond that crosses
    the cell is 1.2 seen from site 3 and 0.8 seen from site 0. Both
    decouplings and every hold refuse it, before the loop starts."""
    g = geometry.chain().get_supercell(4)
    x0 = np.mean(g.r[:, 0])
    def J(r1, r2):
        if abs(np.linalg.norm(np.array(r1) - np.array(r2)) - 1.) > 1e-3:
            return 0.
        return 1. + 0.1*((r1[0] + r2[0])/2. - x0)
    ansatz = g.get_hamiltonian(has_spin=True)
    for kw in (dict(decoupling="hartree-fock"), dict(decoupling="bond"),
               dict(decoupling="bond", ansatz=ansatz, hold="phases"),
               dict(decoupling="bond", ansatz=ansatz, hold="pattern")):
        with pytest.raises(ValueError, match="not reciprocal") as err:
            SpinonHamiltonian(g).get_mean_field_hamiltonian(Jr=J, rcut=1.5,
                    nk=10, **kw)
        assert "mismatch of 0.4" in str(err.value), str(err.value)
        # the message names the exchange of the bond from its two ends,
        # with the keyword it was given with
        assert "Jr = 1.2" in str(err.value) and "Jr = 0.8" in str(err.value)


def modulated_dimer_chain():
    """A chain of two sites per cell with the exchange 1.3 on the bond
    inside the cell and 0.7 on the bond between cells, from both ends"""
    g = geometry.chain().get_supercell(2)
    x0 = np.mean(g.r[:, 0])
    def J(r1, r2):
        if abs(np.linalg.norm(np.array(r1) - np.array(r2)) - 1.) > 1e-3:
            return 0.
        return 1. + 0.3*np.cos(np.pi*((r1[0] + r2[0])/2. - x0))
    assert abs(J(g.r[0], g.r[1]) - 1.3) < 1e-12
    assert abs(J(g.r[1], g.r[0] + g.a1) - 0.7) < 1e-12
    assert abs(J(g.r[0], g.r[1] - g.a1) - 0.7) < 1e-12
    return g, J


def hopping_energy_at_zero_temperature(h, nk):
    """sum over the mesh of Tr[rho0 (H_k - onsite)], with rho0 the
    projector on the negative levels, from the Hamiltonian alone"""
    hk, ks = h.get_hk_gen(), h.geometry.get_kmesh(nk=nk)
    onsite = np.diag(np.diag(np.array(h.get_multihopping().get_dict()[(0, 0, 0)])))
    out = 0.
    for k in ks:
        m = np.array(hk(k))
        e, u = np.linalg.eigh(m)
        u = u[:, e < 0.]
        out += np.trace(u.conj().T@(m - onsite)@u).real/len(ks)
    return out


@pytest.mark.parametrize("gutzwiller", [1.0, 2.0])
def test_pattern_energy_is_the_bond_functional_on_inequivalent_bonds(
        gutzwiller):
    """With hold="pattern" on inequivalent bonds the energy is the bond
    functional at the converged hoppings, the hopping energy plus
    2 t_ij^2/(g J_ij) per bond, and it is stationary in the amplitude c of
    t_ij = c J_ij: E = -2 c^2 sum_bonds J_ij / g, half of the hopping
    energy. The gap 1.2 c, about 0.4 g, is far above T, so the hopping
    energy at zero temperature is the one of the loop. The Hartree-Fock
    energy of the engine is above it by sum g J |chi|^2/2 - sum 2t^2/(gJ),
    about 0.09 g here."""
    g, J = modulated_dimer_chain()
    nk = 100
    h, e = bond_field(g, Jr=J, rcut=1.5, ansatz=g.get_hamiltonian(has_spin=True),
                      hold="pattern", nk=nk, T=0.01*gutzwiller, maxerror=1e-8,
                      maxite=3000, gutzwiller=gutzwiller,
                      return_total_energy=True)
    hop = {tuple(int(i) for i in d): np.array(m)
           for d, m in h.get_multihopping().get_dict().items()}
    t_in, t_out = hop[(0, 0, 0)][0, 2].real, hop[(1, 0, 0)][2, 0].real
    c = t_in/1.3
    assert abs(t_out/0.7 - c) < 1e-10 and 0.2 < abs(c)/gutzwiller < 0.5
    assert 2.*abs(t_in - t_out) > 30.*0.01*gutzwiller # the gap against T
    ehop = hopping_energy_at_zero_temperature(h, nk)
    field = 2.*t_in**2/(gutzwiller*1.3) + 2.*t_out**2/(gutzwiller*0.7)
    assert abs(e - (ehop + field)) < 1e-7, (e, ehop + field)
    assert abs(e + 2.*c**2*(1.3 + 0.7)/gutzwiller) < 1e-8, (e, c)
    assert abs(e - ehop/2.) < 1e-8, (e, ehop)


@pytest.mark.parametrize("gutzwiller", [1.0, 2.0])
def test_energy_takes_out_the_lagrange_multiplier(gutzwiller):
    """At one fermion per site a uniform onsite energy eps is the constant
    eps per site. The loop takes it out with the Lagrange multiplier, which
    moves by eps, and the energy moves by eps per site, which fixes the
    sign and the weight with which the multiplier is undone. The uniform
    state of the triangular lattice already has a multiplier of about
    -0.14 g J, since its band is not particle-hole symmetric."""
    g = geometry.triangular_lattice()
    def loop(eps):
        h = SpinonHamiltonian(g)
        if eps != 0.: h.add_onsite(eps)
        return h.get_mean_field_hamiltonian(J1=1.0, decoupling="bond",
                nk=30, T=0.01*gutzwiller, mix=0.5, maxerror=1e-9,
                maxite=3000, ansatz=g.get_hamiltonian(has_spin=True),
                gutzwiller=gutzwiller, return_total_energy=True)
    (h0, e0), (h1, e1) = loop(0.), loop(0.3)
    lam0, lam1 = h0.constraint_lambda[0], h1.constraint_lambda[0]
    assert abs(lam0) > 0.1*gutzwiller, lam0
    assert abs(lam1 - lam0 - 0.3) < 1e-8, (lam0, lam1)
    assert abs(e1 - e0 - 0.3) < 1e-8, (e0, e1)


def test_phases_energy_is_the_bond_functional_with_a_twisted_ansatz(
        triangular):
    """With hold="phases" every hopping keeps the phase of the ansatz, here
    the pi-flux pattern twisted by a phase that breaks time reversal, and
    chi_ij is not aligned with that phase. The energy is the bond
    functional at the held hoppings and is stationary in each amplitude,
    so it is half of the hopping energy, up to the difference between the
    hopping energy at zero temperature and at the T=0.01 of the loop,
    4e-6 for this gapless state. The Hartree-Fock energy of the engine is
    8e-5 above the functional here."""
    g, ansatz = triangular
    nk = 20
    h, e = bond_field(g, J1=1.0, ansatz=twisted(ansatz), nk=nk,
                      maxerror=1e-7, maxite=3000, return_total_energy=True)
    field = 0. # 2|t_ij|^2/J_ij per bond with J_ij = 1, each bond seen twice
    for d, m in h.get_multihopping().get_dict().items():
        m = np.array(m)[0::2, 0::2]
        if tuple(int(i) for i in d) == (0, 0, 0): m = m - np.diag(np.diag(m))
        field += np.sum(np.abs(m)**2)
    ehop = hopping_energy_at_zero_temperature(h, nk)
    assert abs(e - (ehop + field)) < 1e-6, (e, ehop + field)
    assert abs(e - ehop/2.) < 2e-5, (e, ehop)


def chain_in_field(b, J1=1.0, T=0.1, weiss=None, gutzwiller=1.0):
    """The uniform chain in a Zeeman field b, a number (along z) or a
    vector"""
    g = geometry.chain()
    h = SpinonHamiltonian(g)
    if np.any(np.array(b) != 0.):
        h.add_zeeman(b if np.ndim(b) == 1 else [0., 0., b])
    return h.get_mean_field_hamiltonian(J1=J1, decoupling="bond", nk=100,
            ansatz=g.get_hamiltonian(has_spin=True), T=T, mix=0.3,
            maxerror=1e-9, maxite=3000, weiss=weiss, gutzwiller=gutzwiller)


def test_bond_states_take_the_fast_routes_of_the_bubble(triangular):
    """The bubble of chitk.localspinchi dresses a spin-degenerate bond
    state block by block when its vertex has one block per spin component,
    and contracts the two spins apart for a state in a field, along z or
    not; the checks are structural, so a converged bond state must pass
    them exactly"""
    from pyqula.chitk import localspinchi, rpa, spinchi
    h = dirac(triangular)[0]
    n = len(h.geometry.r)
    assert localspinchi._route(h) == localspinchi._DEGENERATE
    for q in (K, M):
        Vq = np.asarray(rpa.interaction_at_q(spinchi._full_spin_U(h), h, q))
        assert localspinchi._common_block(Vq, n) is not None
    hz = chain_in_field(0.1)
    assert hz.weiss is True
    assert localspinchi._route(hz) == localspinchi._COLLINEAR
    assert localspinchi._route_frame(hz) == (localspinchi._COLLINEAR, None)
    # a field that is not along z: the Zeeman and the Weiss field of the
    # converged state are along one direction, so the route is the
    # collinear one in the spin frame of the field
    b = np.array([0.06, -0.03, 0.07])
    hn = chain_in_field(b)
    assert hn.weiss is True
    assert localspinchi._route(hn) == localspinchi._COLLINEAR
    route, (U, R) = localspinchi._route_frame(hn)
    assert np.allclose(np.abs(R[:, 2]), np.abs(b)/np.linalg.norm(b),
                       atol=1e-8)


def test_static_rpa_is_the_derivative_of_the_weiss_loop():
    """With the Weiss field in the loop, the moment induced by a small
    Zeeman field b sigma_z = 2 b S_z is the static RPA response of the
    state without field: d<sigma_z>/db = 4 chi_zz(q=0,w=0). Without the
    Weiss field it is the bare one. This fixes the vertex J(q) and the
    weight of the Weiss field independently of any reference."""
    h0 = chain_in_field(0.)
    assert h0.weiss is False
    kw = dict(q=[0., 0., 0.], energies=[0.], delta=1e-5, nk=100, T=0.1)
    chi = h0.get_spinchi_full(**kw)[1][0][2, 2].real
    chi0 = h0.get_spinchi_full(RPA=False, **kw)[1][0][2, 2].real
    assert abs(chi) > 0.1 and abs(chi - chi0) > 0.05
    db = 0.005
    def slope(weiss):
        mp = chain_in_field(db, weiss=weiss).get_magnetization()[0][2]
        mm = chain_in_field(-db, weiss=weiss).get_magnetization()[0][2]
        return (mp - mm)/(2.*db)
    assert chain_in_field(db).weiss is True
    assert abs(slope(None) - 4.*chi) < 2e-4, (slope(None), 4.*chi)
    assert abs(slope(False) - 4.*chi0) < 2e-4, (slope(False), 4.*chi0)
    # a field in a generic direction: the moment is along it, with the
    # same slope, which also runs the spin-flip part of the Weiss field
    b = np.array([0.003, -0.002, 0.003])
    m = np.array(chain_in_field(b).get_magnetization()[0])
    assert np.allclose(m, 4.*chi*b, atol=1e-6), (m, 4.*chi*b)


@pytest.mark.parametrize("J1,gutzwiller", [(1.0, 1.0), (0.5, 1.0), (1.0, 2.0)])
def test_uniform_transverse_mode_is_at_the_zeeman_energy(J1, gutzwiller):
    """The total spin commutes with the exchange, so the uniform
    transverse mode of a Heisenberg model in a field is at the Zeeman
    energy 2b whatever J is. The RPA on the loop with the Weiss field
    keeps this exactly: the Weiss field shifts the spinon splitting away
    from 2b and the vertex brings the pole back, also with the Gutzwiller
    factor. Without the Weiss field in the loop the pole is displaced."""
    b = 0.1
    es = np.linspace(0., 0.6, 121)
    def peak(weiss):
        h = chain_in_field(b, J1=J1, T=0.02, weiss=weiss,
                           gutzwiller=gutzwiller)
        chi = h.get_spinchi_full(q=[0., 0., 0.], energies=es, delta=0.005,
                                 nk=100, T=0.02)[1]
        return es[np.argmax(np.abs(chi[:, 0, 0].imag))]
    assert abs(peak(None) - 2.*b) < 1.5*(es[1] - es[0]), peak(None)
    assert abs(peak(False) - 2.*b) > 0.05, peak(False)


def test_weiss_field_without_zeeman_field_follows_the_stiffness(triangular):
    """weiss=True without a field gives the magnetic channel back. From a
    guess with a small staggered field, the two-site chain, whose
    stiffness is negative at q=pi, goes to the Neel state. The pi-flux
    state at J2 = 0 has a positive stiffness at M, the momentum of a
    moment that alternates between its two sites, and goes back to zero
    moment (its instability is at K, which this cell cannot hold)."""
    def seeded(pattern, scale=1.):
        out = {tuple(int(i) for i in d): scale*np.array(m, dtype=np.complex128)
               for d, m in pattern.items()}
        out[(0, 0, 0)] = out[(0, 0, 0)] + 1e-3*np.diag([1., -1., -1., 1.])
        return out
    g = geometry.chain().get_supercell(2)
    a = g.get_hamiltonian(has_spin=True)
    kw = dict(J1=1.0, ansatz=a, hold="pattern", nk=100, T=0.02,
              maxerror=1e-5, maxite=2000)
    assert bond_field(g, **kw).get_rpa_stiffness([0., 0., 0.], nk=101,
                                                 T=0.02) < -0.5
    m = bond_field(g, mf=seeded(a.get_dict()), weiss=True,
                   **kw).get_magnetization()[:, 2]
    assert abs(m[0]) > 0.5 and abs(m[0] + m[1]) < 1e-3, m
    g, ansatz = triangular
    assert dirac(triangular)[0].get_rpa_stiffness(M, nk=31, delta=1e-3,
                                                  T=1e-3) > 0.1
    h = bond_field(g, J1=1.0, ansatz=ansatz, weiss=True, nk=20,
                   mf=seeded(ansatz, 0.2), T=0.02, maxerror=1e-5,
                   maxite=3000)
    assert np.max(np.abs(h.get_magnetization())) < 1e-3
