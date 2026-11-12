import numpy as np
import pytest

from pyqula import geometry
from pyqula.kondolattice import KondoLatticeHamiltonian
from pyqula.multihopping import MultiHopping

# Matrix spectral function of the conduction electron c and the composite
# fermion F = (2/3)(sigma.S) c of the Kondo lattice mean field (Chen,
# Stoudenmire, Komijani and Coleman, arXiv:2302.09701). The paper writes
# the Kondo term as J'(c^dag sigma c).S, and the loop of this library
# takes the Coqblin-Schrieffer coupling J = 2J'. Every test checks an
# invariant of the construction against an independent route, not a
# recorded number.

DELTA = 0.05
JP = 2.0 # the J' of the paper, in units of the hopping


def _chain_kondo(natoms=1):
    g = geometry.chain()
    if natoms > 1: g = g.get_supercell(natoms)
    return KondoLatticeHamiltonian(g.get_hamiltonian(has_spin=True))


def _converge(h, jp, nk=200):
    n = len(h._kondo_pairs)
    return h.get_mean_field_hamiltonian(J=2*jp, filling=0.5, nk=nk,
            mf=([jp+0j]*n, [0.0]*n), maxerror=1e-8, maxite=3000)


@pytest.fixture(scope="module")
def hchain():
    """Converged half-filled chain, the Kondo insulator of the paper"""
    return _converge(_chain_kondo(), JP)


def _eps(k):
    """Conduction band of geometry.chain() at the reduced momentum k"""
    return 2*np.cos(2*np.pi*k)


def _two_band_green(k, energies, V, zF, lam=0.0):
    """Green function of (c, F = zF f) of one spin from the inverse of
    the 2 x 2 matrix, one inversion per energy"""
    D = np.diag([1.0, zF])
    out = []
    for w in energies:
        m = np.array([[w+1j*DELTA-_eps(k), -np.conjugate(V)],
                [-V, w+1j*DELTA-lam]])
        out.append(D@np.linalg.inv(m)@np.conjugate(D))
    return np.array(out)


def test_half_filled_loop_is_the_gap_equation(hchain):
    """At half filling the loop gives lam=0, n_f=1 and the V of the T=0
    gap equation 1/J' = 1/N sum_k 1/sqrt((eps_k/2)^2+V^2), which follows
    from J = 2J'"""
    from scipy.optimize import brentq
    e = _eps((np.arange(4000)+0.5)/4000)
    V0 = brentq(lambda V: np.mean(1/np.sqrt((e/2)**2+V**2))-1/JP, 1e-6, 50)
    assert abs(hchain.hybridization[0]-V0) < 1e-4
    assert abs(hchain.constraint_lambda[0]) < 1e-8
    assert abs(hchain.local_occupation[0]-1.0) < 1e-8


@pytest.mark.parametrize("J,scale", [(None, 1.0), (2*JP, 1.0),
        (2*JP, 1.5), (None, 1.5)])
def test_matches_the_inverse_of_the_two_band_matrix(hchain, J, scale):
    """The routine builds the spectrum from the eigenstates; here the
    2 x 2 matrix z - H(k) is inverted at every energy instead"""
    V = hchain.hybridization[0]*scale
    zF = 1.0 if J is None else 4*np.conjugate(V)/(3*J)
    es = np.linspace(-6.0, 6.0, 61)
    kw = dict(energies=es, delta=DELTA, J=J, hybridization_scale=scale,
            nk=9)
    ks, e, A = hchain.get_composite_spectral_function(**kw)
    ks, e, G = hchain.get_composite_spectral_function(green=True, **kw)
    assert np.max(A["FF"]) > 0.3 # there is a signal
    for ik, k in enumerate(ks):
        ref = _two_band_green(k[0], es, V, zF)
        assert np.allclose(G["cc"][ik], ref[:, 0, 0], atol=1e-10)
        assert np.allclose(G["cF"][ik], ref[:, 0, 1], atol=1e-10)
        assert np.allclose(G["Fc"][ik], ref[:, 1, 0], atol=1e-10)
        assert np.allclose(G["FF"][ik], ref[:, 1, 1], atol=1e-10)
        assert np.allclose(A["cc"][ik], -ref[:, 0, 0].imag/np.pi, atol=1e-10)
        assert np.allclose(A["FF"][ik], -ref[:, 1, 1].imag/np.pi, atol=1e-10)
        assert np.allclose(A["cF"][ik], -ref[:, 0, 1].imag/np.pi, atol=1e-10)


@pytest.mark.parametrize("scale", [1.0, 1.5])
def test_equations_of_motion_of_the_paper(hchain, scale):
    """Eq. relations and Eq. matrixG of arXiv:2302.09701: with the
    amplitude of the composite fermion that J gives, the Green functions
    of the mean field obey G_cc = g_c + g_c v G_FF v g_c and
    G_cF = g_c v G_FF, and the inverse of the 2 x 2 matrix is
    [[z - eps_k, -v], [-v, 1/g_F]], with v = 3J'/2 the hybridization of c
    and F. A rescale of V changes the weight of F and not v, so the
    relations hold for any scale"""
    es = np.linspace(-6.0, 6.0, 41)
    ks, e, G = hchain.get_composite_spectral_function(energies=es,
            delta=DELTA, J=2*JP, hybridization_scale=scale, nk=7,
            green=True)
    v = 1.5*JP
    z = es+1j*DELTA
    for ik, k in enumerate(ks):
        gc = 1/(z-_eps(k[0]))
        assert np.allclose(G["cc"][ik], gc+gc*v*G["FF"][ik]*v*gc, atol=1e-10)
        assert np.allclose(G["cF"][ik], gc*v*G["FF"][ik], atol=1e-10)
        for iw in range(len(es)):
            m = np.array([[G["cc"][ik, iw], G["cF"][ik, iw]],
                    [G["Fc"][ik, iw], G["FF"][ik, iw]]])
            mi = np.linalg.inv(m)
            assert abs(mi[0, 0]-(z[iw]-_eps(k[0]))) < 1e-8
            assert abs(mi[0, 1]+v) < 1e-8
            assert abs(mi[1, 0]+v) < 1e-8
            assert abs(1/mi[1, 1]-G["gF"][ik, iw]) < 1e-8


def test_sum_rules(hchain):
    """A_cc integrates to one state per spin, and A_FF to the weight
    |z|^2 = |4V/(3J)|^2 of the f fermion in the composite fermion"""
    es = np.linspace(-60.0, 60.0, 6001)
    V = hchain.hybridization[0]
    ks, e, A = hchain.get_composite_spectral_function(energies=es,
            delta=0.05, J=2*JP, nk=5)
    de = es[1]-es[0]
    assert np.allclose(np.sum(A["cc"], axis=1)*de, 1.0, atol=2e-3)
    assert np.allclose(np.sum(A["FF"], axis=1)*de, abs(2*V/(3*JP))**2,
            atol=2e-3)
    ks, e, A = hchain.get_composite_spectral_function(energies=es,
            delta=0.05, nk=5) # F = f
    assert np.allclose(np.sum(A["FF"], axis=1)*de, 1.0, atol=2e-3)


def test_unhybridized_composite_fermion_is_the_f_level(hchain):
    """g_F has one pole at the f level, with the weight of the composite
    fermion, at every k: the mean field has no dispersion of its own for
    the composite fermion"""
    es = np.linspace(-1.0, 1.0, 21)
    V = hchain.hybridization[0]
    ks, e, A = hchain.get_composite_spectral_function(energies=es,
            delta=DELTA, J=2*JP, nk=5)
    ref = abs(2*V/(3*JP))**2/(es+1j*DELTA)
    for ik in range(len(ks)):
        assert np.allclose(A["gF"][ik], ref, atol=1e-10)


def test_rescaled_bands_are_the_exact_strong_coupling_energy():
    """One site with J'(c^dag sigma c).S, diagonalized exactly, costs
    3J'/2 to add or to remove an electron. The loop gives V = J' at
    strong coupling, so the bands of the mean field are at J', and at
    3J'/2 with the rescale of 3/2 of the paper"""
    jp = 10.0
    # exact: one conduction orbital and one spin 1/2
    a = np.array([[0, 1.], [0, 0]])
    cu, cd = np.kron(a, np.eye(2)), np.kron(np.diag([1., -1.]), a)
    c = [cu, cd]
    pauli = [np.array([[0, 1], [1, 0]]), np.array([[0, -1j], [1j, 0]]),
            np.diag([1., -1.])]
    H = np.zeros((8, 8), dtype=np.complex128)
    for s in pauli:
        cs = sum(s[i, j]*c[i].T@c[j] for i in range(2) for j in range(2))
        H += jp*np.kron(cs, s/2)
    N = np.kron(cu.T@cu+cd.T@cd, np.eye(2))
    ev, vs = np.linalg.eigh(H)
    occ = np.einsum("in,ij,jn->n", vs.conj(), N, vs).real
    e0 = ev[0] # the Kondo singlet, one electron
    assert abs(occ[0]-1.0) < 1e-10
    exact = np.min(ev[np.abs(occ-2.0) < 1e-8])-e0 # add one electron
    assert abs(exact-1.5*jp) < 1e-10
    # mean field
    h2 = _converge(_chain_kondo(), jp, nk=100)
    assert abs(h2.hybridization[0]-jp) < 0.05 # V = J' up to t^2/J'
    es = np.linspace(0.0, 20.0, 2001)
    kq = [[0.25, 0., 0.]] # eps_k = 0, so the bands are at +-V
    for scale, target in [(1.0, jp), (1.5, exact)]:
        ks, e, A = h2.get_composite_spectral_function(kpath=kq,
                energies=es, delta=0.02, hybridization_scale=scale)
        assert abs(es[np.argmax(A["FF"][0])]-target) < 0.05


def test_supercell_folds_the_spectrum(hchain):
    """The average over the two pairs of a doubled cell at the momentum k
    is the mean of the primitive spectra at k/2 and k/2 + 1/2"""
    h2 = _converge(_chain_kondo(2), JP, nk=100)
    assert np.allclose(h2.hybridization, hchain.hybridization[0], atol=1e-5)
    es = np.linspace(-6.0, 6.0, 61)
    kw = dict(energies=es, delta=DELTA, J=2*JP)
    for k in (0.0, 0.13, 0.37):
        ks, e, As = h2.get_composite_spectral_function(
                kpath=[[k, 0., 0.]], **kw)
        ks, e, Ap = hchain.get_composite_spectral_function(
                kpath=[[k/2, 0., 0.], [k/2+0.5, 0., 0.]], **kw)
        for key in ("cc", "FF", "cF"):
            ref = (Ap[key][0]+Ap[key][1])/2
            assert np.max(np.abs(As[key][0]-ref)) < 1e-4*np.max(np.abs(ref))
    # one pair, and a list of pairs
    ks, e, A1 = h2.get_composite_spectral_function(site=1, nk=3, **kw)
    ks, e, Al = h2.get_composite_spectral_function(site=[1, 0], nk=3, **kw)
    ks, e, Aa = h2.get_composite_spectral_function(nk=3, **kw)
    assert A1["FF"].shape == (3, len(es))
    assert Al["FF"].shape == (2, 3, len(es))
    assert np.allclose(Al["FF"][0], A1["FF"], atol=1e-12)
    assert np.allclose(np.mean(Al["FF"], axis=0), Aa["FF"], atol=1e-12)


def test_phase_of_the_hybridization_is_a_gauge(hchain):
    """Multiplying V by a phase is a change of the phase of f. The
    composite fermion that J defines does not depend on it, and the one
    identified with f does, in G_cF only"""
    phase = np.exp(0.7j)
    hop = dict(hchain.get_multihopping().get_dict())
    m = np.array(hop[(0, 0, 0)], dtype=np.complex128)
    for (ci, fi) in hchain._kondo_pairs:
        m[2*fi:2*fi+2, 2*ci:2*ci+2] *= phase # V f^dag c
        m[2*ci:2*ci+2, 2*fi:2*fi+2] *= np.conjugate(phase)
    hop[(0, 0, 0)] = m
    hg = hchain.copy()
    hg.set_multihopping(MultiHopping(hop))
    es = np.linspace(-6.0, 6.0, 41)
    kw = dict(energies=es, delta=DELTA, nk=7, green=True)
    ks, e, G0 = hchain.get_composite_spectral_function(J=2*JP, **kw)
    ks, e, G1 = hg.get_composite_spectral_function(J=2*JP, **kw)
    for key in ("cc", "cF", "Fc", "FF", "gF"):
        assert np.allclose(G0[key], G1[key], atol=1e-10)
    ks, e, G0 = hchain.get_composite_spectral_function(**kw)
    ks, e, G1 = hg.get_composite_spectral_function(**kw)
    assert np.max(np.abs(G0["cF"])) > 0.1
    assert np.allclose(G1["cF"], np.conjugate(phase)*G0["cF"], atol=1e-10)
    assert np.allclose(G1["FF"], G0["FF"], atol=1e-10)


def test_requires_a_kondo_lattice_and_valid_options(hchain):
    from pyqula.kondotk import compositespectrum
    h = geometry.chain().get_hamiltonian(has_spin=True)
    with pytest.raises(TypeError, match="KondoLatticeHamiltonian"):
        compositespectrum.composite_spectral_function(h)
    with pytest.raises(ValueError, match="Kondo pair"):
        hchain.get_composite_spectral_function(site=3, nk=3)
    with pytest.raises(ValueError, match="delta"):
        hchain.get_composite_spectral_function(delta=0.0, nk=3)
    with pytest.raises(ValueError, match="hybridization_scale"):
        hchain.get_composite_spectral_function(hybridization_scale=0.0, nk=3)
    with pytest.raises(ValueError, match="J=0"):
        hchain.get_composite_spectral_function(J=0.0, nk=3)
