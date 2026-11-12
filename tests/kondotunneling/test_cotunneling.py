import numpy as np
import pytest

from pyqula import geometry
from pyqula.kondolattice import KondoLatticeHamiltonian

# Cotunneling STM spectrum of the Kondo lattice mean field: the tip tunnels
# into psi = tc c + tf f, and dI/dV = tc^2 N_cc + tf^2 N_ff + 2 tc tf N_cf
# (Maltseva, Dzero and Coleman, arXiv:0910.1138; Figgins and Morr,
# arXiv:1001.4530). Every test checks an invariant of that construction
# against an independent route, not a recorded number.

DELTA = 0.02


def _chain_kondo(natoms=1, onsite=None):
    g = geometry.chain()
    if natoms > 1: g = g.get_supercell(natoms)
    hc = g.get_hamiltonian(has_spin=True)
    if onsite is not None: hc.add_onsite(onsite)
    return KondoLatticeHamiltonian(hc)


@pytest.fixture(scope="module")
def hchain():
    """Converged hybridized chain, primitive cell"""
    h = _chain_kondo()
    return h.get_mean_field_hamiltonian(J=1.5, filling=0.15, nk=100,
            mf=([0.3+0j], [0.0]), maxerror=1e-8, maxite=3000)


def _onsites(h):
    """Conduction center and f level of a one-pair chain, from the
    converged Hamiltonian (measured from the chemical potential)"""
    return h.intra[0, 0].real, h.intra[2, 2].real


def test_hamiltonian_keeps_kondo_data(hchain):
    assert isinstance(hchain, KondoLatticeHamiltonian)
    assert len(hchain._kondo_pairs) == 1
    assert abs(hchain.hybridization[0]) > 0.05


def test_tf_zero_is_the_conduction_ldos(hchain):
    """With tf=0 the spectrum is the LDOS of the conduction electron,
    computed here by an independent route: the Green function of the
    whole Hamiltonian, inverted at every energy and k-point."""
    nk = 60
    energies = np.linspace(-1.0, 1.5, 11)
    e, d = hchain.get_cotunneling_didv(energies=energies, tc=1.0, tf=0.0,
            nk=nk, delta=DELTA)
    gk = hchain.get_gk_gen(delta=DELTA)
    ref = []
    for en in energies:
        tot = 0.0
        for k in np.linspace(0., 1., nk, endpoint=False):
            g = gk([k, 0., 0.], e=en)
            tot += -(g[0, 0]+g[1, 1]).imag/np.pi
        ref.append(tot/nk)
    assert np.allclose(d[0], ref, atol=1e-8)


def test_no_hybridization_has_no_f_signal():
    """Without hybridization the f level is decoupled: no interference,
    and the f part is a Lorentzian line at the f level only"""
    h = _chain_kondo()
    h2 = h.get_mean_field_hamiltonian(J=0.0, filling=0.5, nk=40, mix=0.5,
            maxerror=1e-7, maxite=500)
    assert np.allclose(h2.hybridization, 0.0, atol=1e-8)
    e, d, parts = h2.get_cotunneling_didv(tc=1.0, tf=0.5, nk=40,
            delta=DELTA, return_parts=True)
    assert np.max(np.abs(parts["cf"])) < 1e-8
    lam = _onsites(h2)[1]
    line = 2*(DELTA/np.pi)/((e-lam)**2+DELTA**2) # two spins
    assert np.allclose(parts["ff"][0], 0.25*line, atol=1e-6)


def test_uniform_lattice_matches_the_closed_formula(hchain):
    """G_psi(k,w) = (tc + tf V/(w-lam))^2/(w-eps_k-V^2/(w-lam)) of
    arXiv:0910.1138, plus the bare f pole tf^2/(w-lam), summed over k and
    over the two spins"""
    nk = 200
    tc, tf = 1.0, 0.3
    V = hchain.hybridization[0].real
    e0, lam = _onsites(hchain)
    e, d = hchain.get_cotunneling_didv(tc=tc, tf=tf, nk=nk, delta=DELTA,
            emax=5.0, ne=201)
    w = e+1j*DELTA
    ks = 2*np.pi*np.arange(nk)/nk
    G = np.mean([(tc+tf*V/(w-lam))**2/(w-e0-2*np.cos(k)-V**2/(w-lam))
            for k in ks], axis=0)+tf**2/(w-lam)
    assert np.allclose(d[0], -2*G.imag/np.pi, atol=1e-10)


def test_interference_changes_sign_with_tf(hchain):
    """Flipping the sign of tf flips the interference term and leaves the
    conduction and f parts untouched"""
    kw = dict(tc=1.0, nk=100, delta=DELTA, emax=4.0, ne=161,
            return_parts=True)
    e, dp, pp = hchain.get_cotunneling_didv(tf=0.4, **kw)
    e, dm, pm = hchain.get_cotunneling_didv(tf=-0.4, **kw)
    assert np.max(np.abs(pp["cf"])) > 0.1 # really there
    assert np.allclose(pp["cf"], -pm["cf"], atol=1e-12)
    assert np.allclose(pp["cc"], pm["cc"], atol=1e-12)
    assert np.allclose(pp["ff"], pm["ff"], atol=1e-12)
    assert np.allclose(dp, pp["cc"]+pp["ff"]+pp["cf"], atol=1e-12)
    assert np.allclose(dp+dm, 2*(pp["cc"]+pp["ff"]), atol=1e-12)


def test_cf_part_scales_as_tc_tf(hchain):
    kw = dict(nk=60, delta=DELTA, emax=3.0, ne=61, return_parts=True)
    e, d1, p1 = hchain.get_cotunneling_didv(tc=1.0, tf=1.0, **kw)
    e, d2, p2 = hchain.get_cotunneling_didv(tc=0.5, tf=2.0, **kw)
    assert np.allclose(p2["cc"], 0.25*p1["cc"], atol=1e-12)
    assert np.allclose(p2["ff"], 4.0*p1["ff"], atol=1e-12)
    assert np.allclose(p2["cf"], p1["cf"], atol=1e-12)


def test_sum_rules(hchain):
    """Each spin of c and of f carries one state: the integrals of the cc
    and ff parts are 2 for the two spins (up to the Lorentzian tails)"""
    e, d, parts = hchain.get_cotunneling_didv(tc=1.0, tf=1.0, nk=100,
            delta=0.02, emax=40.0, ne=8001, return_parts=True)
    de = e[1]-e[0]
    assert abs(np.sum(parts["cc"][0])*de-2.0) < 0.05
    assert abs(np.sum(parts["ff"][0])*de-2.0) < 0.05


def test_thermal_broadening_is_the_fermi_window(hchain):
    """dI/dV at temperature T is the T=0 spectrum averaged with -df/dw,
    evaluated here by brute force at the interior energies"""
    T = 0.1
    kw = dict(tc=1.0, tf=0.2, nk=80, delta=0.05)
    e = np.linspace(-3.0, 3.0, 601)
    e0, dT = hchain.get_cotunneling_didv(energies=e, T=T, **kw)
    e0, d0 = hchain.get_cotunneling_didv(energies=e, T=0.0, **kw)
    de = e[1]-e[0]
    for i in (150, 300, 450):
        x = e-e[i]
        win = 1.0/(4*T*np.cosh(x/(2*T))**2)
        assert abs(np.sum(win*d0[0])*de-dT[0, i]) < 2e-3*d0[0].max()


def test_supercell_gives_the_same_spectrum_per_site(hchain):
    """The two sites of a doubled cell, on the same k-mesh, give the
    spectrum of the primitive cell"""
    h2 = _chain_kondo(2).get_mean_field_hamiltonian(J=1.5, filling=0.15,
            nk=50, mf=([0.3+0j, 0.3+0j], [0.0, 0.0]), maxerror=1e-8,
            maxite=3000)
    assert h2 is not None
    kw = dict(tc=1.0, tf=0.3, delta=DELTA, emax=4.0, ne=161)
    e, dp = hchain.get_cotunneling_didv(nk=100, **kw)
    e, ds = h2.get_cotunneling_didv(nk=50, **kw)
    assert ds.shape[0] == 2
    scale = dp.max()
    assert np.max(np.abs(ds[0]-dp[0])) < 2e-3*scale
    assert np.max(np.abs(ds[1]-dp[0])) < 2e-3*scale


def test_modulated_cell_has_inequivalent_sites():
    """A staggered onsite energy makes the two sites differ, and the site
    keyword selects them"""
    h = _chain_kondo(2, onsite=lambda r: 0.4*np.sin(np.pi*r[0]))
    h2 = h.get_mean_field_hamiltonian(J=1.5, filling=0.15, nk=50,
            mf=([0.3+0j, 0.3+0j], [0.0, 0.0]), maxerror=1e-7, maxite=3000)
    assert h2 is not None
    kw = dict(tc=1.0, tf=0.3, nk=50, delta=DELTA, emax=4.0, ne=161)
    e, d = h2.get_cotunneling_didv(**kw)
    e, d1 = h2.get_cotunneling_didv(site=1, **kw)
    e, dl = h2.get_cotunneling_didv(site=[1, 0], **kw)
    assert np.max(np.abs(d[0]-d[1])) > 1e-2*d.max()
    assert d1.shape == e.shape # a single index gives one array
    assert np.allclose(d1, d[1], atol=1e-12)
    assert np.allclose(dl, d[[1, 0]], atol=1e-12)


def test_J_gives_the_composite_amplitude(hchain):
    """With J given, the f amplitude is tf*conj(V)/J"""
    V = hchain.hybridization[0]
    kw = dict(tc=1.0, nk=60, delta=DELTA, emax=3.0, ne=61)
    e, da = hchain.get_cotunneling_didv(tf=0.7, J=1.5, **kw)
    e, db = hchain.get_cotunneling_didv(tf=0.7*np.conjugate(V)/1.5, **kw)
    assert np.allclose(da, db, atol=1e-12)


def test_requires_a_kondo_lattice_and_valid_options(hchain):
    from pyqula.kondotk import cotunneling
    h = geometry.chain().get_hamiltonian(has_spin=True)
    with pytest.raises(TypeError, match="KondoLatticeHamiltonian"):
        cotunneling.cotunneling_didv(h)
    with pytest.raises(ValueError, match="Kondo pair"):
        hchain.get_cotunneling_didv(site=3, nk=4)
    with pytest.raises(ValueError, match="delta"):
        hchain.get_cotunneling_didv(delta=0.0, nk=4)
    with pytest.raises(ValueError, match="uniformly"):
        hchain.get_cotunneling_didv(energies=[0.0, 0.1, 0.5], nk=4)
