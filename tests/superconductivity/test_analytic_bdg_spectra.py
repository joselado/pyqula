"""BdG spectra against closed forms that tie the pairing to physical spin.

The d-vector round trip (add a triplet with d, extract d back) shares one
convention between the two directions, so a global relabeling of the spin
axes would pass it. A Zeeman field breaks that symmetry: a field along d is
pair breaking and a field perpendicular to d is not, and d=(1,i,0) pairs the
up spins alone. The normal-state dispersion xi(k) is read from the spinless
Hamiltonian, so only the pairing side is taken from a formula."""
import numpy as np
import pytest

from pyqula import geometry

rng = np.random.default_rng(3)


def _eig(h, k):
    return np.sort(np.linalg.eigvalsh(np.array(h.get_hk_gen()(k))))


def _xi(g, mu, k):
    hn = g.get_hamiltonian(has_spin=False)
    hn.add_onsite(mu)
    return np.linalg.eigvalsh(np.array(hn.get_hk_gen()(k)))[0]


def test_swave_with_a_zeeman_field_in_any_direction():
    """E = +-sqrt(xi^2+Delta^2) +- |B|, the field only splits the
    Bogoliubov bands of a spin-singlet"""
    g = geometry.chain()
    mu, D, B = 0.37, 0.21, np.array([0.1, -0.25, 0.18])
    h = g.get_hamiltonian()
    h.add_onsite(mu)
    h.add_zeeman(list(B))
    h.add_swave(D)
    b = np.linalg.norm(B)
    for k in rng.uniform(size=(4, 3)):
        s = np.sqrt(_xi(g, mu, k)**2 + D**2)
        assert np.allclose(_eig(h, k), np.sort([s+b, s-b, -s+b, -s-b]),
                           atol=1e-12)


def test_anderson_theorem_for_a_time_reversal_symmetric_normal_state():
    """For any time-reversal symmetric H and a uniform onsite singlet,
    H_BdG^2 = (H^2 + Delta^2), so E = +-sqrt(eps_n^2 + Delta^2) band by
    band, here with Kane-Mele, Rashba and an onsite modulation"""
    g = geometry.honeycomb_lattice().supercell(2)
    D = 0.3
    h = g.get_hamiltonian()
    h.add_kane_mele(0.1)
    h.add_rashba(0.3)
    h.add_onsite(lambda r: np.sin(3*r[0]) + 0.5*np.cos(2*r[1]))
    hn = h.copy()
    h.add_swave(D)
    for k in rng.uniform(size=(3, 3)):
        e = np.linalg.eigvalsh(np.array(hn.get_hk_gen()(k)))
        s = np.sqrt(e**2 + D**2)
        assert np.allclose(_eig(h, k), np.sort(np.concatenate([s, -s])),
                           atol=1e-12)


@pytest.mark.parametrize("ia", [0, 1, 2])
@pytest.mark.parametrize("ib", [0, 1, 2])
def test_triplet_is_pair_broken_only_by_a_field_along_d(ia, ib):
    """Chain p-wave with |Delta_k| = 2 Delta |sin k|. B || d gives
    E = +-|B| +- sqrt(xi^2+|Delta_k|^2), B perpendicular to d gives
    E = +-sqrt((xi +- |B|)^2 + |Delta_k|^2)"""
    g = geometry.chain()
    mu, D, b = 0.3, 0.25, 0.4
    ax = np.identity(3)
    h = g.get_hamiltonian()
    h.add_onsite(mu)
    h.add_zeeman(list(b*ax[ib]))
    h.add_pairing(mode="pwave", delta=D, d=list(ax[ia]))
    for k in rng.uniform(size=(4, 3)):
        x = _xi(g, mu, k)
        dk = 2*D*abs(np.sin(2*np.pi*k[0]))
        if ia == ib:
            s = np.sqrt(x**2 + dk**2)
            ref = [s+b, s-b, -s+b, -s-b]
        else:
            sp, sm = np.sqrt((x+b)**2 + dk**2), np.sqrt((x-b)**2 + dk**2)
            ref = [sp, sm, -sp, -sm]
        assert np.allclose(_eig(h, k), np.sort(ref), atol=1e-12)


def test_d_equal_1_i_0_pairs_the_up_spins_alone():
    """Delta_uu = -dx + i dy and Delta_dd = dx + i dy, so d=(1,i,0) pairs
    only the up spins: with B along +z the down electrons stay unpaired,
    E = +-(xi - B), and the up ones give +-sqrt((xi+B)^2 + |2 Delta_k|^2).
    The non-unitarity q = i d x d^* points along +z, the pair spin"""
    g = geometry.chain()
    mu, D, b = 0.3, 0.25, 0.4
    h = g.get_hamiltonian()
    h.add_onsite(mu)
    h.add_zeeman([0., 0., b])
    h.add_pairing(mode="pwave", delta=D, d=[1., 1j, 0.])
    for k in rng.uniform(size=(6, 3)):
        x = _xi(g, mu, k)
        s = np.sqrt((x+b)**2 + (4*D*np.sin(2*np.pi*k[0]))**2)
        assert np.allclose(_eig(h, k), np.sort([x-b, b-x, s, -s]),
                           atol=1e-12)
    q = h.get_dvector_non_unitarity(nk=6)[0]
    assert q[2] > 0.1 and abs(q[0]) < 1e-12 and abs(q[1]) < 1e-12


@pytest.mark.parametrize("mode,f", [
    ("chiral_pwave", lambda k: 4*(np.sin(k[0])**2 + np.sin(k[1])**2)),
    ("dx2y2", lambda k: 4*(np.cos(k[0]) - np.cos(k[1]))**2),
    ("extended_swave", lambda k: 4*(np.cos(k[0]) + np.cos(k[1]))**2)])
def test_square_lattice_dispersions_away_from_half_filling(mode, f):
    """E = +-sqrt(xi^2 + Delta^2 f(k)) with the first-neighbor form factor
    of each channel, for a generic unit d-vector"""
    g = geometry.square_lattice()
    mu, D = -0.7, 0.3
    d = np.array([0.3, -0.5, 0.8])
    d = d/np.linalg.norm(d)
    h = g.get_hamiltonian()
    h.add_onsite(mu)
    h.add_pairing(mode=mode, delta=D, d=list(d))
    for k in rng.uniform(size=(6, 3)):
        s = np.sqrt(_xi(g, mu, k)**2 + D**2*f(2*np.pi*k))
        assert np.allclose(_eig(h, k), np.sort([s, s, -s, -s]), atol=1e-12)
