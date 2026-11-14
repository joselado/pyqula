import numpy as np
import pytest
from scipy import linalg

from pyqula import geometry
from pyqula.kondolattice import KondoLatticeHamiltonian
from pyqula.kondotk import pairs as pairstk
from pyqula.multihopping import MultiHopping

# A Kondo lattice mean field without a Zeeman field is the same for the
# two spins and does not mix them. The cotunneling spectrum and the
# composite fermion spectrum then diagonalize the block of one spin, half
# the size, and count every state twice. The result must be the one of the
# full matrix and the one of the Green function inverted at every energy,
# and a Hamiltonian with a field must still take the full matrix.

DELTA = 0.05


def kondo_chain(field=None):
    """Two Kondo pairs per cell with different conduction levels, f levels
    and complex hybridizations, set by hand, so that nothing is equal by
    symmetry"""
    hc = geometry.chain().get_supercell(2).get_hamiltonian(has_spin=True)
    hc.add_onsite(lambda r: 0.3*r[0])
    h = KondoLatticeHamiltonian(hc)
    if field is not None: h.add_zeeman(field)
    d = {k: np.array(m, dtype=np.complex128)
         for k, m in h.get_multihopping().get_dict().items()}
    V, lam = [0.5*np.exp(0.3j), 0.8*np.exp(-0.9j)], [0.1, -0.2]
    for (ci, fi), v, l in zip(h._kondo_pairs, V, lam):
        for s in (0, 1):
            d[(0, 0, 0)][2*fi+s, 2*fi+s] += l
            d[(0, 0, 0)][2*fi+s, 2*ci+s] += v
            d[(0, 0, 0)][2*ci+s, 2*fi+s] += np.conjugate(v)
    h.set_multihopping(MultiHopping(d))
    return h


def sizes_diagonalized(monkeypatch):
    """Record the size of every matrix that kondotk diagonalizes"""
    sizes = []
    eigh = linalg.eigh
    def recording(m, *args, **kwargs):
        sizes.append(m.shape[0])
        return eigh(m, *args, **kwargs)
    monkeypatch.setattr(linalg, "eigh", recording)
    return sizes


def didv_reference(h, energies, nk, tc, tf):
    """-Im <psi|G|psi>/pi summed over the spin, psi = tc c + tf f of each
    pair, with G the inverse of e + i delta - H(k) at every energy"""
    hk, n = h.get_hk_gen(), 2*len(h.geometry.r)
    out = np.zeros((len(h._kondo_pairs), len(energies)))
    for k in np.linspace(0., 1., nk, endpoint=False):
        m = np.array(hk([k, 0., 0.]))
        for ie, e in enumerate(energies):
            G = np.linalg.inv((e + 1j*DELTA)*np.identity(n) - m)
            for ip, (ci, fi) in enumerate(h._kondo_pairs):
                for s in (0, 1):
                    psi = np.zeros(n)
                    psi[2*ci+s], psi[2*fi+s] = tc, tf
                    out[ip, ie] += -(psi@G@psi).imag/np.pi/nk
    return out


@pytest.mark.parametrize("field", [None, [0.1, 0.05, 0.2]],
                         ids=["no field", "field"])
def test_cotunneling_is_the_inverted_green_function(field, monkeypatch):
    h = kondo_chain(field)
    sizes = sizes_diagonalized(monkeypatch)
    energies = np.linspace(-2.5, 2.5, 11)
    kw = dict(energies=energies, tc=1.0, tf=0.4, nk=8, delta=DELTA)
    e, d = h.get_cotunneling_didv(**kw)
    ref = didv_reference(h, energies, 8, 1.0, 0.4)
    assert np.max(np.abs(d - ref)) < 1e-12, np.max(np.abs(d - ref))
    # 4 sites: the block of one spin without a field, the full matrix with
    assert set(sizes) == ({4} if field is None else {8}), sizes
    assert len(sizes) == 8, sizes # one diagonalization per k-point


def test_half_block_gives_the_spectra_of_the_full_matrix(monkeypatch):
    """Every output of the two routines, with the block of one spin and
    with the full matrix forced on the same Hamiltonian"""
    h = kondo_chain()
    assert pairstk.is_spin_degenerate(h)
    kw = dict(energies=np.linspace(-2.5, 2.5, 21), delta=DELTA)
    def run():
        e, d, parts = h.get_cotunneling_didv(tc=1.0, tf=-0.6, nk=6, T=0.3,
                return_parts=True, **kw)
        out = dict(parts, didv=d)
        for green in (False, True):
            ks, e, c = h.get_composite_spectral_function(nk=7, site=[0, 1],
                    green=green, **kw)
            out.update({(key, green): val for key, val in c.items()})
        return out
    half = run()
    sizes = sizes_diagonalized(monkeypatch)
    monkeypatch.setattr(pairstk, "is_spin_degenerate", lambda h: False)
    full = run()
    # the full route: the 8 x 8 matrix and the 4 x 4 block of the moments
    assert set(sizes) == {8, 4} and sizes.count(8) > 6, sizes
    assert set(half) == set(full) and len(half) == 13
    for key in half:
        assert np.max(np.abs(half[key])) > 1e-3, key
        diff = np.max(np.abs(half[key] - full[key]))
        assert diff < 1e-12, (key, diff)


def test_composite_spectrum_diagonalizes_one_spin_block(monkeypatch):
    """Without a field: the 4 x 4 block of one spin and the 2 x 2 block of
    the moments of one spin, at every k-point. With a field: twice that."""
    for field, expected in ((None, {4, 2}), ([0.1, 0.05, 0.2], {8, 4})):
        h = kondo_chain(field)
        with monkeypatch.context() as mp:
            sizes = sizes_diagonalized(mp)
            h.get_composite_spectral_function(nk=5, delta=DELTA,
                    energies=np.linspace(-2., 2., 5))
        assert set(sizes) == expected, (field, sizes)
