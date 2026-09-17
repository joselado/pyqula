"""The prefactor of the BOND (d != 0) anomalous mean field.

The onsite pairing channel is pinned against the BCS gap equation of the
attractive Hubbard model. The bond channel had no such oracle: its
prefactor was only derived and checked for SU(2) equivariance, and a
uniform factor on it would survive both. The stationarity test in
test_bdg_total_energy.py does not catch it either, since it evaluates the
pairing energy with get_mf_bdg itself, so a wrong factor scales the energy
and the field together and the state stays stationary.

Here the mean-field energy is written out by hand, independently of the
SCF kernels. For H_int = 1/2 sum W_ab(d) n_a(0) n_b(d) (bare_interaction's
convention, W = 2 h.V), Wick's theorem on the BdG ground state gives

    E_int = 1/2 sum_d sum_ab W_ab(d) [ <n_a><n_b> - |<c_a^dag(0) c_b(d)>|^2
                                       + |<c_b(d) c_a(0)>|^2 ]

with the correlators taken from a direct diagonalization of H_BdG(k). The
self-consistent state is a stationary point of <H0 - mu N> + E_int at fixed
mu, so scaling the converged anomalous field by lambda must leave the
energy flat at lambda=1, meaning the slope falls as eps^2. A bond pairing
field off by a uniform factor would give a first-order slope instead, and
the controls check that the test would see it."""
import numpy as np

from pyqula import geometry, meanfield
from pyqula.multihopping import MultiHopping
from pyqula.sctk.reorder import nambu2block, block2nambu
from pyqula.superconductivity import get_eh_sector
from pyqula.bsetk.interaction import bare_interaction


def _dense(m):
    return np.array(m.todense()) if hasattr(m, "todense") else np.array(m)


def _block(m):
    """Nambu matrix in the (electrons | holes) block order"""
    return _dense(nambu2block(np.array(m, dtype=np.complex128)))


def _part(m, which):
    """The anomalous or the normal (electron and hole) part of a Nambu
    matrix"""
    b = _block(m)
    n = b.shape[0]//2
    out = np.zeros_like(b)
    if which == "anomalous":
        out[:n, n:], out[n:, :n] = b[:n, n:], b[n:, :n]
    else:
        out[:n, :n], out[n:, n:] = b[:n, :n], b[n:, n:]
    return _dense(block2nambu(out))


def _energy(H, H0, W, nk, cfock=1.0, cpair=1.0):
    """<H0 - mu N> + E_int of the BdG ground state of H, from its own
    diagonalization. For P = sum_{E<0} |u><u| in the block basis
    (c_k | c_-k^dag), <c_ka^dag c_kb> = P[b,a] and <c_-kb c_ka> = +-P[a,n+b];
    the sign and the order of the hole spins drop out of |.|^2 for a
    spin-independent W, which is asserted by the caller."""
    ks = H.geometry.get_kmesh(nk=nk)
    hk, hk0 = H.get_hk_gen(), H0.get_hk_gen()
    ds = [tuple(int(x) for x in d) for d in W]
    G = {d: 0. for d in ds + [(0, 0, 0)]}
    F = {d: 0. for d in ds}
    e0 = 0.
    for k in ks:
        m = _block(hk(k))
        n = m.shape[0]//2
        es, U = np.linalg.eigh(m)
        P = U[:, es < 0]@U[:, es < 0].conj().T
        e0 += np.sum(_block(hk0(k))[:n, :n]*P[:n, :n])/len(ks)
        for d in G:
            G[d] = G[d] + P[:n, :n].T*np.exp(2j*np.pi*np.dot(k, d))/len(ks)
        for d in F:
            F[d] = F[d] + P[:n, n:]*np.exp(-2j*np.pi*np.dot(k, d))/len(ks)
    dens = np.diag(G[(0, 0, 0)]).real
    eint = 0.
    for d, Wd in zip(ds, W.values()):
        Wd = np.array(Wd)
        eint += 0.5*np.sum(Wd*np.outer(dens, dens))
        eint -= cfock*0.5*np.sum(Wd*np.abs(G[d])**2)
        eint += cpair*0.5*np.sum(Wd*np.abs(F[d])**2)
    return (e0 + eint).real


def _check(guess, triplet, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    nk, filling = 20, 0.3
    h = geometry.chain().get_hamiltonian()
    h.setup_nambu_spinor()
    s = meanfield.Vinteraction(h.copy(), V1=-2.5, filling=filling, nk=nk,
            mf=guess, mix=0.5, maxerror=1e-12, maxite=5000, load_mf=False,
            verbose=0)
    hs = s.hamiltonian.get_multicell().get_dense()
    mu = hs.fermi
    H0 = h.get_multicell().get_dense()
    H0.shift_fermi(-mu)
    W = bare_interaction(hs)
    # the premise: a spin-independent bond interaction and nothing onsite
    for d, Wd in W.items():
        Wd = np.array(Wd)
        if tuple(d) == (0, 0, 0): assert np.max(np.abs(Wd)) < 1e-12
        else: assert np.max(np.abs(Wd - Wd[0, 0])) < 1e-12
    X = {k: np.array(m) for k, m in hs.get_multihopping().get_dict().items()}
    X0 = H0.get_multihopping().get_dict()
    mf = {k: m - np.array(X0.get(k, 0.*m)) for k, m in X.items()}
    anomalous = {k: _part(m, "anomalous") for k, m in mf.items()}
    fock = {k: (_part(m, "normal") if tuple(k) != (0, 0, 0) else 0.*m)
            for k, m in mf.items()}
    # the premise: bond pairing only, of the intended parity
    assert np.max(np.abs(get_eh_sector(X[(0, 0, 0)], i=0, j=1))) < 1e-10
    assert max(np.abs(m).max() for m in anomalous.values()) > 0.3
    d = np.array(hs.get_average_dvector())
    if triplet: assert abs(d[2]) > 0.3 and np.max(np.abs(d[:2])) < 1e-3
    else: assert np.max(np.abs(d)) < 1e-3
    # the hand-written functional is the SCF's own total energy
    nel = filling*2*len(hs.geometry.r)
    assert abs(_energy(hs, H0, W, nk) + mu*nel - s.total_energy) < 1e-8
    def slope(P, eps, **kwargs):
        es = []
        for lam in (1-eps, 1+eps):
            H = hs.copy()
            H.set_multihopping(MultiHopping(X) + (lam-1)*MultiHopping(P))
            es.append(_energy(H, H0, W, nk, **kwargs))
        return abs(es[1]-es[0])/(2*eps)
    for P in (anomalous, fock):
        s3, s4 = slope(P, 1e-3), slope(P, 1e-4)
        assert s4 < 1e-7 and s4 < s3/30  # zero slope, eps^2 residue
    # the check discriminates a uniform factor on either channel
    for c in (2.0, 0.5):
        assert slope(anomalous, 1e-3, cpair=c) > 1e-2
        assert slope(fock, 1e-3, cfock=c) > 1e-2


def test_extended_swave_bond_pairing_is_stationary(tmp_path, monkeypatch):
    """Attractive V1 on a doped chain, singlet (extended s-wave) state"""
    _check("swave", False, tmp_path, monkeypatch)


def test_triplet_bond_pairing_is_stationary(tmp_path, monkeypatch):
    """The same interaction converged to the triplet (p-wave) state"""
    _check("pwave", True, tmp_path, monkeypatch)
