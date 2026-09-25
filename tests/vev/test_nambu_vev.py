"""Expectation values on a Nambu (BdG) Hamiltonian.

The sum over the negative-energy BdG states runs over a particle-hole
redundant set, so a normal observable, which the Nambu lift puts both in the
electron-electron and in the hole-hole block, is counted twice unless the
hole-hole block is dropped, while a pairing operator lives in the
electron-hole blocks and must keep its value. get_vev used to restrict every
operator to the electron sector, which made every pairing operator zero,
and get_single_vev, get_several_vev and get_dm_vev applied no restriction at
all, so they returned twice the moment of a BdG copy of a normal state.
"""

import numpy as np
from scipy.sparse import csc_matrix

from pyqula import geometry, spectrum, superconductivity
from pyqula.operators import Operator


def dense(m):
    return m.toarray() if hasattr(m, "toarray") else np.array(m)


def blind(m):
    """An operator defined only by its action, with no matrix"""
    m = csc_matrix(m)
    out = Operator(lambda v, k=None: m @ v)
    assert out.matrix is None
    return out


def random_hermitian(n, seed=1):
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(n, n)) + 1j*rng.normal(size=(n, n))
    return a + a.conj().T


def nambu_lift(m):
    """A normal-basis operator in the Nambu basis, lifted the way the
    Hamiltonian is, so it has a hole-hole block as well"""
    return csc_matrix(superconductivity.build_eh(csc_matrix(m)))


def occupied_sum(h, m, nk):
    """sum over the negative-energy eigenstates of <psi|m|psi>, by hand"""
    hk = h.get_hk_gen()
    ks = h.geometry.get_kmesh(nk=nk) if h.dimensionality > 0 else [None]
    m = dense(m)
    out = 0.
    for k in ks:
        hm = dense(hk(k)) if k is not None else dense(h.intra)
        e, v = np.linalg.eigh(hm)
        v = v[:, e < 0.]
        out += np.trace(v.conj().T @ m @ v).real
    return out/len(ks)


def correlator(h, nk):
    """C[a,b] = <Psi_a^dag Psi_b> in the Nambu basis, by hand from the
    negative-energy eigenvectors of the Bloch BdG matrix"""
    hk = h.get_hk_gen()
    ks = h.geometry.get_kmesh(nk=nk) if h.dimensionality > 0 else [None]
    out = 0.
    for k in ks:
        hm = dense(hk(k)) if k is not None else dense(h.intra)
        e, v = np.linalg.eigh(hm)
        v = v[:, e < 0.]
        out = out + v.conj() @ v.T
    return out/len(ks)


def rashba_honeycomb():
    g = geometry.honeycomb_lattice()
    h = g.get_hamiltonian(has_spin=True)
    h.add_rashba(0.3)
    h.add_exchange([0.3, 0.5, 0.2])  # a generic direction: complex amplitudes
    h.add_onsite(0.2)
    return h


def rashba_island():
    g = geometry.chain().get_supercell(6)
    g.dimensionality = 0
    h = g.get_hamiltonian(has_spin=True)
    h.add_rashba(0.4)
    h.add_exchange([0.3, 0.5, 0.2])
    h.add_onsite(lambda r: 0.1*r[0])
    return h


def test_zero_pairing_periodic_counts_once():
    """A BdG copy at zero pairing describes the same state as the normal
    Hamiltonian, so every route has to give the normal value"""
    nk = 6
    h = rashba_honeycomb()
    hb = h.copy()
    hb.add_swave(0.0)
    names = ["sx", "sy", "sz"]
    normal = [h.get_operator(s).get_matrix() for s in names]
    nambu = [hb.get_operator(s).get_matrix() for s in names]
    m = random_hermitian(h.intra.shape[0])
    normal.append(m)
    nambu.append(nambu_lift(m))
    several = hb.get_several_vev(nambu, nk=nk)
    for (i, (mn, mb)) in enumerate(zip(normal, nambu)):
        ref = occupied_sum(h, mn, nk)  # the oracle, on the normal state
        assert abs(h.get_single_vev(mn, nk=nk) - ref) < 1e-8
        assert abs(hb.get_single_vev(mb, nk=nk) - ref) < 1e-8, i
        assert abs(several[i] - ref) < 1e-8, i
        assert abs(np.sum(hb.get_vev(mb, nk=nk)) - ref) < 1e-8, i
        # and an operator defined only by its action
        assert abs(hb.get_single_vev(blind(mb), nk=nk) - ref) < 1e-8, i
        assert abs(np.sum(hb.get_vev(blind(mb), nk=nk)) - ref) < 1e-8, i
    # the matrix-free branch of get_several_vev
    blinds = hb.get_several_vev([blind(mb) for mb in nambu], nk=nk)
    assert np.max(np.abs(blinds - several)) < 1e-8


def test_zero_pairing_island_counts_once(tmp_path, monkeypatch):
    """The same on a 0d island, where get_dm_vev and real_space_vev apply"""
    monkeypatch.chdir(tmp_path)  # real_space_vev writes a file
    h = rashba_island()
    hb = h.copy()
    hb.add_swave(0.0)
    names = ["sx", "sy", "sz"]
    normal = [h.get_operator(s).get_matrix() for s in names]
    nambu = [hb.get_operator(s).get_matrix() for s in names]
    m = random_hermitian(h.intra.shape[0], seed=2)
    normal.append(m)
    nambu.append(nambu_lift(m))
    several = hb.get_several_vev(nambu)
    for (i, (mn, mb)) in enumerate(zip(normal, nambu)):
        ref = occupied_sum(h, mn, 1)  # the oracle, on the normal state
        assert abs(h.get_dm_vev(mn) - ref) < 1e-8, i
        assert abs(hb.get_dm_vev(mb) - ref) < 1e-8, i
        assert abs(hb.get_dm_vev(blind(mb)) - ref) < 1e-8, i
        assert abs(hb.get_single_vev(mb) - ref) < 1e-8, i
        assert abs(several[i] - ref) < 1e-8, i
        assert abs(np.sum(hb.get_vev(mb)) - ref) < 1e-8, i
        rsv = spectrum.real_space_vev(hb, operator=mb, name="RSV.OUT")
        assert abs(np.sum(rsv) - ref) < 1e-8, i


def test_pairing_operator_per_site():
    """get_vev("spair") is, on each site, the anomalous correlator
    <c_up^dag c_dn^dag> + <c_dn^dag (-c_up^dag)> = 2<c_up^dag c_dn^dag>
    read off the Nambu correlation matrix, and its sum is get_single_vev"""
    nk = 20
    g = geometry.chain().get_supercell(2)
    h = g.get_hamiltonian(has_spin=True)
    h.add_onsite(lambda r: 0.3 if r[0] < 0.5 else -0.2)  # two inequivalent sites
    h.add_swave(0.4)
    c = correlator(h, nk)
    ref = np.array([c[4*i, 4*i+2] + c[4*i+1, 4*i+3]
                    for i in range(len(g.r))]).real
    assert np.min(np.abs(ref)) > 1e-2  # a genuine pairing amplitude
    assert abs(ref[0] - ref[1]) > 1e-3  # and not the same on both sites
    out = h.get_vev("spair", nk=nk)
    assert np.max(np.abs(out - ref)) < 1e-8
    single = h.get_single_vev("spair", nk=nk)
    assert abs(np.sum(out) - single) < 1e-8
    assert abs(h.get_several_vev(["spair"], nk=nk)[0] - single) < 1e-8
    # an operator defined only by its action, which used to be zero too
    p = h.get_operator("spair").get_matrix()
    assert np.max(np.abs(h.get_vev(blind(p), nk=nk) - ref)) < 1e-8
    # and with no pairing there is no anomalous correlator
    h0 = g.get_hamiltonian(has_spin=True)
    h0.add_swave(0.0)
    assert np.max(np.abs(h0.get_vev("spair", nk=nk))) < 1e-10


def test_pairing_single_site_fock_space():
    """A single site eps*(n_up+n_dn) + D c_up^dag c_dn^dag + h.c., solved in
    its four-state Fock space, gives |<c_dn c_up>| = D/(2E) with
    E = sqrt(eps^2+D^2)"""
    eps, d0 = 0.3, 0.4
    g = geometry.chain()
    g.dimensionality = 0
    h = g.get_hamiltonian(has_spin=True)
    h.add_onsite(eps)
    h.add_swave(d0)
    hm = dense(h.intra)
    d = hm[0, 2]  # the pairing as the BdG matrix stores it
    assert abs(abs(d) - d0) < 1e-12
    # the four-state Fock space, with Jordan-Wigner fermions
    a = np.array([[0., 1.], [0., 0.]])  # annihilates the occupied state
    z = np.diag([1., -1.])  # string: +1 empty, -1 occupied
    cup = np.kron(a, np.identity(2))
    cdn = np.kron(z, a)
    nup = cup.T @ cup
    ndn = cdn.T @ cdn
    pair = cup.T @ cdn.T  # c_up^dag c_dn^dag
    hf = eps*(nup + ndn) + d*pair + np.conjugate(d)*pair.T
    e, v = np.linalg.eigh(hf)
    gs = v[:, 0]
    corr = gs.conj() @ pair @ gs  # <c_up^dag c_dn^dag>
    energy = np.sqrt(eps**2 + d0**2)
    assert abs(abs(corr) - d0/(2*energy)) < 1e-12  # the closed form
    out = h.get_vev("spair")
    assert abs(out[0] - 2*corr.real) < 1e-8
    assert abs(h.get_single_vev("spair") - 2*corr.real) < 1e-8
    # the occupation keeps its value too
    assert abs(np.sum(h.get_vev()) - (gs.conj() @ (nup + ndn) @ gs).real) < 1e-8


def test_operator_with_both_parts():
    """A normal part plus a pairing part: the normal part is counted once
    and the pairing part keeps its value"""
    nk = 10
    g = geometry.chain().get_supercell(2)
    h = g.get_hamiltonian(has_spin=True)
    h.add_rashba(0.3)
    h.add_exchange([0.2, -0.4, 0.3])
    h.add_onsite(lambda r: 0.3 if r[0] < 0.5 else -0.2)
    h.add_swave(0.3)
    n = len(g.r)
    m = random_hermitian(2*n, seed=3)
    c = correlator(h, nk)
    e = [4*(i//2) + i % 2 for i in range(2*n)]  # electron entries of the spinor
    ref = np.sum(m*c[np.ix_(e, e)]).real  # sum_ab m_ab <c_a^dag c_b>
    lifted = nambu_lift(m)
    assert abs(h.get_single_vev(lifted, nk=nk) - ref) < 1e-8
    assert abs(np.sum(h.get_vev(lifted, nk=nk)) - ref) < 1e-8
    p = h.get_operator("spair").get_matrix()
    both = h.get_single_vev(lifted + p, nk=nk)
    assert abs(both - (ref + h.get_single_vev(p, nk=nk))) < 1e-8
