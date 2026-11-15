from pyqula import geometry
from pyqula.superconductivity import hopping2deltaud


def test_hopping2deltaud_matches_add_pairing_swave():
    """Building an extended s-wave pairing via hopping2deltaud must give the
    same Hamiltonian as adding it directly with add_pairing."""
    g = geometry.honeycomb_lattice()
    g = g.get_supercell((2, 2))
    h = g.get_hamiltonian()
    h0 = h.copy()

    h1 = h0.copy()
    h1.add_pairing(mode="swave", nn=1, delta=0.1)
    h2 = hopping2deltaud(h0, h0 * 0.1)

    assert (h1 - h2).is_zero(), "hopping2deltaud disagrees with add_pairing(mode='swave')"


def test_hopping2deltaud_with_a_complex_hermitian_hopping():
    """For a Hermitian t the pairing sum t_ij c^dag_iup c^dag_jdn + h.c. is a
    valid operator for any t, with the symmetric part of t as its singlet
    and the antisymmetric part as a d_z triplet. The routine used t_R for
    the dn-up block too, which breaks Fermi antisymmetry as soon as t is
    complex; that block has to be (t_-R)^T, so in k-space the up-dn block is
    t(k) and the dn-up block t(-k)^T, and U H_R^* U^dag = -H_R holds."""
    import numpy as np
    from pyqula import algebra
    from pyqula.superconductivity import get_eh_sector
    g = geometry.honeycomb_lattice()
    T = g.get_hamiltonian(has_spin=False)*0.2
    T.add_haldane(0.15)
    h = hopping2deltaud(g.get_hamiltonian(), T)
    u4 = np.array([[0, 0, 0, -1], [0, 0, 1, 0], [0, 1, 0, 0], [-1, 0, 0, 0]],
                  dtype=complex)
    U = np.kron(np.identity(len(g.r)), u4)
    for m in h.get_multihopping().get_dict().values():
        m = np.array(algebra.todense(m))
        assert np.max(np.abs(U@np.conj(m)@U.T + m)) < 1e-12
    for k in [np.array([0.13, 0.29, 0.]), np.array([0.41, 0.07, 0.])]:
        tk = np.array(algebra.todense(T.get_hk_gen()(k)))
        tmk = np.array(algebra.todense(T.get_hk_gen()(-k)))
        D = np.array(algebra.todense(get_eh_sector(
            np.array(algebra.todense(h.get_hk_gen()(k))), i=0, j=1)))
        assert np.allclose(D[0::2, 0::2], tk, atol=1e-12)
        assert np.allclose(D[1::2, 1::2], tmk.T, atol=1e-12)
        assert np.allclose(D[0::2, 1::2], 0.) and np.allclose(D[1::2, 0::2], 0.)
        assert np.max(np.abs(tk - tmk.T)) > 0.05  # a triplet part is there


def test_hopping2deltaud_with_an_antisymmetric_hopping_is_a_pure_triplet():
    """1j times the anti-Haldane hopping is real and antisymmetric, so the
    pairing it gives is a d_z triplet with no singlet part at all, the
    second-neighbor f-wave of the honeycomb lattice"""
    import numpy as np
    from pyqula import algebra
    from pyqula.sctk.extract import (extract_singlet_pairing,
                                     extract_triplet_pairing)
    g = geometry.honeycomb_lattice()
    T = g.get_hamiltonian(has_spin=False)*0.
    T.add_antihaldane(0.2)
    h = hopping2deltaud(g.get_hamiltonian(), T*1j)
    for k in [np.array([0.13, 0.29, 0.]), np.array([0.41, 0.07, 0.])]:
        m = np.array(algebra.todense(h.get_hk_gen()(k)))
        assert np.max(np.abs(extract_singlet_pairing(m))) < 1e-12
        assert np.max(np.abs(extract_triplet_pairing(m)[2])) > 0.05


def test_updn_block_is_the_double_loop_over_the_pairs():
    """updn_block puts [[mud_ij,0],[0,mdu_ij]] on every pair of sites with
    one of the two at least 1e-6, as the double loop over the pairs of
    sites that it replaces did: a small entry is kept when its partner is
    large, and dropped when both are small"""
    import numpy as np
    from pyqula.sctk.fastdeltaud import updn_block
    rng = np.random.default_rng(1)
    n = 7
    mud = (rng.random((n, n)) - 0.5) + 1j*(rng.random((n, n)) - 0.5)
    mdu = (rng.random((n, n)) - 0.5) + 1j*(rng.random((n, n)) - 0.5)
    mud[rng.random((n, n)) < 0.4] = 0.
    mdu[rng.random((n, n)) < 0.4] = 0.
    mud[0, 1], mdu[0, 1] = 1e-8, 0.3 # kept, its partner is large
    mud[2, 3], mdu[2, 3] = 1e-8, 0. # dropped, both below 1e-6
    ref = np.zeros((2*n, 2*n), dtype=complex)
    for i in range(n):
        for j in range(n):
            if abs(mud[i, j]) < 1e-6 and abs(mdu[i, j]) < 1e-6: continue
            ref[2*i, 2*j] = mud[i, j]
            ref[2*i+1, 2*j+1] = mdu[i, j]
    out = updn_block(np.matrix(mud), np.matrix(mdu)).toarray()
    assert out.shape == (2*n, 2*n)
    assert np.array_equal(out, ref)
    assert out[0, 2] == 1e-8 # the pair (0,1) is kept
    assert out[4, 6] == 0. # the pair (2,3) is dropped
