import numpy as np
import pytest
from scipy.linalg import expm

from pyqula import geometry
from pyqula import superconductivity
from pyqula.sctk import dvector

# The d-vector convention is the standard one, Delta(k) = i psi sigma_y +
# i (d.sigma) sigma_y in H = 1/2 sum_k Psi^dag H(k) Psi with Psi = (c_k,
# c^dag_-k) (Sato and Ando, arXiv:1608.03395, Eqs. 72, 78 and 81). pyqula's
# spinor (c_up, c_dn, c^dag_dn, -c^dag_up) carries a factor i sigma_y in the
# hole half, so its electron-hole block is simply psi + d.sigma, and a spin
# rotation acts on both halves with the same SU(2) matrix. So the d-vector
# rotates like a magnetization: U H[m,d] U^dag = H[Rm,Rd].

s0 = np.identity(2, dtype=np.complex128)
sx = np.array([[0., 1.], [1., 0.]], dtype=np.complex128)
sy = np.array([[0., -1j], [1j, 0.]], dtype=np.complex128)
sz = np.array([[1., 0.], [0., -1.]], dtype=np.complex128)
sig = [sx, sy, sz]


def _rotation(seed=0):
    """A random spin rotation, as its SU(2) matrix u and the SO(3) matrix R
    with u (v.sigma) u^dag = (R v).sigma"""
    rng = np.random.default_rng(seed)
    n = rng.normal(size=3)
    n = n/np.linalg.norm(n)
    u = expm(-1j*1.234/2.*sum(n[i]*sig[i] for i in range(3)))
    R = np.array([[0.5*np.trace(sig[a]@u@sig[b]@u.conj().T).real
                   for b in range(3)] for a in range(3)])
    return u, R


def _random_d(seed):
    rng = np.random.default_rng(seed)
    return rng.normal(size=3) + 1j*rng.normal(size=3) # generically non-unitary


def test_chain_spectrum_matches_standard_basis_bdg():
    """p-wave on a chain with a complex, non-unitary d: the BdG spectrum
    equals that of Delta(k) = (d(k).sigma) i sigma_y in the standard basis,
    which is also E^2 = xi^2 + |d|^2 +- |i d x d^*| (Sigrist and Ueda), and
    the extracted d(k) is the input d times the bond form factor"""
    delta0 = 0.37
    d0 = _random_d(1)
    g = geometry.chain()
    h0 = g.get_hamiltonian()
    h = g.get_hamiltonian()
    h.add_pairing(delta=delta0, mode="pwave", d=list(d0))
    hk0 = h0.get_hk_gen()
    hk = h.get_hk_gen()
    fd = dvector.extract_dvector_from_hamiltonian(h)
    for k in np.linspace(0., 1., 13):
        xi = np.linalg.eigvalsh(hk0([k, 0., 0.]))[0] # spin-degenerate band
        # e^{i phi(r1-r2)} on the two bonds of the chain gives -2i sin(2 pi k)
        dk = delta0*(-2j*np.sin(2.*np.pi*k))*d0
        Dk = sum(dk[i]*sig[i] for i in range(3))@(1j*sy)
        Hstd = np.block([[xi*s0, Dk], [Dk.conj().T, -xi*s0]])
        e = np.linalg.eigvalsh(hk([k, 0., 0.]))
        assert np.allclose(e, np.linalg.eigvalsh(Hstd), atol=1e-10)
        q = np.linalg.norm((1j*np.cross(dk, dk.conj())).real)
        esu = np.sqrt(xi**2 + np.vdot(dk, dk).real + np.array([-q, q]))
        assert np.allclose(e, np.sort(np.concatenate([-esu, esu])), atol=1e-10)
        assert np.allclose(fd([k, 0., 0.])[:, 0, 0], dk, atol=1e-10)


@pytest.mark.parametrize("lattice,mode", [
    (geometry.honeycomb_lattice, "pwave"),
    (geometry.triangular_lattice, "chiral_pwave"),
    (geometry.triangular_lattice, "chiral_fwave"),
    ])
def test_dvector_rotates_like_a_magnetization(lattice, mode):
    """Rotating the exchange field and the d-vector together by R is the
    spin rotation U = 1 x u of the whole BdG matrix, and the extracted
    d-vector, the non-unitarity q and the moment all come back rotated by
    that same R"""
    u, R = _rotation(2)
    g = lattice()
    n = len(g.r)
    U = np.kron(np.identity(2*n), u) # same u on (e_up,e_dn) and (h_dn,h_up)
    d0 = _random_d(3)
    m0 = np.array([0.3, -0.2, 0.4])
    def build(m, d):
        h = g.get_hamiltonian()
        h.add_exchange(list(m))
        h.add_pairing(delta=0.3, mode=mode, d=list(d))
        return h
    hA = build(m0, d0)
    hB = build(R@m0, R@d0)
    hkA = hA.get_hk_gen()
    hkB = hB.get_hk_gen()
    fA = dvector.extract_dvector_from_hamiltonian(hA)
    fB = dvector.extract_dvector_from_hamiltonian(hB)
    rng = np.random.default_rng(4)
    for k in rng.random((4, 3)):
        assert np.allclose(hkB(k), U@hkA(k)@U.conj().T, atol=1e-10)
        assert np.allclose(fB(k), np.einsum("ab,bij->aij", R, fA(k)),
                           atol=1e-10)
    qA = hA.get_dvector_non_unitarity(nk=4)
    qB = hB.get_dvector_non_unitarity(nk=4)
    assert np.max(np.abs(qA)) > 1e-3 # d is non-unitary, so this is a test
    assert np.allclose(qB, qA@R.T, atol=1e-10)
    mA = hA.get_magnetization(nk=4)
    mB = hB.get_magnetization(nk=4)
    assert np.max(np.abs(mA)) > 1e-3
    assert np.allclose(mB, mA@R.T, atol=1e-8)


def test_site_resolved_non_unitarity_is_that_of_delta_delta_dagger():
    """On a three-site cell with a d-vector that differs on every bond, the
    q of each site is the Brillouin-zone average of the spin part of
    (Delta Delta^dag)_ii, read off the raw electron-hole block of H(k), and
    the sum of i d x d^* over the bonds of that site. The three sites have
    different q, so summing the wrong index of the pairing matrix fails"""
    g = geometry.chain().supercell(3)
    n = len(g.r)
    def dfun(r):
        p = 2.*np.pi*r[0]/3.
        return np.array([1.+0.4*np.sin(p), 1j*(0.6+0.5*np.cos(p)),
                         0.3*np.cos(2.*p)])
    h = g.get_hamiltonian()
    h.add_pairing(delta=0.3, mode="pwave", d=dfun)
    q = h.get_dvector_non_unitarity(nk=6)
    hk = h.get_hk_gen()
    e = [4*i+s for i in range(n) for s in (0, 1)]
    o = [4*i+s for i in range(n) for s in (2, 3)]
    ks = g.get_kmesh(nk=6)
    qref = np.zeros((n, 3))
    for k in ks:
        D = np.array(hk(k))[np.ix_(e, o)]
        DD = D@D.conj().T
        for i in range(n):
            b = DD[2*i:2*i+2, 2*i:2*i+2]
            qref[i] += [0.5*np.trace(s@b).real/len(ks) for s in sig]
    qbond = np.zeros((n, 3))
    for i in range(n):
        for dx in (-1., 1.):
            d = 0.3*dfun(g.r[i]+np.array([dx/2., 0., 0.]))
            qbond[i] += (1j*np.cross(d, d.conj())).real
    assert np.min(np.abs(np.diff(q[:, 2]))) > 1e-2 # the sites differ
    assert np.allclose(q, qref, atol=1e-10)
    assert np.allclose(q, qbond, atol=1e-10)
    # and a unitary d, a real vector times a phase, has none
    h = g.get_hamiltonian()
    h.add_pairing(delta=0.3, mode="pwave",
                  d=np.exp(0.7j)*np.array([0.2, -0.5, 0.9]))
    assert np.allclose(h.get_dvector_non_unitarity(nk=6), 0., atol=1e-12)


def test_pairing_operators_form_a_vector():
    """The onsite operators deltax, deltay, deltaz have electron-hole blocks
    sigma_x/2, sigma_y/2, sigma_z/2, the ones d.sigma couples to, so they
    rotate into each other as a vector. deltay used to be -sigma_y/2.
    Being onsite, their ground-state expectation vanishes on a one-orbital
    lattice by Fermi antisymmetry; they are only informative k-resolved or
    state by state."""
    ops = [np.array(getattr(superconductivity, name).todense())
           for name in ("deltax", "deltay", "deltaz")]
    for (op, s) in zip(ops, sig):
        assert np.allclose(op[0:2, 2:4], s/2.)
        assert np.allclose(op[2:4, :], 0.) and np.allclose(op[:, 0:2], 0.)
    u, R = _rotation(5)
    U = np.kron(np.identity(2), u)
    for a in range(3):
        rotated = U.conj().T@ops[a]@U
        assert np.allclose(rotated, sum(R[a, b]*ops[b] for b in range(3)))
