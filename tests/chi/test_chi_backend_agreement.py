import numpy as np

from pyqula import geometry, chi, algebra
from pyqula.chitk import chiAB

ES = np.linspace(-1., 1., 12)


def _chain():
    g = geometry.chain().supercell(2)
    h = g.get_hamiltonian(has_spin=True)
    h.add_exchange([0., 0., 0.2])
    return h


def _chiAB(h, ij_mode, T, **kwargs):
    return np.array(chiAB.chiAB_q(h, energies=ES, nk=8, delta=0.1, T=T,
                                  ij_mode=ij_mode, **kwargs)[1])


def test_accelerated_path_uses_the_temperature_it_was_given():
    """chiAB_full_matrix_jit built its occupations as (1-tanh(beta*E))/2,
    which is Fermi-Dirac at T/2, while chiAB_jit/chiAB_matrix use
    1/(1+exp(beta*E)) at T -- so ij_mode="accelerated" silently answered
    at half the requested temperature. The two loop modes must agree, and
    must NOT agree when one of them is run at half the temperature."""
    h = _chain()
    for T in [0.1, 0.05]:
        same = _chiAB(h, "explicit", T)
        accel = _chiAB(h, "accelerated", T)
        halved = _chiAB(h, "explicit", T / 2.)
        assert np.max(np.abs(same - accel)) < 1e-10, T
        # the T/2 gap shrinks fast as T falls below delta (both occupation
        # functions become step functions), so this margin is deliberately
        # loose: 1e-3 at T=0.1, 3e-6 at T=0.05
        assert np.max(np.abs(halved - accel)) > 1e-7, T


def test_accelerated_path_accepts_a_real_operator():
    """The numba kernel multiplies the operator against complex
    wavefunctions, and numba's @ refuses a mixed-dtype product, so a
    real-valued named operator failed to compile on this path."""
    h = _chain()
    explicit = _chiAB(h, "explicit", 0.1, A="sz", B="sz")
    accel = _chiAB(h, "accelerated", 0.1, A="sz", B="sz")
    assert np.max(np.abs(explicit - accel)) < 1e-10


def _make_sparse(h):
    from scipy.sparse import csc_matrix
    h.intra = csc_matrix(h.intra)
    h.inter = csc_matrix(h.inter)
    h.is_sparse = True
    return h


def test_sparse_hamiltonian_reaches_every_backend():
    """hk(k) is sparse for an is_sparse Hamiltonian, and np.array over
    those gives a dtype=object array. The GPU branch fed it straight to
    jax ("Dtype object is not a valid JAX array type") and the
    accelerated branch to scipy.linalg.eigh -- the hole hk_matrix_batch
    was added to close for the other numba paths."""
    dense = _chiAB(_chain(), "explicit", 0.1)
    for kwargs in [dict(ij_mode="explicit"),
                   dict(ij_mode="accelerated"),
                   dict(ij_mode="explicit", chi_cpugpu="GPU")]:
        mode = kwargs.pop("ij_mode")
        out = _chiAB(_make_sparse(_chain()), mode, 0.1, **kwargs)
        assert np.max(np.abs(dense - out)) < 1e-10, (mode, kwargs)


def _gauge_transform(h, phi):
    h2 = h.copy()
    m = h.intra
    m = np.array(m.todense() if hasattr(m, "todense") else m)
    u = np.diag(np.exp(1j * phi))
    h2.intra = u @ m @ np.conjugate(u).T
    return h2


def _lehmann(m, ii, jj, delta):
    """chi_ij(w) = sum_nm (f_n-f_m) <n|rho_i|m><m|rho_j|n>/(E_n-E_m-w+i d)"""
    (es, ws) = algebra.eigh(np.array(m))
    ws = np.transpose(ws)
    out = 0 * ES + 0j
    for n in range(len(es)):
        for k in range(len(es)):
            f = float(es[n] < 0.) - float(es[k] < 0.)
            if f == 0.:
                continue
            el = (np.conjugate(ws[n][ii]) * ws[k][ii]
                  * np.conjugate(ws[k][jj]) * ws[n][jj])
            out = out + f * el / (es[n] - es[k] - ES + 1j * delta)
    return out


def test_chargechi_matches_the_lehmann_matrix_element():
    """chi.elementchi formed psi_n(i)psi_m(i)conj(psi_n(j)psi_m(j))
    instead of conj(psi_n(i))psi_m(i)conj(psi_m(j))psi_n(j) -- the two
    coincide only for real amplitudes. chitk/static.py's elementchi and
    chitk/chiAB.py's chiAB_jit both already write it correctly."""
    g = geometry.chain().supercell(5)
    g.dimensionality = 0
    h = g.get_hamiltonian(has_spin=False)
    h.add_peierls(0.3)  # complex amplitudes
    (_, out) = chi.chargechi(h, i=0, j=3, es=ES, delta=0.05)
    m = h.intra
    m = m.todense() if hasattr(m, "todense") else m
    ref = _lehmann(m, 0, 3, 0.05)
    assert np.max(np.abs(np.array(out) - ref)) < 1e-10


def test_chargechi_is_gauge_invariant():
    """A site-local phase change of the Hamiltonian leaves the physical
    density-density response untouched. With the wrong conjugation it
    moved by 0.014 on values of order 0.2."""
    g = geometry.chain().supercell(5)
    g.dimensionality = 0
    h = g.get_hamiltonian(has_spin=False)
    h.add_peierls(0.3)
    h2 = _gauge_transform(h, np.array([0., 0.7, 1.3, -0.4, 2.1]))
    a = np.array(chi.chargechi(h, i=0, j=3, es=ES, delta=0.05)[1])
    b = np.array(chi.chargechi(h2, i=0, j=3, es=ES, delta=0.05)[1])
    assert np.max(np.abs(a)) > 1e-3  # not a vacuous comparison of zeros
    assert np.max(np.abs(a - b)) < 1e-10
