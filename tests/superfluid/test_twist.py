"""The twist convention itself, checked block by block.

This is the one thing the analytic-vs-finite-difference cross-check
*cannot* catch: both routes share the same twisted Hamiltonian, so a wrong
tau_z mask (wrong Nambu block ordering, sign on the wrong block, anomalous
entries leaking in) or a wrong bond vector would make them agree and both
be wrong.  Here the twisted Bloch matrix is compared against h(k+Q) and
-Theta h(-k+Q) Theta^-1 built independently from the normal-state
Hamiltonian, up to the diagonal position phases that separate pyqula's
lattice (cell) gauge from the atomic gauge the twist is written in."""
import numpy as np
from scipy.sparse import bmat

from pyqula import geometry
from pyqula.hamiltonians import sy
from pyqula.sctk import superfluidweight as sw


def _time_reversal(m):
    """Theta m Theta^-1 with Theta = i sigma_y K, on a spinful matrix"""
    ns = m.shape[0]//2
    msy = bmat([[sy if i == j else None for j in range(ns)]
                for i in range(ns)]).todense()
    return np.asarray(msy@np.conjugate(m)@msy)


def _models():
    g = geometry.honeycomb_lattice()
    h = g.get_hamiltonian()
    h.add_onsite(0.3)
    h.add_rashba(0.2)   # complex, spin-mixing hoppings
    yield h
    g = geometry.square_lattice()
    h = g.get_hamiltonian()
    h.add_zeeman([0.1, 0.2, 0.3])
    yield h


def _reduced(g, Q):
    """Reduced coordinates of a Cartesian wavevector, 2 pi q_i = Q.a_i"""
    return np.array([Q.dot(a) for a in [g.a1, g.a2, g.a3]])/(2.*np.pi)


def _position_phases(h, Q):
    """diag(exp(i Q.r_a)) over the normal-state components of h"""
    r = np.repeat(np.array(h.geometry.r), h.intra.shape[0]//len(h.geometry.r),
                  axis=0)
    return np.diag(np.exp(1j*r@Q))


def test_twist_reproduces_electron_and_hole_blocks():
    """The Peierls phase exp(i tau Q.d_ij) on the stored hoppings must give
    h(k+Q) in the electron block and -Theta h(-k+Q) Theta^-1 in the hole
    block -- both conjugated by the diagonal position phases, because
    pyqula's Bloch matrices live in the lattice gauge -- with the anomalous
    block untouched."""
    rng = np.random.default_rng(0)
    for h0 in _models():
        n = h0.intra.shape[0]
        perm = None
        hbdg = h0.copy()
        hbdg.add_swave(0.25)
        f = sw.get_twisted_hk_gen(hbdg)
        perm = sw._nambu2block_permutation(hbdg.get_multicell())
        f0 = h0.get_hk_gen()
        for _ in range(5):
            k = np.array([rng.random(), rng.random(), 0.])
            Q = np.array([rng.random(), rng.random(), 0.])*0.3
            q = _reduced(h0.geometry, Q)
            lam = _position_phases(h0, Q)
            m = np.asarray(f(k, Q))[np.ix_(perm, perm)]
            ee = np.conjugate(lam).T@f0(k+q)@lam
            hh = -lam@_time_reversal(f0(-k+q))@np.conjugate(lam).T
            assert np.max(np.abs(m[0:n, 0:n] - ee)) < 1e-10
            assert np.max(np.abs(m[n:, n:] - hh)) < 1e-10


def test_twist_phase_is_the_peierls_phase_of_every_bond():
    """Element by element in real space: the electron entries of each stored
    hopping pick up exp(+i Q.d_ij) with the full bond vector d_ij = R + r_j
    - r_i, the hole entries exp(-i Q.d_ij), the anomalous entries nothing."""
    h = geometry.honeycomb_lattice().get_hamiltonian()
    h.add_onsite(0.3)
    h.add_swave(0.25)
    ops = sw.TwistOperators(h)
    Q = np.array([0.31, -0.17, 0.])
    tau, diag = sw.twist_masks(ops.h)
    r = sw.component_positions(ops.h)
    avecs = np.array([ops.geometry.a1, ops.geometry.a2, ops.geometry.a3])
    for (m, d, bond) in zip(ops.ms, ops.ds, ops.bonds):
        rv = d@avecs
        expected = rv[None, None, :] + r[None, :, :] - r[:, None, :]
        assert np.max(np.abs(bond-expected)) < 1e-12
    # and the twisted matrix is the elementwise product with that phase
    k = [0.13, 0.41, 0.]
    ref = np.zeros(h.intra.shape, dtype=np.complex128)
    for (m, d, bond) in zip(ops.ms, ops.ds, ops.bonds):
        ph = np.where(diag != 0., np.exp(1j*tau*(bond@Q)), 1.)
        ref = ref + m*ph*np.exp(1j*2.*np.pi*np.array(k).dot(d))
    assert np.max(np.abs(ops.hk(k, Q)-ref)) < 1e-12


def test_untwisted_generator_matches_the_ordinary_bloch_hamiltonian():
    """At Q=0 the twisted generator must be h.get_hk_gen() itself"""
    for h0 in _models():
        h = h0.copy()
        h.add_swave(0.2)
        f = sw.get_twisted_hk_gen(h)
        fk = h.get_hk_gen()
        for k in [[0., 0., 0.], [0.13, 0.27, 0.], [0.5, 0.5, 0.]]:
            assert np.max(np.abs(np.asarray(f(k, [0., 0., 0.]))
                                 - np.asarray(fk(k)))) < 1e-12


def test_anomalous_block_is_untouched_by_the_twist():
    """|Delta_ij| held fixed means the electron-hole blocks of every stored
    hopping are left alone, including the non-local ones of a p-wave
    superconductor."""
    h = geometry.square_lattice().get_hamiltonian()
    h.add_pairing(mode="pwave", delta=0.3, d=[0., 0., 1.])
    h.add_swave(0.1)
    perm = sw._nambu2block_permutation(h.get_multicell())
    n = h.intra.shape[0]//2
    f = sw.get_twisted_hk_gen(h)
    k = [0.17, 0.41, 0.]
    a0 = np.asarray(f(k, [0., 0., 0.]))[np.ix_(perm, perm)][0:n, n:]
    aq = np.asarray(f(k, [0.4, 0.2, 0.]))[np.ix_(perm, perm)][0:n, n:]
    assert np.max(np.abs(a0)) > 1e-3       # there *is* an anomalous block
    assert np.max(np.abs(a0 - aq)) < 1e-12  # and the twist does not move it
