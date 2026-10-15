import numpy as np
import pytest

from pyqula import geometry
from pyqula.qpitk import tmatrix


def _qpi(h, **kwargs):
    kw = dict(mode="tmatrix", nsuper=1, nk=24, energies=[0.4], delta=0.1,
            write=False)
    kw.update(kwargs)
    return h.get_qpi(**kw)


@pytest.mark.parametrize("v", [0.3, 5.0])
def test_tmatrix_is_exact_on_the_torus(v):
    """With nk points per direction, the T-matrix describes one impurity
    on an nk x nk torus, which get_qpi_impurity diagonalizes exactly with
    nk=1 (only the Gamma point of the supercell). The two must agree to
    numerical precision at every q but q=0, where the real-space one also
    holds the clean DOS, for a weak and a strong impurity. The honeycomb
    lattice has two sites per cell, so this also fixes the phase
    exp(-i q.tau) of the sublattice and the order of G0(-n) T G0(n): the
    reversed order fails it"""
    g = geometry.honeycomb_lattice()
    h = g.get_hamiltonian(has_spin=False)
    n, e, d = 6, 0.4, 0.15
    r, ld, q, qq = h.get_qpi_impurity(nsuper=n, energies=e, delta=d, nk=1,
            impurities=[{"index": 0, "onsite": v}], num_waves=2*n*n-1,
            write=False)
    b = tmatrix.reciprocal_vectors(g)
    qred = np.linalg.solve(b.T, (q[:, 0:2]/(2*np.pi)).T).T
    qred = np.concatenate([qred, 0*qred[:, :1]], axis=1)
    out = tmatrix.get_qpi_tmatrix(h, qred, [e], nk=n, delta=d, impurity=v)
    mask = np.linalg.norm(q, axis=1) > 1e-6
    assert np.max(qq[mask]) > 1e-3 # a real signal
    assert np.allclose(out[0][0][mask], qq[mask], rtol=1e-6, atol=1e-9)


def test_born_limit():
    """The Born approximation T=V is the first order of the T-matrix, so
    the difference between the two is of second order in the impurity:
    five times weaker impurity, about 25 times smaller difference
    relative to a signal five times smaller, so a ratio of relative
    differences close to 1/5. The Born QPI is linear in V"""
    g = geometry.square_lattice()
    h = g.get_hamiltonian(has_spin=False)
    def reldiff(v):
        qt = _qpi(h, impurity=v)[2]
        qb = _qpi(h, impurity=v, mode="born")[2]
        return np.linalg.norm(qt-qb)/np.linalg.norm(qb)
    r1, r2 = reldiff(0.1), reldiff(0.02)
    assert 0.15 < r2/r1 < 0.25
    qb1 = _qpi(h, impurity=0.1, mode="born")[2]
    qb2 = _qpi(h, impurity=0.2, mode="born")[2]
    assert np.allclose(qb2, 2*qb1, rtol=1e-8)


def test_square_lattice_point_group():
    """A point impurity on the square lattice keeps the C4v symmetry, so
    the QPI map is invariant under the rotation by 90 degrees and the
    mirrors, also with a second-neighbor hopping that bends the Fermi
    surface"""
    g = geometry.square_lattice()
    h = g.get_hamiltonian(has_spin=False, tij=[1.0, 0.3])
    q, es, qpi = _qpi(h, impurity=2.0, nk=20, nsuper=2, energies=[0.7])
    m = np.rint(q[:, 0:2]*20).astype(int)
    index = {tuple(mi): i for i, mi in enumerate(m)}
    ops = [lambda a, b: (-b, a), lambda a, b: (b, a), lambda a, b: (-a, b)]
    n = 0
    for i, (a, b) in enumerate(m):
        for op in ops:
            j = index.get(op(a, b))
            if j is None: continue
            assert abs(qpi[0, i]-qpi[0, j]) < 1e-9*np.max(qpi)
            n += 1
    assert n > len(m) # most of the points were compared


def test_honeycomb_threefold_rotation():
    """An impurity on one site of the honeycomb lattice keeps the
    threefold rotation around that site, so |drho(q)| is invariant under
    the rotation of q by 120 degrees; this holds only with the phase
    exp(-i q.tau) of each sublattice taken at its true position"""
    g = geometry.honeycomb_lattice()
    h = g.get_hamiltonian(has_spin=False)
    q, es, qpi = _qpi(h, impurity=1.5, nk=18, nsuper=2)
    b = tmatrix.reciprocal_vectors(g)
    qc = q[:, 0:2]@b # Cartesian, in units of 2pi
    c, s = np.cos(2*np.pi/3), np.sin(2*np.pi/3)
    qr = qc@np.array([[c, s], [-s, c]])
    n = 0
    for i in range(len(qc)):
        dist = np.linalg.norm(qc-qr[i], axis=1)
        j = np.argmin(dist)
        if dist[j] > 1e-8: continue
        assert abs(qpi[0, i]-qpi[0, j]) < 1e-8*np.max(qpi)
        n += 1
    assert n > len(qc)//2


def test_spin_and_nambu_bookkeeping():
    """A scalar impurity in a spin-degenerate system gives twice the
    spinless QPI, and adding the Nambu degree of freedom without pairing
    changes nothing, since only the electron sector enters the LDOS"""
    g = geometry.square_lattice()
    h0 = g.get_hamiltonian(has_spin=False)
    h1 = g.get_hamiltonian(has_spin=True)
    h2 = h1.copy() ; h2.setup_nambu_spinor()
    q0 = _qpi(h0, impurity=1.0)[2]
    q1 = _qpi(h1, impurity=1.0)[2]
    q2 = _qpi(h2, impurity=1.0)[2]
    assert np.allclose(q1, 2*q0, rtol=1e-8, atol=1e-12)
    assert np.allclose(q2, q1, rtol=1e-8, atol=1e-12)


def test_magnetic_impurity_cancels_in_born_with_time_reversal():
    """With time-reversal symmetry, the Kramers partners give opposite
    contributions of a time-reversal-odd (magnetic) impurity, so its Born
    QPI vanishes even with a Rashba spin-orbit coupling that mixes the
    spin (Sec. IV.4.2 of arXiv:2607.22487). The T-matrix brings it back
    at second order, and a scalar impurity is not suppressed"""
    g = geometry.square_lattice()
    h = g.get_hamiltonian(has_spin=True)
    h.add_rashba(0.4)
    n = h.intra.shape[0]
    m = np.zeros((n, n), dtype=complex)
    sx = np.array([[0, 1], [1, 0]]) ; sz = np.array([[1, 0], [0, -1]])
    m[0:2, 0:2] = 0.3*(0.6*sx+0.8*sz) # a generic direction
    qm = _qpi(h, impurity=m, mode="born")[2]
    qs = _qpi(h, impurity=0.3, mode="born")[2]
    qt = _qpi(h, impurity=m)[2]
    assert np.max(qs) > 1e-3
    assert np.max(qm) < 1e-10*np.max(qs)
    assert np.max(qt) > 1e-2*np.max(qs)


def test_wrong_inputs_are_refused():
    g = geometry.square_lattice()
    h = g.get_hamiltonian(has_spin=False)
    with pytest.raises(ValueError, match="'tmatrix'"):
        h.get_qpi(mode="tmatrixx", write=False)
    with pytest.raises(ValueError, match="dimension"):
        _qpi(h, impurity=np.identity(3))
    with pytest.raises(ValueError, match="needs an impurity"):
        _qpi(h)
    with pytest.raises(ValueError, match="does not take an impurity"):
        h.get_qpi(mode="pm", impurity=1.0, write=False)
