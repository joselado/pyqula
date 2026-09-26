"""ARPACK eigenvectors are orthonormal inside a degenerate level.

scipy's eigsh hands a complex Hermitian matrix to the non-Hermitian eigs
driver, whose eigenvectors inside a degenerate level are some basis of the
eigenspace rather than an orthonormal one, so every sum over states counted
the same direction several times. The cases here take whole degenerate
levels, where the sum over the level is basis independent and has to agree
with a dense diagonalization exactly."""

import numpy as np
from scipy.linalg import eigh
from scipy.sparse import csc_matrix

from pyqula import geometry, algebra, ldos


def test_arpack_eigh_returns_orthonormal_vectors_of_a_degenerate_level():
    """A complex Hermitian matrix with a 4-fold level"""
    rng = np.random.RandomState(1)
    u = np.linalg.qr(rng.randn(12, 12) + 1j * rng.randn(12, 12))[0]
    es = np.array([0.1] * 4 + [0.5, -0.7, 1.3, 2., -2.2, 3., -3.1, 4.])
    m = csc_matrix(u @ np.diag(es) @ np.conjugate(u.T))
    e, v = algebra.arpack_eigh(m, k=6, sigma=0.0, which="LM")
    assert np.allclose(np.conjugate(v.T) @ v, np.identity(6), atol=1e-8)
    assert np.allclose(m @ v, v * e[None, :], atol=1e-8)
    assert np.allclose(e, np.sort(e))  # ascending
    assert np.allclose(np.sort(e), np.sort([0.1] * 4 + [0.5, -0.7]))


def test_unfolded_weight_of_whole_degenerate_levels_matches_dense(tmp_path,
                                                                  monkeypatch):
    """At the zone center of a 4x4 triangular supercell with a defect the
    levels are 8-fold, 1-fold and 5-fold, and the 14 states nearest to zero
    energy are those three levels whole. The unfolded weight summed over
    each level used to be off by up to 1.5 times through num_bands"""
    monkeypatch.chdir(tmp_path)
    g0 = geometry.triangular_lattice()
    g = g0.get_supercell(4, store_primal=True)
    h = g.get_hamiltonian(has_spin=False)
    h.add_onsite(lambda r: 2. * (np.sum((r - g.r[0])**2) < 1e-2))
    k = [-1., 1., 0.]
    dense = h.get_bands(kpath=[k], operator="unfold", write=False)
    sparse = h.get_bands(kpath=[k], operator="unfold", num_bands=14,
                         write=False)
    levels = np.unique(np.round(sparse[1], 6))
    assert len(levels) == 3
    for e in levels:
        in_dense = np.isclose(dense[1], e, atol=1e-6)
        in_sparse = np.isclose(sparse[1], e, atol=1e-6)
        assert np.sum(in_dense) == np.sum(in_sparse)  # the level is whole
        assert np.isclose(np.sum(sparse[2][in_sparse]),
                          np.sum(dense[2][in_dense]), atol=1e-6)


def kramers_island():
    """A 0d spinful island with Rashba coupling and seeded onsite disorder:
    a complex Hamiltonian whose levels are Kramers pairs and nothing more"""
    g = geometry.square_lattice().get_supercell(4)
    g.dimensionality = 0
    h = g.get_hamiltonian(has_spin=True)
    h.add_rashba(0.4)
    rng = np.random.RandomState(3)
    ons = {tuple(np.round(r, 6)): rng.uniform(-0.5, 0.5) for r in g.r}
    h.add_onsite(lambda r: ons[tuple(np.round(r, 6))])
    return h


def test_arpack_ldos_over_kramers_pairs_matches_dense():
    """sum_n |v_n|^2 over whole Kramers pairs is the diagonal of the
    projector onto them, whatever basis of each pair is returned"""
    h = kramers_island()
    m = csc_matrix(h.intra)
    E, V = eigh(m.toarray())
    e, delta, nwf = 0.1, 0.05, 8
    keep = np.argsort(np.abs(E - e))[0:nwf]
    assert np.all(np.diff(np.sort(E[keep]))[0::2] < 1e-8)  # whole pairs
    ref = np.sum(np.abs(V[:, keep])**2 * delta / ((e - E[keep])**2 + delta**2),
                 axis=1) / np.pi
    d = ldos.ldos_arpack(m, num_wf=nwf, e=e, delta=delta)
    assert np.allclose(d, ref, atol=1e-8)


def test_sparse_eigenvectors_are_orthonormal():
    h = kramers_island()
    es, vs = h.get_eigenvectors(numw=8)
    assert np.allclose(vs @ np.conjugate(vs.T), np.identity(len(es)),
                       atol=1e-8)
