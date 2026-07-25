import numpy as np

from pyqula import geometry
from pyqula.rashba import rashba_matrix

_sx = np.array([[0., 1.], [1., 0.]])
_sy = np.array([[0., -1j], [1j, 0.]])
_sz = np.array([[1., 0.], [0., -1.]])


def _dense_reference_rashba(r1, r2, c, d=(0., 0., 1.)):
    """Independent O(N^2) dense reference for the Rashba matrix, built
    directly from the Pauli matrices without going through neighbor.py or
    any of the vectorized bond machinery in rashba.rashba_matrix."""
    r1 = np.array(r1); r2 = np.array(r2)
    n1, n2 = len(r1), len(r2)
    m = np.zeros((2 * n1, 2 * n2), dtype=complex)
    for i in range(n1):
        for j in range(n2):
            rij = r2[j] - r1[i]
            if not (0.99 < rij.dot(rij) < 1.01):
                continue
            dx, dy, dz = rij
            rxs = [dy * _sz - dz * _sy, dz * _sx - dx * _sz, dx * _sy - dy * _sx]
            ms = 1j * (d[0] * rxs[0] + d[1] * rxs[1] + d[2] * rxs[2])
            cv = c((r1[i] + r2[j]) / 2.) if callable(c) else c
            m[2 * i:2 * i + 2, 2 * j:2 * j + 2] = ms * cv
    return m


def _dense(m):
    return np.asarray(m.todense()) if hasattr(m, "todense") else np.asarray(m)


def _geometries():
    cases = {}
    g = geometry.honeycomb_lattice().get_supercell(4)
    cases["honeycomb"] = (g.r, g.r)
    g = geometry.kagome_lattice().get_supercell(3)
    cases["kagome"] = (g.r, g.r)
    g = geometry.triangular_lattice().get_supercell(4)
    cases["triangular"] = (g.r, g.r)
    g = geometry.honeycomb_lattice().get_supercell(4)
    cases["honeycomb_shifted"] = (g.r, [ir + g.a1 for ir in g.r])
    return cases


def test_rashba_matrix_matches_dense_reference():
    """rashba_matrix (KD-tree neighbor search + vectorized 2x2 spin-block
    construction) must match a fully independent, dense O(N^2) reference,
    for constant and position-dependent Rashba strength, and for a
    non-z spin quantization axis."""
    for name, (r1, r2) in _geometries().items():
        for is_sparse in [False, True]:
            got = _dense(rashba_matrix(r1, r2=r2, c=0.3, is_sparse=is_sparse))
            ref = _dense_reference_rashba(r1, r2, 0.3)
            assert np.max(np.abs(got - ref)) < 1e-10, (name, is_sparse)

        got = _dense(rashba_matrix(r1, r2=r2, c=lambda p: 0.2 + 0.1 * p[0],
                is_sparse=True))
        ref = _dense_reference_rashba(r1, r2, lambda p: 0.2 + 0.1 * p[0])
        assert np.max(np.abs(got - ref)) < 1e-10, (name, "callable_c")

        d = [0.3, 0.5, 0.8]
        got = _dense(rashba_matrix(r1, r2=r2, c=0.3, d=d, is_sparse=True))
        ref = _dense_reference_rashba(r1, r2, 0.3, d=d)
        assert np.max(np.abs(got - ref)) < 1e-10, (name, "custom_d")


def test_rashba_matrix_edge_localized_profile_matches_dense_reference():
    """A common physical use case: Rashba coupling confined to an edge or
    interface, i.e. c(r) is exactly zero over most of the flake. This
    specifically exercises the `data!=0.0` zero-filtering in the
    vectorized bond construction, not just a smooth nonzero profile."""
    g = geometry.honeycomb_lattice().get_supercell(6)
    r1 = g.r
    profile = lambda p: 0.4 if p[0] > 3.0 else 0.0
    got = _dense(rashba_matrix(r1, r2=r1, c=profile, is_sparse=True))
    ref = _dense_reference_rashba(r1, r1, profile)
    assert np.max(np.abs(got - ref)) < 1e-10
    assert 0 < np.count_nonzero(got) < got.size  # neither all-zero nor all-nonzero


def test_add_rashba_produces_finite_sparse_hamiltonian():
    """End-to-end add_rashba, including on top of SOC and exchange terms,
    must stay finite and sparse."""
    g = geometry.honeycomb_lattice().get_supercell(4)
    g.dimensionality = 0
    for setup in ["plain", "soc", "exchange", "soc_exchange"]:
        h = g.get_hamiltonian(is_sparse=True)
        if setup in ("soc", "soc_exchange"): h.add_soc(0.1)
        if setup in ("exchange", "soc_exchange"): h.add_exchange([0.1, 0., 0.15])
        h.add_rashba(0.2)
        m = h.intra
        assert h.is_sparse
        md = np.asarray(m.todense()) if hasattr(m, "todense") else np.asarray(m)
        assert np.all(np.isfinite(md)), setup
