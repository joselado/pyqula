import numpy as np

from pyqula import geometry
from pyqula.kanemele import generalized_kane_mele, haldane, km_vector

_sx = np.array([[0., 1.], [1., 0.]])
_sy = np.array([[0., -1j], [1j, 0.]])
_sz = np.array([[1., 0.], [0., -1.]])


def _dense(m):
    if isinstance(m, int): return np.zeros((0, 0), dtype=complex)
    return np.asarray(m.todense()) if hasattr(m, "todense") else np.asarray(m)


def _dense_reference_gkm(r1, r2, rm, fun, tol=1e-5):
    """Independent O(N^2) dense reference for generalized_kane_mele, built
    directly with the Pauli matrices without any of the KD-tree/vectorized
    bond machinery in kanemele.generalized_kane_mele."""
    r1 = np.array(r1); r2 = np.array(r2); rm = np.array(rm)
    n = len(r1)
    m = np.zeros((2 * n, 2 * n), dtype=complex)
    for i in range(n):
        for j in range(n):
            dr = r1[i] - r2[j]
            if not dr.dot(dr) < 4.1: continue
            ur = km_vector(r1[i], r2[j], rm, tol=tol)
            if ur[0] == 0.0 and ur[1] == 0.0 and ur[2] == 0.0: continue
            r3 = (r1[i] + r2[j]) / 2.0
            cv = fun(r3) if callable(fun) else fun
            sm = 1j * (ur[0] * _sx + ur[1] * _sy + ur[2] * _sz) * cv
            m[2 * i:2 * i + 2, 2 * j:2 * j + 2] = sm
    return m


def _dense_reference_haldane(r1, r2, rm, fun, sublattice=None):
    """Independent O(N^2) dense reference for haldane."""
    r1 = np.array(r1); r2 = np.array(r2); rm = np.array(rm)
    n = len(r1)
    if sublattice is None: sublattice = np.zeros(n) + 1.0
    m = np.zeros((n, n), dtype=complex)
    for i in range(n):
        for j in range(n):
            dr = r1[i] - r2[j]
            if not dr.dot(dr) < 4.1: continue
            ur = km_vector(r1[i], r2[j], rm)
            if ur[2] == 0.0: continue
            r3 = (r1[i] + r2[j]) / 2.0
            cv = fun(r3) if callable(fun) else fun
            m[i, j] = 1j * ur[2] * cv * (sublattice[i] + sublattice[j]) / 2.
    return m


def _geometries():
    cases = {}
    g = geometry.honeycomb_lattice().get_supercell(4)
    cases["honeycomb"] = (g, g.r, g.r)
    g = geometry.kagome_lattice().get_supercell(3)
    cases["kagome"] = (g, g.r, g.r)
    g = geometry.triangular_lattice().get_supercell(4)
    cases["triangular"] = (g, g.r, g.r)
    g = geometry.honeycomb_lattice().get_supercell(4)
    cases["honeycomb_shifted"] = (g, g.r, [ir + g.a1 for ir in g.r])
    return cases


def test_generalized_kane_mele_matches_dense_reference():
    """generalized_kane_mele (KD-tree candidate search + vectorized 2x2
    spin-block construction) must match a fully independent dense O(N^2)
    reference, for constant and position-dependent coupling strength."""
    for name, (g, r1, r2) in _geometries().items():
        rm = g.multireplicas(3)
        got = _dense(generalized_kane_mele(r1, r2, rm, fun=0.15))
        ref = _dense_reference_gkm(r1, r2, rm, 0.15)
        assert np.max(np.abs(got - ref)) < 1e-8, name

        got = _dense(generalized_kane_mele(r1, r2, rm,
                fun=lambda p: 0.1 + 0.05 * p[0]))
        ref = _dense_reference_gkm(r1, r2, rm, lambda p: 0.1 + 0.05 * p[0])
        assert np.max(np.abs(got - ref)) < 1e-8, (name, "callable")


def test_haldane_matches_dense_reference():
    """haldane (sparse triplet construction instead of a dense nsites x
    nsites array) must match a fully independent dense O(N^2) reference,
    for constant/position-dependent strength and a sublattice pattern."""
    for name, (g, r1, r2) in _geometries().items():
        rm = g.multireplicas(3)
        sub = getattr(g, "sublattice", None)
        got = _dense(haldane(r1, r2, rm, fun=0.2, sublattice=sub))
        ref = _dense_reference_haldane(r1, r2, rm, 0.2, sublattice=sub)
        assert np.max(np.abs(got - ref)) < 1e-8, name

        got = _dense(haldane(r1, r2, rm, fun=lambda p: 0.1 + 0.02 * p[1],
                sublattice=sub))
        ref = _dense_reference_haldane(r1, r2, rm, lambda p: 0.1 + 0.02 * p[1],
                sublattice=sub)
        assert np.max(np.abs(got - ref)) < 1e-8, (name, "callable")


def test_add_soc_and_add_haldane_produce_finite_sparse_hamiltonians():
    """End-to-end add_soc/add_haldane on 0D, 2D, and multicell 2D
    Hamiltonians must stay finite and sparse."""
    g = geometry.honeycomb_lattice()
    for dim0 in [True, False]:
        gg = g.get_supercell(4) if dim0 else g
        if dim0: gg.dimensionality = 0
        for multicell in [False, True]:
            h = gg.get_hamiltonian(is_sparse=True, is_multicell=multicell)
            h.add_soc(0.1)
            m = h.intra
            assert h.is_sparse
            md = np.asarray(m.todense()) if hasattr(m, "todense") else np.asarray(m)
            assert np.all(np.isfinite(md)), (dim0, multicell, "soc")

            h2 = gg.get_hamiltonian(is_sparse=True, is_multicell=multicell)
            h2.add_haldane(0.1)
            m2 = h2.intra
            md2 = np.asarray(m2.todense()) if hasattr(m2, "todense") else np.asarray(m2)
            assert np.all(np.isfinite(md2)), (dim0, multicell, "haldane")
