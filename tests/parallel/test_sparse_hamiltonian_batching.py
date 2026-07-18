import numpy as np

from pyqula import geometry
from pyqula.htk.eigenvectors import get_eigenvectors, hk_matrix_batch
from pyqula.spectrum import eigenvalues_kmesh, total_energy
from pyqula.filling import eigenvalues as filling_eigenvalues
from pyqula.fermisurface import fermi_surface_generator
from pyqula.ldos import ldosmap
from pyqula.densitymatrix import full_dm_accumulate


def _dense_and_sparse():
    """A small Hamiltonian and a sparse copy of it, both built the same
    way -- used to check the numba-prange batching in htk/eigenvectors.py,
    bandstructure.py, fermisurfacetk/singlefs.py, ldos.py, spectrum.py,
    filling.py, fermisurface.py, and densitymatrix.py all go through
    hk_matrix_batch (which densifies each k-point matrix via
    algebra.todense before stacking), so a sparse Hamiltonian never
    reaches numba's dense eigh as a scipy sparse matrix -- previously a
    TypeError: must be real number, not csc_matrix."""
    g = geometry.honeycomb_lattice()
    h = g.get_hamiltonian()
    h.add_zeeman([0., 0., 0.3])
    hs = h.copy()
    hs.turn_sparse()
    return h, hs


def test_hk_matrix_batch_densifies_sparse_output():
    """Direct unit test of the shared helper: it must return a dense
    complex128 array regardless of whether f(k) returns a scipy sparse
    matrix or a dense one, and the two must agree numerically."""
    h, hs = _dense_and_sparse()
    f = h.get_hk_gen()
    fs = hs.get_hk_gen()
    ks = [[0.1, 0.2, 0.], [0.3, -0.1, 0.]]
    dense_batch = hk_matrix_batch(f, ks)
    sparse_batch = hk_matrix_batch(fs, ks)
    assert dense_batch.dtype == np.complex128
    assert sparse_batch.dtype == np.complex128
    assert type(sparse_batch) is np.ndarray  # genuinely dense, not scipy sparse
    assert np.allclose(dense_batch, sparse_batch)


def test_get_eigenvectors_dense_branch_does_not_crash_on_sparse():
    h, hs = _dense_and_sparse()
    es_dense, _ = get_eigenvectors(h, nk=4)
    es_sparse, _ = get_eigenvectors(hs, nk=4)
    assert np.allclose(np.sort(es_dense), np.sort(es_sparse))


def test_get_bands_default_path_does_not_crash_on_sparse():
    h, hs = _dense_and_sparse()
    _, e_dense = h.get_bands(nk=4, write=False)
    _, e_sparse = hs.get_bands(nk=4, write=False)
    assert np.allclose(np.sort(e_dense), np.sort(e_sparse))


def test_get_fermi_surface_default_mode_does_not_crash_on_sparse():
    h, hs = _dense_and_sparse()
    _, _, kdos_dense = h.get_fermi_surface(nk=4, e=0.1, delta=0.1, write=False)
    _, _, kdos_sparse = hs.get_fermi_surface(nk=4, e=0.1, delta=0.1, write=False)
    assert np.allclose(kdos_dense, kdos_sparse)


def test_eigenvalues_kmesh_does_not_crash_on_sparse():
    h, hs = _dense_and_sparse()
    es_dense = np.sort(eigenvalues_kmesh(h, nk=4).reshape(-1))
    es_sparse = np.sort(eigenvalues_kmesh(hs, nk=4).reshape(-1))
    assert np.allclose(es_dense, es_sparse)


def test_ldosmap_does_not_crash_on_sparse(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    h, hs = _dense_and_sparse()
    energies = np.linspace(-1, 1, 5)
    # pin the internally-drawn random kpoints so dense and sparse see the
    # same ones (see tests/ldos/test_ldosmap.py for why plain seeding
    # isn't enough)
    rng = np.random.RandomState(0)
    ks = [rng.random(3) for _ in range(4)]
    it = iter(ks)
    monkeypatch.setattr(np.random, "random", lambda *a, **kw: next(it))
    _, ds_dense = ldosmap(h, energies=energies, delta=0.1, nk=4)
    it = iter(ks)
    monkeypatch.setattr(np.random, "random", lambda *a, **kw: next(it))
    _, ds_sparse = ldosmap(hs, energies=energies, delta=0.1, nk=4)
    assert np.allclose(ds_dense, ds_sparse)


def test_filling_eigenvalues_still_works_on_sparse_input():
    """filling.eigenvalues already called h.get_dense() before batching,
    so it never crashed -- pin it down so a future change can't quietly
    drop that guard."""
    h, hs = _dense_and_sparse()
    es_dense = np.sort(filling_eigenvalues(h, nk=4))
    es_sparse = np.sort(filling_eigenvalues(hs, nk=4))
    assert np.allclose(es_dense, es_sparse)


def test_total_energy_still_works_on_sparse_input():
    """Same guard, for spectrum.total_energy (calls h.get_dense() when
    nbands is None, before the batched mode='mesh' path runs)."""
    h, hs = _dense_and_sparse()
    e_dense = total_energy(h, nk=4, mode="mesh")
    e_sparse = total_energy(hs, nk=4, mode="mesh")
    assert abs(e_dense - e_sparse) < 1e-8


def test_fermi_surface_generator_still_works_on_sparse_input():
    """fermi_surface_generator derives mode from h.is_sparse itself
    (mode='sparse' vs 'full'), so a sparse Hamiltonian is routed away
    from the batched path entirely -- not exercising hk_matrix_batch,
    but pinned here so the routing isn't broken by future changes."""
    h, hs = _dense_and_sparse()
    _, _, kdos_dense = fermi_surface_generator(h, energies=[0.0, 0.2], nk=5, delta=0.1)
    _, _, kdos_sparse = fermi_surface_generator(hs, energies=[0.0, 0.2], nk=5, delta=0.1)
    assert np.allclose(kdos_dense, kdos_sparse)


def test_full_dm_accumulate_does_not_crash_on_sparse():
    h, hs = _dense_and_sparse()
    dm_dense = full_dm_accumulate(h, nk=4)
    dm_sparse = full_dm_accumulate(hs, nk=4)
    assert np.allclose(dm_dense, dm_sparse)
