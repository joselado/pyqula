"""The GPU path of the full density matrix.

Under pyqula.gpu.set_gpu(True), densitymatrix.full_dm_accumulate hands the
whole calculation to dmtk/fulldmjax.py -- Bloch sum, diagonalization,
occupations and the k-sum all on the device -- instead of bringing every
k-point's eigenvectors back to the host to contract them there. It must
return exactly what the numba route returns.
"""
import numpy as np
import pytest

from pyqula import geometry
from pyqula.densitymatrix import full_dm_accumulate
from pyqula.htk import eigenvectors as ev
from testutils import gpu_backend

DIRECTIONS = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]


def _system(nsuper=4):
    """Wide enough to clear gpu_min_dimension, so the switch dispatches"""
    h = geometry.honeycomb_lattice().supercell(nsuper).get_hamiltonian()
    h.turn_spinful()
    h.add_exchange([0., 0., 0.2]) # a non-trivial spin structure to reproduce
    assert h.intra.shape[0] >= ev.gpu_min_dimension
    return h


def test_the_undirected_density_matrix_matches_the_numba_one():
    pytest.importorskip("jax")
    h = _system()
    cpu = full_dm_accumulate(h, nk=6, fermi=0.1, delta=1e-3)
    with gpu_backend():
        got = full_dm_accumulate(h, nk=6, fermi=0.1, delta=1e-3)
    assert got.shape == cpu.shape and got.dtype == cpu.dtype
    assert np.max(np.abs(got - cpu)) < 1e-12*np.max(np.abs(cpu))


def test_the_per_direction_density_matrices_match_the_numba_ones():
    """The directed density matrices carry the Bloch phase of the hopping
    direction, which uses all three components of k while the Bloch sum
    itself uses only the periodic ones -- a split the device path has to
    make the same way"""
    pytest.importorskip("jax")
    h = _system()
    cpu = full_dm_accumulate(h, nk=6, fermi=0.1, delta=1e-3, ds=DIRECTIONS)
    with gpu_backend():
        got = full_dm_accumulate(h, nk=6, fermi=0.1, delta=1e-3, ds=DIRECTIONS)
    assert set(got) == set(cpu) # same dictionary keys
    for d in cpu:
        assert np.max(np.abs(got[d] - cpu[d])) < 1e-12*np.max(np.abs(cpu[d]))


def test_the_density_matrix_is_hermitian_and_has_the_right_trace():
    """An invariant rather than a comparison: the undirected density matrix
    is Hermitian and its trace is the number of occupied states per cell"""
    pytest.importorskip("jax")
    h = _system()
    with gpu_backend():
        dm = full_dm_accumulate(h, nk=6, fermi=0., delta=1e-4)
    assert np.max(np.abs(dm - np.conjugate(dm.T))) < 1e-10
    n = h.intra.shape[0]
    assert abs(np.trace(dm).real - n/2.) < 1e-6 # half filling
    assert abs(np.trace(dm).imag) < 1e-12


def test_a_mesh_bigger_than_one_dispatch_still_agrees():
    """The k-mesh is chunked against the device's memory, and a last chunk
    that is not full must not change the sum"""
    pytest.importorskip("jax")
    from pyqula.htk import eigenvectorsjax as evjax
    h = _system()
    cpu = full_dm_accumulate(h, nk=5, fermi=0.1, delta=1e-3, ds=DIRECTIONS)
    n = h.intra.shape[0]
    chunk = evjax.CHUNK_ELEMENTS
    try:
        evjax.CHUNK_ELEMENTS = 4*n*n # 4 kpoints per dispatch, 25 kpoints
        with gpu_backend():
            got = full_dm_accumulate(h, nk=5, fermi=0.1, delta=1e-3,
                                     ds=DIRECTIONS)
    finally:
        evjax.CHUNK_ELEMENTS = chunk
    for d in cpu:
        assert np.max(np.abs(got[d] - cpu[d])) < 1e-12*np.max(np.abs(cpu[d]))


def test_a_small_hamiltonian_stays_on_the_cpu():
    """Below gpu_min_dimension the device is not worth it, so the numba
    route runs even with the switch on"""
    pytest.importorskip("jax")
    from pyqula.dmtk import fulldmjax
    calls = []
    real = fulldmjax.full_dm_gpu

    def spy(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    fulldmjax.full_dm_gpu = spy
    try:
        small = geometry.honeycomb_lattice().get_hamiltonian() # 2 orbitals
        with gpu_backend():
            full_dm_accumulate(small, nk=4, delta=1e-3)
            assert calls == []
            full_dm_accumulate(_system(), nk=4, delta=1e-3)
            assert len(calls) == 1
    finally:
        fulldmjax.full_dm_gpu = real


def test_the_occupations_saturate_instead_of_overflowing():
    """delta is ~1e-6 by default, so a level a bandwidth away from the
    Fermi energy gives es/delta ~ 1e6: the Fermi function has to saturate
    rather than overflow the exponential. Checked through the trace, which
    would come out NaN if it did not"""
    pytest.importorskip("jax")
    h = _system()
    with gpu_backend():
        dm = full_dm_accumulate(h, nk=4, fermi=0., delta=1e-7)
    assert np.all(np.isfinite(dm))
    assert abs(np.trace(dm).real - h.intra.shape[0]/2.) < 1e-6
