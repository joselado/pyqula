"""The fused Bloch-build-and-diagonalize path.

htk/eigenvectors.py's peigh_bloch/peigvalsh_bloch take a Bloch generator
and a k-mesh instead of a finished stack of matrices. On the CPU they are
hk_matrix_batch followed by the numba kernels, unchanged. Under
pyqula.gpu.set_gpu(True) they hand the hopping matrices to the device and
do the Bloch sum there, so the host never builds the (nk,n,n) stack and
never transfers it -- the numbers must come out the same either way.
"""
import numpy as np
import pytest

from pyqula import geometry
from pyqula.htk import eigenvectors as ev
from pyqula.klist import kmesh
from testutils import gpu_backend


def _system(nsuper=6, spinful=True):
    """At the default size, a Hamiltonian wide enough to clear
    gpu_min_dimension, so that the switch actually dispatches"""
    g = geometry.honeycomb_lattice().supercell(nsuper)
    h = g.get_hamiltonian()
    if spinful: h.turn_spinful()
    return h


def _reference(h, ks):
    """What the call sites used to write out by hand"""
    from pyqula.htk.eigenvectors import hk_matrix_batch
    return hk_matrix_batch(h.get_hk_gen(), ks)


def test_the_generator_carries_its_hoppings():
    """The fused route needs the Bloch ingredients, which htk/bloch.py
    attaches to the dense generator it builds"""
    h = _system(nsuper=2)
    ms, ds = h.get_hk_gen().bloch_data
    assert ms.shape[0] == ds.shape[0] # one lattice vector per matrix
    assert ms.shape[1] == ms.shape[2] == h.intra.shape[0]
    assert ds.shape[1] == h.dimensionality # cropped to the periodic directions


def test_the_cpu_route_is_the_old_one_exactly():
    """With the switch off nothing about the arithmetic changed, so this
    is bit-for-bit, not up to a tolerance"""
    h = _system()
    ks = kmesh(h.dimensionality, nk=6)
    es = ev.peigvalsh(_reference(h, ks))
    assert np.array_equal(ev.peigvalsh_bloch(h.get_hk_gen(), ks), es)


@pytest.mark.parametrize("nsuper", [4, 6])
def test_the_device_eigenvalues_match_the_host_built_ones(nsuper):
    pytest.importorskip("jax")
    h = _system(nsuper=nsuper)
    assert h.intra.shape[0] >= ev.gpu_min_dimension # so it really dispatches
    ks = kmesh(h.dimensionality, nk=6)
    cpu = ev.peigvalsh(_reference(h, ks))
    with gpu_backend():
        got = ev.peigvalsh_bloch(h.get_hk_gen(), ks)
    assert got.shape == cpu.shape and got.dtype == np.float64
    assert np.max(np.abs(got - cpu)) < 1e-10


def test_the_device_eigenvectors_diagonalize_the_host_built_hamiltonian():
    """Eigenvectors are defined only up to a phase, so check that they
    diagonalize the SAME H(k) the host would have built -- which also
    checks that the device's Bloch sum agrees with the numba one"""
    pytest.importorskip("jax")
    h = _system()
    ks = kmesh(h.dimensionality, nk=4)
    hks = _reference(h, ks)
    with gpu_backend():
        es, ws = ev.peigh_bloch(h.get_hk_gen(), ks)
    assert es.dtype == np.float64 and ws.dtype == np.complex128
    residual = np.einsum('kij,kjn->kin', hks, ws) - ws*es[:, None, :]
    assert np.max(np.abs(residual)) < 1e-10


@pytest.mark.parametrize("dimensionality", [1, 2, 3])
def test_every_dimensionality_agrees(dimensionality):
    """ds is cropped to the periodic directions and the k-points are not,
    so the device build has to crop them the same way the numba kernel
    does -- a 3d system is where getting that wrong shows up"""
    pytest.importorskip("jax")
    gs = {1: geometry.chain().supercell(40),
          2: geometry.honeycomb_lattice().supercell(6),
          3: geometry.cubic_lattice().supercell(3)}
    h = gs[dimensionality].get_hamiltonian()
    h.turn_spinful()
    ks = kmesh(h.dimensionality, nk=4)
    cpu = ev.peigvalsh(_reference(h, ks))
    with gpu_backend():
        got = ev.peigvalsh_bloch(h.get_hk_gen(), ks)
    assert np.max(np.abs(got - cpu)) < 1e-10


def test_single_precision_loses_about_six_digits():
    pytest.importorskip("jax")
    h = _system()
    ks = kmesh(h.dimensionality, nk=4)
    cpu = ev.peigvalsh(_reference(h, ks))
    with gpu_backend():
        got = ev.peigvalsh_bloch(h.get_hk_gen(), ks, eigh_prec="single")
    err = np.max(np.abs(got - cpu))
    assert 0. < err < 1e-4*np.max(np.abs(cpu))


def test_a_mesh_bigger_than_one_dispatch_still_agrees():
    """The k-mesh is chunked by the entries of the matrices the Bloch sum
    produces, for the same cuSOLVER workspace reason the plain stack is.
    A last chunk that is not full must not change the answer"""
    pytest.importorskip("jax")
    from pyqula.htk import eigenvectorsjax as evjax
    h = _system()
    ks = kmesh(h.dimensionality, nk=5) # 25 kpoints, not a multiple of 8
    cpu = ev.peigvalsh(_reference(h, ks))
    n = h.intra.shape[0]
    chunk = evjax.CHUNK_ELEMENTS
    try:
        evjax.CHUNK_ELEMENTS = 8*n*n # 8 kpoints per dispatch
        with gpu_backend():
            got = ev.peigvalsh_bloch(h.get_hk_gen(), ks)
    finally:
        evjax.CHUNK_ELEMENTS = chunk
    assert got.shape == cpu.shape
    assert np.max(np.abs(got - cpu)) < 1e-10


def test_a_generator_without_hoppings_falls_back():
    """A sparse Hamiltonian, a zero-dimensional one, or any hand-written
    closure has no bloch_data, and must simply take the ordinary route
    rather than raising"""
    pytest.importorskip("jax")
    h = _system()
    ks = kmesh(h.dimensionality, nk=4)
    gen = h.get_hk_gen()
    plain = lambda k: gen(k) # a closure carrying no bloch_data
    assert not hasattr(plain, "bloch_data")
    with gpu_backend():
        got = ev.peigvalsh_bloch(plain, ks)
    assert np.max(np.abs(got - ev.peigvalsh(_reference(h, ks)))) < 1e-10


def test_small_matrices_stay_on_the_cpu_even_with_the_switch_on():
    """The same size threshold the plain stack uses -- below it the device
    loses to numba, so the fused route must not dispatch either"""
    pytest.importorskip("jax")
    from pyqula.htk import eigenvectorsjax as evjax
    calls = []
    real = evjax.peigvalsh_bloch_gpu

    def spy(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    evjax.peigvalsh_bloch_gpu = spy
    try:
        small = geometry.honeycomb_lattice().get_hamiltonian() # 2 orbitals
        assert small.intra.shape[0] < ev.gpu_min_dimension
        with gpu_backend():
            ev.peigvalsh_bloch(small.get_hk_gen(), kmesh(2, nk=4))
            assert calls == []
            ev.peigvalsh_bloch(_system().get_hk_gen(), kmesh(2, nk=4))
            assert len(calls) == 1
    finally:
        evjax.peigvalsh_bloch_gpu = real
