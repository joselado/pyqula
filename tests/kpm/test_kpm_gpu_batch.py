import numpy as np
import pytest

from pyqula import kpm
from pyqula.kpmtk.kpmnumba import kpm_moments_batch, kpm_moments_A_batch


def _random_hermitian_and_vectors(n, nvec, complex_input, seed):
    rng = np.random.RandomState(seed)
    if complex_input:
        m = rng.random((n, n)) + 1j*rng.random((n, n))
        vs = rng.random((nvec, n)) + 1j*rng.random((nvec, n))
    else:
        m = rng.random((n, n))
        vs = rng.random((nvec, n))
    m = m + np.conjugate(m).T
    m = m/np.max(np.abs(np.linalg.eigvalsh(m)))/1.1  # rescale into (-1,1)
    vs = vs/np.sqrt(np.sum(np.conjugate(vs)*vs, axis=1))[:, None]
    return m, vs


@pytest.mark.parametrize("complex_input", [False, True])
@pytest.mark.parametrize("kpm_prec,tol", [("double", 1e-8), ("single", 1e-4)])
def test_jax_batched_moments_match_numba_cpu_batch(complex_input, kpm_prec, tol):
    """The batched JAX GPU path (kpm_moments_batch_gpu, dispatched via
    jax.lax.map in fixed-size chunks) must agree with the batched numba
    CPU path, for real and complex input, at both precisions -- this is
    the GPU-less-machine correctness check for the previously-unfinished
    batched GPU branch of kpm_moments_batch (it used to loop in Python over
    the single-vector GPU kernel instead)."""
    pytest.importorskip("jax")
    m, vs = _random_hermitian_and_vectors(15, 5, complex_input, seed=10)
    mus_cpu = kpm_moments_batch(vs, m, n=20, kpm_prec=kpm_prec, kpm_cpugpu="CPU")
    mus_gpu = kpm_moments_batch(vs, m, n=20, kpm_prec=kpm_prec, kpm_cpugpu="GPU")
    assert np.max(np.abs(mus_cpu - mus_gpu)) < tol


def test_jax_batched_moments_chunking_matches_unchunked():
    """kpm_moments_batch_gpu dispatches the batch through jax.lax.map in
    gpu_batch_size-sized chunks (default 256) instead of one jax.vmap over
    the whole batch, so that a full-space trace (nvec=nsites, e.g.
    kpm.full_trace on a ~10,000-site system) doesn't try to materialize the
    whole batch worth of recursion state on the GPU at once. Chunking must
    not change the result: check a batch bigger than the default chunk
    size, and an explicit small gpu_batch_size that forces several chunks
    of very different sizes, all against the same CPU reference."""
    pytest.importorskip("jax")
    m, vs = _random_hermitian_and_vectors(12, 600, True, seed=13)
    mus_cpu = kpm_moments_batch(vs, m, n=10, kpm_prec="double", kpm_cpugpu="CPU")
    mus_gpu_default = kpm_moments_batch(vs, m, n=10, kpm_prec="double", kpm_cpugpu="GPU")
    mus_gpu_small_chunk = kpm_moments_batch(vs, m, n=10, kpm_prec="double",
            kpm_cpugpu="GPU", gpu_batch_size=7)
    assert np.max(np.abs(mus_cpu - mus_gpu_default)) < 1e-8
    assert np.max(np.abs(mus_cpu - mus_gpu_small_chunk)) < 1e-8


@pytest.mark.parametrize("complex_input", [False, True])
@pytest.mark.parametrize("kpm_prec,tol", [("double", 1e-8), ("single", 1e-4)])
def test_jax_batched_momentsA_match_numba_cpu_batch(complex_input, kpm_prec, tol):
    """Same check as above, for the operator-weighted batched moments
    (kpm_moments_A_batch / kpm_momentsA_batch_gpu), which previously had no
    GPU path at all (kpm_cpugpu='GPU' raised ValueError)."""
    pytest.importorskip("jax")
    rng = np.random.RandomState(11)
    m, vs = _random_hermitian_and_vectors(15, 5, complex_input, seed=12)
    if complex_input:
        A = rng.random((15, 15)) + 1j*rng.random((15, 15))
    else:
        A = rng.random((15, 15))
    A = A + np.conjugate(A).T
    mus_cpu = kpm_moments_A_batch(vs, m, A, n=20, kpm_prec=kpm_prec, kpm_cpugpu="CPU")
    mus_gpu = kpm_moments_A_batch(vs, m, A, n=20, kpm_prec=kpm_prec, kpm_cpugpu="GPU")
    assert np.max(np.abs(mus_cpu - mus_gpu)) < tol


def test_jax_batched_momentsA_chunking_matches_unchunked():
    """Same chunking check as test_jax_batched_moments_chunking_matches_unchunked,
    for the operator-weighted batched path (kpm_momentsA_batch_gpu)."""
    pytest.importorskip("jax")
    rng = np.random.RandomState(14)
    m, vs = _random_hermitian_and_vectors(12, 600, True, seed=15)
    A = rng.random((12, 12)) + 1j*rng.random((12, 12))
    A = A + np.conjugate(A).T
    mus_cpu = kpm_moments_A_batch(vs, m, A, n=10, kpm_prec="double", kpm_cpugpu="CPU")
    mus_gpu_default = kpm_moments_A_batch(vs, m, A, n=10, kpm_prec="double", kpm_cpugpu="GPU")
    mus_gpu_small_chunk = kpm_moments_A_batch(vs, m, A, n=10, kpm_prec="double",
            kpm_cpugpu="GPU", gpu_batch_size=11)
    assert np.max(np.abs(mus_cpu - mus_gpu_default)) < 1e-8
    assert np.max(np.abs(mus_cpu - mus_gpu_small_chunk)) < 1e-8


def test_kpm_cpugpu_reaches_public_dos_entry_points():
    """kpm_cpugpu must actually be selectable from the user-facing KPM
    entry points (kpm.tdos/kpm.ldos/kpm.full_trace/kpm.full_trace_A), not
    just from the low-level kpmnumba functions -- random_trace/random_trace_A
    /full_trace_A/tdos used to drop any **kwargs on the floor instead of
    forwarding them to get_moments_batch/get_moments_A_batch, so kpm_cpugpu
    could not reach these callers at all even though the GPU kernels
    existed. Random tries are reseeded identically before each backend so
    CPU and GPU see the same starting vectors."""
    pytest.importorskip("jax")
    from pyqula import geometry
    g = geometry.honeycomb_lattice()
    h = g.get_hamiltonian()
    m = h.get_hk_gen()([0., 0., 0.])

    np.random.seed(123)
    _, ys_cpu = kpm.tdos(m, npol=30, ne=40, ntries=6, kpm_cpugpu="CPU")
    np.random.seed(123)
    _, ys_gpu = kpm.tdos(m, npol=30, ne=40, ntries=6, kpm_cpugpu="GPU")
    mask = ~(np.isnan(ys_cpu) | np.isnan(ys_gpu))  # KPM edge-of-window artifact, unrelated to the backend
    assert np.max(np.abs(ys_cpu[mask] - ys_gpu[mask])) < 1e-8

    xs_cpu, ldos_cpu = kpm.ldos(m, i=0, npol=30, ne=40, kpm_cpugpu="CPU")
    xs_gpu, ldos_gpu = kpm.ldos(m, i=0, npol=30, ne=40, kpm_cpugpu="GPU")
    assert np.max(np.abs(ldos_cpu - ldos_gpu)) < 1e-8

    mus_cpu = kpm.full_trace(m, n=15, kpm_cpugpu="CPU")
    mus_gpu = kpm.full_trace(m, n=15, kpm_cpugpu="GPU")
    assert np.max(np.abs(mus_cpu - mus_gpu)) < 1e-8

    A = np.eye(m.shape[0])
    mus_cpu_A = kpm.full_trace_A(m, n=15, A=A, kpm_cpugpu="CPU")
    mus_gpu_A = kpm.full_trace_A(m, n=15, A=A, kpm_cpugpu="GPU")
    assert np.max(np.abs(mus_cpu_A - mus_gpu_A)) < 1e-8


def test_kpm_cpugpu_reaches_random_trace_operator_branch():
    """random_trace's operator-weighted branch (kpm.py:191, the route
    behind kpm.pdos / projected-DOS calculations) calls get_moments_A_batch
    -- i.e. the newly-reachable kpm_momentsA_batch_gpu -- with the operator
    kwarg mixed in, unlike the plain tdos() call in
    test_kpm_cpugpu_reaches_public_dos_entry_points which never touches
    that branch. Check it separately since it's the call site with the
    most surrounding kwargs, and thus the one most likely to trip a
    CPU/GPU kwarg-handling mismatch."""
    pytest.importorskip("jax")
    from pyqula import geometry
    g = geometry.honeycomb_lattice()
    h = g.get_hamiltonian()
    m = h.get_hk_gen()([0., 0., 0.])
    P = np.eye(m.shape[0])  # trivial projector, P^2 = P

    np.random.seed(77)
    _, ys_cpu = kpm.tdos(m, npol=20, ne=30, ntries=4, operator=P, kpm_cpugpu="CPU")
    np.random.seed(77)
    _, ys_gpu = kpm.tdos(m, npol=20, ne=30, ntries=4, operator=P, kpm_cpugpu="GPU")
    mask = ~(np.isnan(ys_cpu) | np.isnan(ys_gpu))
    assert np.max(np.abs(ys_cpu[mask] - ys_gpu[mask])) < 1e-8


def test_kpm_cpugpu_reaches_kdos_bands_operator_branch():
    """h.get_kdos_bands(mode='KPM', operator=...) is a second, independent
    production caller of the operator-weighted A-batch kernel (via
    kdos.kdos_bands -> kpm.pdos -> kpm.tdos's operator= argument -- not to
    be confused with h.get_dos(mode='KPM', operator=...), which routes the
    operator through subspace-confined random vectors instead and never
    reaches get_moments_A_batch at all, see the user guide note on this).
    Use parallel.set_cores(1) and a single explicit k-point so the random
    draws consumed are identical between backends -- with the default
    multi-k path, parallel dispatch order/count affects how the shared
    global random state is consumed, independent of anything the GPU port
    changed, so it isn't a meaningful equivalence check."""
    pytest.importorskip("jax")
    from pyqula import geometry, parallel
    parallel.set_cores(1)
    g = geometry.honeycomb_lattice()
    h = g.get_hamiltonian()
    op = np.eye(h.intra.shape[0])
    k0 = [0.1, 0.2, 0.]

    np.random.seed(9)
    out_cpu = h.get_kdos_bands(mode="KPM", operator=op, kpath=[k0],
            ntries=3, delta=0.1, kpm_cpugpu="CPU")
    np.random.seed(9)
    out_gpu = h.get_kdos_bands(mode="KPM", operator=op, kpath=[k0],
            ntries=3, delta=0.1, kpm_cpugpu="GPU")
    assert np.max(np.abs(out_cpu - out_gpu)) < 1e-8


def test_stray_kwarg_ignored_identically_on_cpu_and_gpu():
    """A kwarg that kpm_moments_batch/kpm_moments_A_batch don't recognize
    (neither kpm_prec nor kpm_cpugpu) must be silently ignored by both
    backends, not just the CPU/numba one -- kpm_moments_batch_gpu/
    kpm_momentsA_batch_gpu now accept **kwargs and drop it, matching the
    numba kernels (which never took any kwargs at all)."""
    pytest.importorskip("jax")
    from pyqula import geometry
    g = geometry.honeycomb_lattice()
    h = g.get_hamiltonian()
    m = h.get_hk_gen()([0., 0., 0.])

    for backend in ("CPU", "GPU"):
        xs, ys = kpm.tdos(m, npol=20, ne=20, ntries=4, kpm_cpugpu=backend, bogus=1)
        assert xs.shape == ys.shape
