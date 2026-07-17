import numpy as np
import numba

from pyqula import geometry
from testutils import assert_all_consistent, random_hermitian_hamiltonian

NK = 3


def _compute(h, use_ds, batch_size=16):
    ds = [[i, 0, 0] for i in range(3)] if use_ds else None
    o = h.get_density_matrix(nk=NK, ds=ds, dm_mode="accumulate", batch_size=batch_size)
    if use_ds:
        o = np.array([o[key] for key in o])
    return o


def test_full_dm_accumulate_is_independent_of_thread_count():
    """full_dm_accumulate diagonalizes k-points in parallel across numba
    threads (htk.eigenvectors.parallel_diagonalization); its result must
    not depend on how many threads did the work."""
    h = random_hermitian_hamiltonian(geometry.honeycomb_lattice, supercell=3)
    default_threads = numba.get_num_threads()
    try:
        for use_ds in (True, False):
            outs = []
            for nthreads in (1, 2, 4):
                numba.set_num_threads(nthreads)
                outs.append(_compute(h, use_ds))
            assert_all_consistent(outs, 1e-8,
                    f"full_dm_accumulate across thread counts (use_ds={use_ds})")
    finally:
        numba.set_num_threads(default_threads)


def test_full_dm_accumulate_is_independent_of_batch_size():
    """The k-mesh is diagonalized in batches purely to bound memory use;
    the result must not depend on where the batch boundaries fall."""
    h = random_hermitian_hamiltonian(geometry.honeycomb_lattice, supercell=3)
    for use_ds in (True, False):
        outs = [_compute(h, use_ds, batch_size=bs) for bs in (1, 3, 5, 100)]
        assert_all_consistent(outs, 1e-8,
                f"full_dm_accumulate across batch sizes (use_ds={use_ds})")
