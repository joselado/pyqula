import numpy as np

from pyqula import geometry, parallel
from testutils import assert_all_consistent, random_hermitian_hamiltonian

NK = 3


def _compute(h, cores, use_ds):
    parallel.set_cores(cores)
    ds = [[i, 0, 0] for i in range(3)] if use_ds else None
    o = h.get_density_matrix(nk=NK, ds=ds, dm_mode="accumulate")
    if use_ds:
        o = np.array([o[key] for key in o])
    return o


def test_full_dm_accumulate_is_independent_of_core_count():
    """full_dm_accumulate is dispatched through parallel.pcall_shared; its
    result must not depend on how many worker processes computed it."""
    h = random_hermitian_hamiltonian(geometry.honeycomb_lattice, supercell=3)
    try:
        for use_ds in (True, False):
            outs = [_compute(h, cores, use_ds) for cores in (1, 2, 4)]
            assert_all_consistent(outs, 1e-8,
                    f"full_dm_accumulate across core counts (use_ds={use_ds})")
    finally:
        parallel.set_cores(1)
