import numpy as np

from pyqula import geometry
from pyqula.dmtk import fulldm
from testutils import assert_all_consistent, random_hermitian_hamiltonian, temporary_attr

NK = 3


def _compute(h, mode1, mode2, use_ds):
    with temporary_attr(fulldm, "mode", mode1):
        ds = [[i, 0, 0] for i in range(10)] if use_ds else None
        o = h.get_density_matrix(nk=NK, ds=ds, dm_mode=mode2)
    if use_ds:
        o = np.array([o[key] for key in o])
    return o


def test_density_matrix_modes_are_consistent():
    """explicit/vectorized and accumulate/simultaneous density-matrix
    implementations must all agree on the result, on the same Hamiltonian,
    for both the per-hopping (ds) and full-matrix output modes."""
    h = random_hermitian_hamiltonian(geometry.honeycomb_lattice, supercell=4)
    modes = [(m1, m2)
             for m1 in ("explicit", "vectorized")
             for m2 in ("accumulate", "simultaneous")]
    for use_ds in (True, False):
        outs = [_compute(h, m1, m2, use_ds) for m1, m2 in modes]
        assert_all_consistent(outs, 1e-4, f"Density matrix modes (use_ds={use_ds})")
