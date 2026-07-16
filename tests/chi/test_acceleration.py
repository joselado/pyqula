import numpy as np

from pyqula import geometry
from pyqula.chitk import rpa
from testutils import assert_all_consistent, random_hermitian_hamiltonian, temporary_attr

ENERGIES = np.linspace(0., 5.0, 400)


def _compute(h, mode):
    with temporary_attr(rpa, "mode_rpa", mode):
        _, _, chis = h.get_qdos_iets(nk=8, nq=16, energies=ENERGIES)
    return chis


def test_rpa_susceptibility_modes_are_consistent():
    """The sequential and vectorized RPA susceptibility implementations
    must produce the same result."""
    h = random_hermitian_hamiltonian(geometry.bichain, supercell=2)
    outs = [_compute(h, mode) for mode in ("sequential", "vectorized")]
    assert_all_consistent(outs, 1e-4, "RPA susceptibility modes")
