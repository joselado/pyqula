import numpy as np
import pytest

from pyqula import geometry


@pytest.mark.parametrize("solver", ["krylov", "anderson", "broyden1", "linear"])
def test_scipy_solvers_return_the_plain_solution(solver):
    """The scipy root finders of the density-density mean field never set
    scf.converged, so h.get_mean_field_hamiltonian(solver=...) failed with
    an AttributeError. They must return the charge-density wave that the
    plain linear-mixing loop converges to."""
    h = geometry.bichain().get_hamiltonian(has_spin=False)
    kw = dict(V1=2.0, filling=0.5, mf="CDW", nk=10)
    plain = h.get_mean_field_hamiltonian(**kw).get_vev()
    other = h.get_mean_field_hamiltonian(solver=solver, **kw).get_vev()
    assert abs(plain[0] - plain[1]) > 0.1 # a real charge-density wave
    assert np.max(np.abs(np.sort(plain) - np.sort(other))) < 1e-3
