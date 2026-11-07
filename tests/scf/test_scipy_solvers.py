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


@pytest.mark.parametrize("solver", ["krylov", "anderson", "broyden1", "linear"])
def test_scipy_solvers_converge_to_maxerror(solver):
    """scf.converged means the same for every solver: the residual of the
    returned mean field is below maxerror. The scipy solvers were marked
    converged at their own tolerance (the scipy default for krylov and
    anderson, 100*maxerror for broyden1 and linear)."""
    from pyqula.scftk.densitydensity import Vinteraction
    h = geometry.bichain().get_hamiltonian(has_spin=False)
    scf = Vinteraction(h, V1=2.0, filling=0.5, mf="CDW", nk=10,
            solver=solver, maxerror=1e-8, load_mf=False, write=False)
    assert scf.converged
    assert scf.residual <= 1e-8


def test_a_gapped_bdg_state_under_anderson_holds_its_charge(capsys):
    """A gapped s-wave state solved with anderson holds the requested
    charge and is not told it has no gap. With the scipy default tolerance
    the loop stopped with a charge error of 1e-5 and was marked converged,
    so the BdG Fermi level printed that the spectrum has no gap."""
    from pyqula.scftk.densitydensity import Vinteraction
    from pyqula.scftk import superscf
    h = geometry.square_lattice().get_hamiltonian(has_spin=True)
    h.setup_nambu_spinor()
    scf = Vinteraction(h, U=-2., filling=0.3, mf="swave", nk=8,
            solver="anderson", maxerror=1e-8, load_mf=False, write=False)
    assert scf.converged
    assert abs(superscf.get_charge_bdg(scf.dm) - 0.3*2) < 1e-6
    assert "no gap" not in capsys.readouterr().out
