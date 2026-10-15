import pytest

from pyqula import geometry
from pyqula import meanfield
from pyqula.meanfield import MeanFieldConvergenceError


def _h():
    return geometry.honeycomb_lattice().get_hamiltonian()


def test_public_engine_raises_when_not_converged():
    """meanfield.VJinteraction, the entry point of the guide snippets, does
    not hand back an unconverged state silently: it raises like
    h.get_mean_field_hamiltonian, keeping the SCF object as err.scf"""
    with pytest.raises(MeanFieldConvergenceError) as err:
        meanfield.VJinteraction(_h(), U=3.0, filling=0.5, mf="antiferro",
                                nk=6, maxite=3, maxerror=1e-12)
    assert err.value.scf.converged is False


def test_public_engine_returns_the_scf_on_request():
    scf = meanfield.Vinteraction(_h(), U=3.0, filling=0.5, mf="antiferro",
                                 nk=6, maxite=3, maxerror=1e-12,
                                 check_convergence=False)
    assert scf.converged is False


def test_converged_loop_is_returned():
    scf = meanfield.Vinteraction(_h(), U=3.0, filling=0.5, mf="antiferro",
                                 nk=6, maxerror=1e-6)
    assert scf.converged
