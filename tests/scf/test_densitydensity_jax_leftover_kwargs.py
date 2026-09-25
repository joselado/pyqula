import numpy as np
import pytest

pytest.importorskip("jax")

from pyqula import geometry
from pyqula.scftk.densitydensity import Vinteraction

# The use_jax=True route of Vinteraction ends in generic_densitydensity_jax,
# which took **kwargs and never read them, so a misspelled keyword, or one
# only the numpy engine reads, ran with the default in silence.


def _spinless_chain():
    return geometry.chain().get_hamiltonian(has_spin=False)


@pytest.mark.parametrize("extra", [dict(kick_step=30), dict(load_mf=False),
        dict(tolerance=1e-8)])
def test_a_keyword_nothing_reads_is_refused(extra):
    with pytest.raises(TypeError, match="unexpected keyword"):
        Vinteraction(_spinless_chain(), V1=1.0, mu=0.0, nk=4, T=1e-2,
                use_jax=True, solver="newton", verbose=0, **extra)


def test_qtci_has_no_jax_counterpart():
    with pytest.raises(NotImplementedError, match="exact diagonalization"):
        Vinteraction(_spinless_chain(), V1=1.0, mu=0.0, nk=4, T=1e-2,
                use_jax=True, solver="newton", verbose=0, integration="qtci")


def test_the_spinless_public_route_still_runs():
    """get_mean_field_hamiltonian passes integration="ed" explicitly on a
    spinless Hamiltonian, which must still be accepted"""
    h = geometry.chain().get_supercell(2).get_hamiltonian(has_spin=False)
    mf = {(0, 0, 0): np.diag([0.5, -0.5]).astype(complex)}
    hh = h.get_mean_field_hamiltonian(V1=3., nk=6, T=1e-3, mu=0.5, mf=mf,
            maxerror=1e-8, use_jax=True, solver="newton")
    assert hh is not None
