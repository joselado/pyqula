import numpy as np
import pytest

from pyqula import geometry
from pyqula.meanfield import VJinteraction


def _converged_v1_chain():
    """A chain converged by VJinteraction with ONLY a nearest-neighbor
    density-density interaction (V1) -- see
    test_rpa_nononsite_ferro_chain.py for why this converges to a genuine
    ferromagnetic moment at low filling. H.V here is a real, multi-key
    (non-onsite) hopping dict, exercising the guard
    chitk.spinchi._require_onsite_only_V is meant to enforce."""
    g = geometry.chain()
    h = g.get_hamiltonian()
    v = np.array([0., 0., 1.])
    h.add_exchange(1e-2*v)
    mf = h.copy()
    mf.add_exchange(0.5*v)
    scf = VJinteraction(h, V1=1.1, filling=0.1, mf=mf, nk=200, mix=0.2,
                         maxerror=1e-8, maxite=1000)
    assert scf.converged
    assert len(scf.hamiltonian.V) > 1  # genuinely non-onsite
    return scf.hamiltonian


@pytest.mark.slow
def test_spinchi_full_raises_for_nononsite_interaction():
    """get_spinchi_full (RPA=True, the default) must raise ValueError for
    a H.V with neighbor-shell (non-onsite) support, e.g. a V1-converged
    Hamiltonian -- non-onsite spin-channel RPA is not yet properly
    verified against an independent reference (see
    chitk.spinchi._require_onsite_only_V's docstring), so it is rejected
    here rather than silently returning an unverified number. This
    replaces an earlier version of this test that asserted rotational
    symmetry of that (now-blocked) non-onsite path -- that numerical
    property still held (checked to 1e-8 before this guard was added), but
    the guard is deliberately more conservative than that ad hoc check."""
    h = _converged_v1_chain()
    with pytest.raises(ValueError):
        h.get_spinchi_full(energies=np.linspace(-0.4, 0.4, 11), delta=2e-2, nk=100)
