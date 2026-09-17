import numpy as np
import pytest

from pyqula import geometry
from pyqula.meanfield import VJinteraction


def _converged_v1_chain():
    """A chain converged by VJinteraction with ONLY a nearest-neighbor
    density-density interaction (V1) -- see
    test_rpa_nononsite_ferro_chain.py for why this converges to a genuine
    ferromagnetic moment at low filling. H.V here is a real, multi-key
    (non-onsite) hopping dict, whose spin response is summed in the pair
    basis."""
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
def test_spinchi_full_of_a_v1_ferromagnet_does_not_depend_on_its_axis():
    """get_spinchi_full (RPA=True, the default) sums a neighbor-shell
    density-density interaction in the pair basis. The axis a state is
    magnetized along is not an observable, so the trace of the response
    must not change under a global spin rotation of the converged
    Hamiltonian, the seed field included; an axis sneaking back into the
    pair basis would show up here. The response is not zero, so the check
    is not vacuous."""
    h = _converged_v1_chain()
    kw = dict(energies=np.linspace(-0.4, 0.4, 11), delta=2e-2, nk=100,
              q=[0.05, 0., 0.])
    def trace(hi):
        _, chis = hi.get_spinchi_full(**kw)
        return np.array([np.trace(c).imag for c in chis])
    t0 = trace(h)
    assert np.max(np.abs(t0)) > 1e-3
    for vector, angle in (([0., 1., 0.], 0.5), ([1., 0.3, 0.2], 0.31)):
        hr = h.copy()
        hr.global_spin_rotation(vector=vector, angle=angle)
        assert np.max(np.abs(trace(hr) - t0)) < 1e-8
