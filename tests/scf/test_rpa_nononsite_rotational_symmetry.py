import numpy as np
import pytest

from pyqula import geometry
from pyqula.meanfield import VJinteraction

ENERGIES = np.linspace(-0.4, 0.4, 11)


def _converged_v1_chain():
    """A chain converged by VJinteraction with ONLY a nearest-neighbor
    density-density interaction (V1) -- see
    test_rpa_nononsite_ferro_chain.py for why this converges to a genuine
    ferromagnetic moment at low filling. H.V here is a real, multi-key
    (non-onsite) hopping dict, exercising the generalized
    _full_spin_U/interaction_at_q code path this rotational-symmetry
    check is meant to validate."""
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


def _trace_imag(h, **kwargs):
    es, chis = h.get_spinchi_full(energies=ENERGIES, delta=2e-2, nk=100, **kwargs)
    return np.array([np.trace(c).imag for c in chis])


@pytest.mark.slow
def test_spinchi_full_is_rotationally_symmetric_for_nononsite_interaction():
    """Tr[Im chi(w)] must be the same for the V1-converged Hamiltonian and
    for that exact same solution after a global spin rotation -- mirrors
    tests/chi/test_spinchi_rotation.py's onsite-U check, here specifically
    for a H.V with neighbor-shell (non-onsite) support, to confirm the
    generalized _full_spin_U (built per-direction, Fourier-summed by
    interaction_at_q) is still exactly rotationally covariant."""
    h = _converged_v1_chain()

    h_axis = h.copy()
    h_axis.global_spin_rotation(vector=[0., 1., 0.], angle=0.5)
    h_generic = h.copy()
    h_generic.global_spin_rotation(vector=[0., 0., 1.], angle=0.17)
    h_generic.global_spin_rotation(vector=[0., 1., 0.], angle=0.31)
    h_generic.global_spin_rotation(vector=[1., 0., 0.], angle=0.08)

    t0 = _trace_imag(h)
    t1 = _trace_imag(h_axis)
    t2 = _trace_imag(h_generic)

    scale = np.max(np.abs(t0))
    assert scale > 1e-3, "trivial (near-zero) response, rotation check would be vacuous"
    assert np.allclose(t0, t1, atol=1e-8)
    assert np.allclose(t0, t2, atol=1e-8)
