import numpy as np
import pytest

from pyqula import islands
from testutils import assert_all_consistent

ENERGIES = np.linspace(-0.4, 0.4, 21)


def _converged_island():
    """A small zigzag-edged honeycomb island, converged with a tiny
    exchange seed so the SCF (mf='random') settles on a definite
    magnetization direction instead of an arbitrary one."""
    g = islands.get_geometry(name="honeycomb", n=1.2, nedges=3)  # 6 sites
    h = g.get_hamiltonian()
    h.add_exchange([0., 0., 1e-2])
    return h.get_mean_field_hamiltonian(U=3.0, filling=0.5, mf="random")


def _trace_imag(h, **kwargs):
    es, chis = h.get_spinchi_full(energies=ENERGIES, delta=1e-2, **kwargs)
    return np.array([np.trace(c).imag for c in chis])


@pytest.mark.slow
@pytest.mark.parametrize("RPA", [False, True])
def test_spinchi_full_is_rotationally_symmetric(RPA):
    """Tr[Im chi(w)] must be the same for a mean-field solution and for
    that exact same solution after a global spin rotation: get_spinchi_full
    sums over Sx, Sy, Sz, a rotationally-invariant combination, so a global
    spin rotation of the converged Hamiltonian must not change the result."""
    h = _converged_island()
    # a single-axis rotation (z -> x) and a generic compound rotation
    h_axis = h.copy()
    h_axis.global_spin_rotation(vector=[0., 1., 0.], angle=0.5)
    h_generic = h.copy()
    h_generic.global_spin_rotation(vector=[0., 0., 1.], angle=0.17)
    h_generic.global_spin_rotation(vector=[0., 1., 0.], angle=0.31)
    h_generic.global_spin_rotation(vector=[1., 0., 0.], angle=0.08)

    outs = [_trace_imag(hi, RPA=RPA) for hi in (h, h_axis, h_generic)]
    assert_all_consistent(outs, 1e-6, "spinchi_full trace under spin rotation")
