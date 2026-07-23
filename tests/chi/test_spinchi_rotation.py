import numpy as np
import pytest

from pyqula import islands
from testutils import assert_all_consistent, SCF_MAXERROR

ENERGIES = np.linspace(-0.4, 0.4, 21)


def _converged_island():
    """A small zigzag-edged honeycomb island, converged from an explicit
    initial guess -- a copy of the bare Hamiltonian with an exchange field
    along a random direction v -- rather than the SCF's own mf="random"
    mode, which was found to converge erratically (occasionally very
    slowly) for this geometry. A tiny persistent seed field along the same
    direction v is also added to h itself: without any field on h, this
    system's self-consistent solution is the trivial (unmagnetized)
    paramagnetic one regardless of the initial guess, so the seed is what
    actually selects a magnetically ordered solution, along a definite
    (here: random, not fixed) direction.

    filling=0.3 (not the more obvious 0.5, half-filling) is deliberate:
    at exactly half-filling this system's density-density SCF never
    generates any interaction-driven magnetic correction at all (verified
    directly -- the converged Hamiltonian differs from the seeded input by
    less than 1e-13 regardless of U or seed strength), so the converged
    magnetization there is just the input seed field passed through
    unchanged, and a same-direction check wouldn't actually exercise the
    SCF. At filling=0.3 the SCF does contribute substantially (the
    converged Hamiltonian differs from the seeded input by order 1), so
    checking its direction against v is a genuine test of self-consistent
    convergence, not just of the seed surviving unchanged.

    Returns the converged Hamiltonian together with v, so callers can
    check the converged magnetization direction matches the seed/initial-
    guess direction."""
    g = islands.get_geometry(name="honeycomb", n=1.2, nedges=3)  # 6 sites
    h = g.get_hamiltonian()
    v = np.random.random(3) - .5  # random exchange direction
    v = v / np.sqrt(v.dot(v))  # normalize
    h.add_exchange(1e-2*v)
    mf = h.copy()
    mf.add_exchange(0.5*v)  # initial guess
    hmf = h.get_mean_field_hamiltonian(U=3.0, filling=0.3, mf=mf,
                                        maxerror=SCF_MAXERROR)
    return hmf, v


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
    h, v = _converged_island()

    # the SCF must actually converge to a solution magnetized along the
    # seed/initial-guess direction v, not some unrelated (or zero) direction.
    # The threshold (1.0) is well above the ~0.06 that the persistent seed
    # field alone would give with no interaction-driven contribution at all
    # (and well below the ~3.0 actually converged to here), so this also
    # catches a regression where the SCF stops contributing anything.
    mtot = h.get_magnetization().sum(axis=0)
    assert np.linalg.norm(mtot) > 1.0, "converged to a non-magnetic solution"
    assert abs(np.dot(mtot/np.linalg.norm(mtot), v) - 1.) < 1e-3

    # a single-axis rotation (z -> x) and a generic compound rotation
    h_axis = h.copy()
    h_axis.global_spin_rotation(vector=[0., 1., 0.], angle=0.5)
    h_generic = h.copy()
    h_generic.global_spin_rotation(vector=[0., 0., 1.], angle=0.17)
    h_generic.global_spin_rotation(vector=[0., 1., 0.], angle=0.31)
    h_generic.global_spin_rotation(vector=[1., 0., 0.], angle=0.08)

    outs = [_trace_imag(hi, RPA=RPA) for hi in (h, h_axis, h_generic)]
    assert_all_consistent(outs, 1e-6, "spinchi_full trace under spin rotation")
