"""kappa is the power k of G ~ t^k, fitted from two couplings. A negative
dI/dV is physical between two superconductors, so a conductance of one
sign is fitted through |G|; a conductance that changes sign across the
sweep (unconverged sidebands) used to crash np.polyfit with LinAlgError,
and now raises a ValueError naming the conductances and the remedy."""
import numpy as np
import pytest

from pyqula.transporttk.kappa import get_power


def test_power_law_is_recovered_for_either_sign():
    ts = np.array([0.27, 0.33])
    for k in [1.0, 2.0, 3.5]:
        assert np.isclose(get_power(ts, 0.7*ts**k), k)
        assert np.isclose(get_power(ts, -0.7*ts**k), k)


def test_sign_change_raises_a_clear_error():
    ts = np.array([0.27, 0.33])
    with pytest.raises(ValueError, match="nmax_max"):
        get_power(ts, np.array([-0.0103, 0.0072]), energy=0.05)
    with pytest.raises(ValueError, match="one sign"):
        get_power(ts, np.array([0.0, 0.01]))
