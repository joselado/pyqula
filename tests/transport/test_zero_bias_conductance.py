import numpy as np
import pytest

from pyqula import geometry, heterostructures
from pyqula.greentk.rg import (green_renormalization_python,
                               green_renormalization_jit,
                               green_renormalization_jit_batch)


def _chain_lead():
    return geometry.chain().get_hamiltonian()


def _junction(hr):
    ht = heterostructures.build(_chain_lead(), hr)
    ht.set_coupling(1.0)
    return ht


@pytest.mark.parametrize("delta", [1e-3, 1e-6, 1e-9, 1e-12])
def test_surface_green_function_of_a_chain_at_the_band_centre(delta):
    """The retarded surface Green's function of a semi-infinite chain with
    t=1 at E=0 is exactly -i/t, for any broadening. E=0 is the worst case
    for the Sancho-Rubio decimation -- it has to invert e-intra=i*delta at
    the very first step -- and at small delta the decimation converges to a
    wrong fixed point (-8932j at delta=1e-12), which is why the result is
    checked against its own Dyson equation and repaired."""
    intra = np.array([[0.+0j]])
    inter = np.array([[1.+0j]])
    for f in [green_renormalization_python, green_renormalization_jit]:
        _, gs = f(intra, inter, energy=0.0, delta=delta)
        assert abs(gs[0, 0] - (-1j)) < 1e-3, (f, delta, gs)
    _, gsb = green_renormalization_jit_batch(intra, inter, np.array([0.0]),
                                             delta=delta)
    assert abs(gsb[0, 0, 0] - (-1j)) < 1e-3


def test_normal_junction_zero_bias_conductance_is_the_landauer_value():
    """A perfectly transparent junction between two identical normal 1D
    leads transmits both spin channels perfectly: G = 2 at every in-band
    energy, zero bias included. Zero bias used to return 0."""
    ht = _junction(_chain_lead())
    for e in [0.0, 1e-9, 1e-6, 1e-3, 0.5]:
        assert abs(ht.didv(energy=e) - 2.0) < 1e-3, e


def test_andreev_doubling_at_zero_bias():
    """BTK: a perfectly transparent NS contact has a subgap conductance of
    exactly twice the normal state one (every incoming electron is Andreev
    reflected as a hole), so G = 4 here. This must hold at the default
    energy=0, the bias point of a zero-bias conductance peak."""
    hs = _chain_lead()
    hs.add_swave(0.01)
    ht = _junction(hs)
    assert abs(ht.didv() - 4.0) < 1e-2          # the default, energy=0
    for e in [0.0, 1e-9, 1e-4, 5e-3]:           # anywhere inside the gap
        assert abs(ht.didv(energy=e) - 4.0) < 1e-2, e
    assert abs(ht.didv(energy=0.05) - 2.0) < 0.1  # normal well above the gap
