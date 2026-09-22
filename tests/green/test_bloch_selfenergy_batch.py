"""greentk.selfenergy.bloch_selfenergy_batch must answer, for a whole set
of energies at once, exactly what bloch_selfenergy answers one at a time.

Only a 1d first-neighbour Hamiltonian solved by decimation actually
batches (one numba prange-parallel Sancho-Rubio call for the whole set);
everything else falls back to a loop, so the contract this file pins is
that the two branches are indistinguishable from the outside.
"""
import numpy as np
import pytest

from pyqula import geometry
from pyqula.greentk.selfenergy import bloch_selfenergy, bloch_selfenergy_batch


ENERGIES = np.array([-0.4, -0.05, 0.0, 0.13, 0.62])


def _scalar(h, es, **kwargs):
    out = [bloch_selfenergy(h, energy=e, **kwargs) for e in es]
    return (np.array([o[0] for o in out]), np.array([o[1] for o in out]))


@pytest.mark.parametrize("gtype", ["bulk", "surface"])
def test_batch_matches_scalar_on_the_batched_branch(gtype):
    """A superconducting chain: 1d, first-neighbour, mode='adaptive' --
    the one shape that goes through green_renormalization_jit_batch."""
    h = geometry.chain().get_hamiltonian()
    h.shift_fermi(1.); h.add_swave(0.1)
    kwargs = dict(delta=1e-3, mode="adaptive", gtype=gtype)
    gr, sr = _scalar(h, ENERGIES, **kwargs)
    gb, sb = bloch_selfenergy_batch(h, ENERGIES, **kwargs)
    assert gb.shape == gr.shape and sb.shape == sr.shape
    assert np.max(np.abs(gb-gr)) == 0.
    assert np.max(np.abs(sb-sr)) == 0.


def test_batch_matches_scalar_on_the_fallback_branch():
    """A 2d Hamiltonian is not batchable (the adaptive selfenergy there
    integrates over a transverse k), so this exercises the loop."""
    h = geometry.honeycomb_lattice().get_hamiltonian()
    kwargs = dict(delta=1e-2, mode="adaptive", gtype="bulk")
    gr, sr = _scalar(h, ENERGIES[:3], **kwargs)
    gb, sb = bloch_selfenergy_batch(h, ENERGIES[:3], **kwargs)
    assert np.max(np.abs(gb-gr)) == 0.
    assert np.max(np.abs(sb-sr)) == 0.


def test_batch_matches_scalar_for_longer_range_hoppings():
    """Second-neighbour hopping in 1d leaves the decimation for the Dyson
    route, which the batch does not reimplement either."""
    h = geometry.chain().get_hamiltonian(tij=[1., 0.4])
    kwargs = dict(delta=1e-2, mode="adaptive", gtype="bulk")
    gr, sr = _scalar(h, ENERGIES[:3], **kwargs)
    gb, sb = bloch_selfenergy_batch(h, ENERGIES[:3], **kwargs)
    assert np.max(np.abs(gb-gr)) == 0.
    assert np.max(np.abs(sb-sr)) == 0.


def test_batch_refuses_an_unknown_gtype():
    h = geometry.chain().get_hamiltonian()
    with pytest.raises(ValueError):
        bloch_selfenergy_batch(h, ENERGIES, delta=1e-2, gtype="middle")
