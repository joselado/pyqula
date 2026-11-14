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


def test_batch_matches_scalar_on_the_tetrahedron_branch():
    """mode='triangle' of a 3d crystal batches too: one diagonalization of
    the mesh for every energy, the same simplex sum at each energy, so
    the matrices agree to rounding (the matrix products of the energies
    are grouped differently)"""
    h = geometry.diamond_lattice_minimal().get_hamiltonian(has_spin=True)
    h.add_sublattice_imbalance(0.3)
    kwargs = dict(delta=0.05, mode="triangle", nk=6)
    gr, sr = _scalar(h, ENERGIES, **kwargs)
    gb, sb = bloch_selfenergy_batch(h, ENERGIES, **kwargs)
    assert gb.shape == gr.shape and sb.shape == sr.shape
    assert np.max(np.abs(gr)) > 1e-2
    assert np.max(np.abs(gb-gr)) < 1e-12*np.max(np.abs(gr))
    assert np.max(np.abs(sb-sr)) < 1e-10*np.max(np.abs(sr))


def test_batch_passes_nk_to_the_fallback_loop():
    """mode='full' sums the Green's function over nk k-points, and the
    batch hands its nk to every call of the loop: the same matrices as the
    scalar calls with that nk, and not the ones of another nk"""
    h = geometry.honeycomb_lattice().get_hamiltonian()
    kwargs = dict(delta=5e-2, mode="full", gtype="bulk")
    gr, sr = _scalar(h, ENERGIES[:3], nk=7, **kwargs)
    go, so = _scalar(h, ENERGIES[:3], nk=9, **kwargs)
    gb, sb = bloch_selfenergy_batch(h, ENERGIES[:3], nk=7, **kwargs)
    assert np.max(np.abs(gr-go)) > 1e-3*np.max(np.abs(gr))
    assert np.max(np.abs(gb-gr)) == 0.
    assert np.max(np.abs(sb-sr)) == 0.


def test_the_tetrahedra_refuse_a_complex_energy():
    """mode='triangle' takes a real energy and the broadening in delta. A
    complex energy was cast to its real part with only a ComplexWarning, so
    an embedding got the bulk at the real energy and the Dyson step at the
    complex one; a Python complex stopped with a TypeError. Both raise a
    ValueError now, for the single energy and for the batch"""
    h = geometry.diamond_lattice_minimal().get_hamiltonian(has_spin=True)
    kwargs = dict(delta=0.05, mode="triangle", nk=4)
    for e in [np.complex128(0.3+0.4j), 0.3+0.4j]:
        with pytest.raises(ValueError, match="real energies"):
            bloch_selfenergy(h, energy=e, **kwargs)
    with pytest.raises(ValueError, match="real energies"):
        bloch_selfenergy_batch(h, np.array([0.1, 0.3+0.4j]), **kwargs)
