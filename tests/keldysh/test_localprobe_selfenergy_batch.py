"""LocalProbe.get_selfenergy_batch must answer exactly what a loop over
LocalProbe.get_selfenergy answers.

It exists purely so that keldyshtk.current's Floquet sideband sweep stops
solving Sancho-Rubio one energy at a time (tens of thousands of scalar
solves per dI/dV point); it routes both of a probe's selfenergies through
the numba prange-parallel decimation instead. That is a cost change and
nothing else, so every check here is an equality against the scalar path
rather than a tolerance on the physics.
"""
import numpy as np
import pytest

from pyqula import geometry
from pyqula.transporttk.localprobe import LocalProbe


def _sc_probe(delta=1e-3, i=0, spinful=False, nsuper=1):
    """Superconducting probe on a superconducting chain -- the case that
    routes through Floquet-Keldysh, see test_localprobe_keldysh.py."""
    g = geometry.chain()
    if nsuper > 1: g = g.supercell(nsuper)
    h = g.get_hamiltonian(has_spin=spinful)
    h.shift_fermi(1.); h.add_swave(0.1)
    tip = geometry.chain().get_hamiltonian(has_spin=spinful)
    tip.shift_fermi(1.); tip.add_swave(0.1)
    lp = LocalProbe(h, lead=tip, delta=delta, i=i)
    lp.T = 0.3
    return lp


def _scalar(lp, es, lead, delta):
    return np.array([np.asarray(lp.get_selfenergy(e, lead=lead, delta=delta,
                                    pristine=True, numba=True)) for e in es])


ENERGIES = np.array([-0.31, -0.05, 0.0, 0.017, 0.25, 0.4])


@pytest.mark.parametrize("lead", [0, 1])
@pytest.mark.parametrize("frozen", [False, True])
def test_batched_selfenergy_matches_the_scalar_one(lead, frozen):
    """Both leads, and both settings of `frozen_lead` -- which decides
    whether the probe's own selfenergy is evaluated at its actual energy
    or pinned at zero (keldyshtk.current._prepare_bias_target sets it from
    whether the probe is superconducting, so both occur in practice)."""
    lp = _sc_probe()
    lp.frozen_lead = frozen
    ref = _scalar(lp, ENERGIES, lead, 1e-3)
    bat = np.asarray(lp.get_selfenergy_batch(ENERGIES, lead=lead, delta=1e-3,
                                              pristine=True))
    assert bat.shape == ref.shape
    assert np.max(np.abs(bat-ref)) == 0.


@pytest.mark.parametrize("lead", [0, 1])
def test_batched_selfenergy_matches_on_a_multi_orbital_sample(lead):
    """A spinful sample probed away from site 0, so the local block the
    sample selfenergy slices out is neither the first one nor 2x2 -- the
    batched slice is taken once for the whole set rather than per energy,
    and must land on the same entries."""
    lp = _sc_probe(spinful=True, i=2, nsuper=3)
    ref = _scalar(lp, ENERGIES, lead, 1e-3)
    bat = np.asarray(lp.get_selfenergy_batch(ENERGIES, lead=lead, delta=1e-3,
                                              pristine=True))
    assert bat.shape == ref.shape
    assert np.max(np.abs(bat-ref)) == 0.


@pytest.mark.parametrize("lead", [0, 1])
def test_batched_selfenergy_matches_on_a_two_dimensional_sample(lead):
    """A 2d sample takes greentk.selfenergy.bloch_selfenergy_batch's
    fallback branch (only a 1d first-neighbour decimation batches), so
    this pins that the fallback is wired up and returns the same stack."""
    h = geometry.honeycomb_lattice().get_hamiltonian()
    h.add_swave(0.1)
    tip = geometry.chain().get_hamiltonian()
    tip.shift_fermi(1.); tip.add_swave(0.05)
    lp = LocalProbe(h, lead=tip, delta=1e-3, i=1)
    ref = _scalar(lp, ENERGIES, lead, 1e-3)
    bat = np.asarray(lp.get_selfenergy_batch(ENERGIES, lead=lead, delta=1e-3,
                                              pristine=True))
    assert bat.shape == ref.shape
    assert np.max(np.abs(bat-ref)) == 0.


def test_batched_selfenergy_rejects_an_unknown_lead():
    lp = _sc_probe()
    with pytest.raises(ValueError):
        lp.get_selfenergy_batch(ENERGIES, lead=2)


def test_keldysh_didv_is_unchanged_by_the_batched_selfenergy(monkeypatch):
    """End to end: the Floquet-Keldysh dI/dV must come out bit-identical
    with the batched selfenergy and without it. keldyshtk.current picks
    the batch up with `hasattr`, so removing the method restores the
    per-energy path this replaced."""
    kwargs = dict(nmax=4, nmax_max=8, tol=5e-2)
    batched = _sc_probe().didv(energy=0.25, **kwargs)
    monkeypatch.delattr(LocalProbe, "get_selfenergy_batch")
    scalar = _sc_probe().didv(energy=0.25, **kwargs)
    assert scalar == batched
