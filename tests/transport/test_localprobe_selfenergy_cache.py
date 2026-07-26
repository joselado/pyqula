import numpy as np

from pyqula import geometry
from pyqula.transporttk.localprobe import LocalProbe
from pyqula.transporttk import localprobe as lp_mod
from pyqula.transporttk import kappa as kappa_mod


def _sc_localprobe(delta=1e-3, T=0.2):
    g = geometry.chain()
    h = g.get_hamiltonian()
    h.add_swave(0.1)
    lp = LocalProbe(h, delta=delta)
    lp.T = T
    return lp


def test_reuse_selfenergy_returns_identical_values():
    """LocalProbe.reuse_selfenergy is purely a memoization flag: turning it
    on must not change get_selfenergy's return value, only how often the
    underlying Sancho-Rubio/sample-GF solve actually runs (see
    get_central_gmatrix, where the coupling T never enters either
    selfenergy)."""
    lp = _sc_localprobe()
    energy = 0.05
    sel0 = lp.get_selfenergy(energy, lead=0)
    sel1 = lp.get_selfenergy(energy, lead=1)

    lp.reuse_selfenergy = True
    lp._selfenergy_cache = {}
    sel0_cached = lp.get_selfenergy(energy, lead=0)
    sel1_cached = lp.get_selfenergy(energy, lead=1)

    assert np.allclose(sel0, sel0_cached)
    assert np.allclose(sel1, sel1_cached)


def test_reuse_selfenergy_actually_skips_recomputation():
    """With reuse_selfenergy on, a second get_selfenergy call at the same
    (energy, lead) must not touch the expensive Sancho-Rubio iteration
    again -- otherwise the cache exists but buys nothing."""
    lp = _sc_localprobe()
    lp.reuse_selfenergy = True
    lp._selfenergy_cache = {}

    calls = []
    orig = lp_mod.lead_selfenergy
    def spy(self, energy=0.0, **kwargs):
        calls.append(energy)
        return orig(self, energy=energy, **kwargs)
    lp_mod.lead_selfenergy = spy
    try:
        lp.get_selfenergy(0.05, lead=0)
        lp.get_selfenergy(0.05, lead=0)
        lp.get_selfenergy(0.05, lead=0)
    finally:
        lp_mod.lead_selfenergy = orig

    assert len(calls) == 1


def test_get_conductances_cache_scope_halves_selfenergy_calls():
    """transporttk.kappa.get_conductances scans two couplings (0.9T,
    1.1T) at fixed energy -- the whole point of _selfenergy_cache_scope is
    that this no longer means two independent selfenergy solves."""
    lp = _sc_localprobe()

    calls = []
    orig = lp_mod.lead_selfenergy
    def spy(self, energy=0.0, **kwargs):
        calls.append(energy)
        return orig(self, energy=energy, **kwargs)
    lp_mod.lead_selfenergy = spy
    try:
        kappa_mod.get_conductances(HT=lp, T=lp.T, energies=[0.05])
    finally:
        lp_mod.lead_selfenergy = orig

    assert len(calls) == 1  # not 2 (one per coupling point)


def test_cache_scope_restores_prior_state():
    """_selfenergy_cache_scope must not leak reuse_selfenergy=True (or a
    stale cache) onto the object once get_conductances returns -- other
    call sites never expect a LocalProbe to silently start caching."""
    lp = _sc_localprobe()
    assert lp.reuse_selfenergy is False
    kappa_mod.get_conductances(HT=lp, T=lp.T, energies=[0.05])
    assert lp.reuse_selfenergy is False
    assert lp._selfenergy_cache == {}


def test_cache_scope_noop_for_objects_without_the_flag():
    """Heterostructure (or any object without reuse_selfenergy) must be
    accepted as a no-op, not raise."""
    class Dummy:
        pass
    with kappa_mod._selfenergy_cache_scope(Dummy()):
        pass  # must not raise
