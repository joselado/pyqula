import numpy as np
import pytest

from pyqula import geometry
from pyqula import heterostructures
from pyqula.transporttk import kappa as kappa_mod
from pyqula.transporttk.localprobe import LocalProbe


def _sc_junction(delta=0.3, transparency=0.3):
    """Two-lead heterostructure with both leads genuinely superconducting
    -- the case get_kappa_ratio/get_kappa_finite_temperature_energies is
    meant to diagnose (SC vs normal power-law-ratio of the conductance)."""
    g = geometry.chain()
    h1 = g.get_hamiltonian(); h1.add_swave(delta)
    h2 = g.get_hamiltonian(); h2.add_swave(delta)
    HT = heterostructures.build(h1, h2)
    HT.set_coupling(transparency)
    HT.delta = 1e-4
    return HT


def test_get_kappa_finite_temperature_energies_does_not_recurse_or_reference_undefined_helper(monkeypatch):
    """Regression guard for the exact bug this function used to have:
    get_kappa_finite_temperature_energies was defined twice under the same
    name (the second definition shadowed the first and called itself,
    causing infinite recursion) and the first definition referenced an
    undefined get_conductances_finite_temp. Stub out the expensive
    physics (Heterostructure.didv) with an instant fake so this test only
    exercises the plumbing -- SC/normal branch generation, coupling scan,
    thermal-quadrature wiring, self-energy sharing -- not the actual
    Floquet-Keldysh solve, which is what makes a real run of this
    function slow (see _shared_selfenergy_for_branch's docstring)."""
    calls = []
    def fake_didv(self, energy=0.0, **kwargs):
        calls.append(energy)
        return 1.0 + 0.01*abs(energy) # smooth, coupling-independent stand-in
    monkeypatch.setattr(heterostructures.Heterostructure, "didv", fake_didv)

    HT = _sc_junction()
    out = kappa_mod.get_kappa_finite_temperature_energies(
        HT, energies=[0.05, 0.1], temp=0.02)
    assert np.all(np.isfinite(out))
    assert len(out) == 2
    assert len(calls) > 0 # the stub was actually reached, not just imported


def test_get_kappa_finite_temperature_energies_accepts_caller_selfenergy_qtci(monkeypatch):
    """branch_kappas builds its own extra={"selfenergy_qtci": shared} for
    the superconducting branch and used to merge it into the call with
    get_conductances_finite_temp(...,**extra,**kwargs) -- if the caller's
    own kwargs already contained selfenergy_qtci (a real, documented
    didv/dc_current kwarg), that raised "got multiple values for keyword
    argument" instead of letting the freshly-built, branch-specific
    interpolant take precedence."""
    def fake_didv(self, energy=0.0, **kwargs):
        return 1.0 + 0.01*abs(energy)
    monkeypatch.setattr(heterostructures.Heterostructure, "didv", fake_didv)

    HT = _sc_junction()
    out = kappa_mod.get_kappa_finite_temperature_energies(
        HT, energies=[0.05, 0.1], temp=0.02,
        selfenergy_qtci={"caller": "value"})
    assert np.all(np.isfinite(out))


def test_shared_selfenergy_for_branch_only_builds_for_both_sc_leads():
    """_shared_selfenergy_for_branch is the piece that lets keldysh_didv's
    self-energy interpolant be reused across a whole finite-temperature
    sweep instead of rebuilt at every (coupling, energy, thermal-node)
    combination -- see its docstring. It must only kick in when both
    leads are actually superconducting (didv's "auto" method otherwise
    picks the already-cheap "smatrix" formula, see transporttk.didv.didv),
    and the interpolant it builds must actually converge."""
    HT = _sc_junction()
    ht_sc = kappa_mod.generate_HT(HT, SC=True)
    ht_normal = kappa_mod.generate_HT(HT, SC=False)

    shared_normal = kappa_mod._shared_selfenergy_for_branch(
        ht_normal, [0.05, 0.1], 0.02, nmax_max=4)
    assert shared_normal is None

    shared_sc = kappa_mod._shared_selfenergy_for_branch(
        ht_sc, [0.05, 0.1], 0.02, nmax_max=4)
    assert shared_sc is not None
    assert all(s.converged for s in shared_sc.values())


def test_get_conductances_finite_temp_matches_direct_thermal_didv_calls():
    """get_conductances_finite_temp must be a thin, correct wrapper: its
    conductances should exactly match calling HT.didv(energy=e,temp=temp)
    directly at the same coupling, for a plain normal-normal junction
    (cheap: routes through "smatrix" + thermaldidv.finite_T_didv, no
    Floquet-Keldysh involved)."""
    g = geometry.chain()
    h1 = g.get_hamiltonian(); h2 = g.get_hamiltonian()
    HT = heterostructures.build(h1, h2)
    HT.delta = 1e-4
    c = 0.4

    energies = [0.1, 0.2]
    temp = 0.05
    ts, Gs = kappa_mod.get_conductances_finite_temp(
        HT=HT, energies=energies, temp=temp, T=c)
    # get_conductances_finite_temp scans two couplings (0.9*T, 1.1*T)
    # around its own reference T -- reproduce the exact same couplings
    # directly through HT.didv(temp=...) and compare there.
    for i, t in enumerate(ts):
        HT.set_coupling(t)
        direct = [HT.didv(energy=e, temp=temp) for e in energies]
        for g_direct, g_wrapped in zip(direct, Gs[i]):
            assert abs(g_direct-g_wrapped) < 1e-8


def test_localprobe_didv_now_actually_applies_temp(monkeypatch):
    """LocalProbe.didv used to call the bare method-selecting didv()
    directly instead of generic_didv, so a `temp` kwarg was silently
    swallowed (accepted into **kwargs, then discarded by whichever of
    didv_BdG/get_smatrix/keldysh_didv it landed in) -- callers thought
    they were getting a thermally-averaged conductance but silently got
    the T=0 one instead. Confirm temp is now genuinely applied: patch
    finite_T_didv (only reached when temp!=0) with a spy so it's provably
    invoked, and confirm zero_T_didv_1D's zero-temperature call chain is
    otherwise untouched by requiring plain zero-temperature didv to
    remain unaffected (dimensionality attribute defaults to 1D, matching
    LocalProbe's only supported case)."""
    from pyqula.transporttk import thermaldidv
    calls = []
    orig = thermaldidv.finite_T_didv
    def spy(self, temp, energy=0.0, **kwargs):
        calls.append(temp)
        return orig(self, temp, energy=energy, **kwargs)
    monkeypatch.setattr(thermaldidv, "finite_T_didv", spy)
    # generic_didv imports finite_T_didv at call time (see didv.py:
    # "from .thermaldidv import finite_T_didv"), so patch that binding too.
    from pyqula.transporttk import didv as didv_mod
    monkeypatch.setattr(didv_mod, "finite_T_didv", spy)

    g = geometry.chain()
    h = g.get_hamiltonian(); h.shift_fermi(1.); h.add_swave(0.1)
    lp = LocalProbe(h, delta=1e-3)
    lp.T = 0.2
    assert lp.dimensionality == 1

    lp.didv(energy=0.05, temp=0.02)
    assert len(calls) == 1 and calls[0] == 0.02

    # zero-temperature path (the default) must still take the old,
    # generic_didv-free-looking route with no behavior change.
    calls.clear()
    val_zero_T = lp.didv(energy=0.05)
    assert len(calls) == 0
