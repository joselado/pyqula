import numpy as np
import pytest

jax = pytest.importorskip("jax")

from pyqula import geometry, heterostructures
from pyqula.transporttk.localprobe import LocalProbe
from pyqula.transporttk import kappa as kappa_mod
from pyqula.transporttk import kappa_jax


def _normal_probe_localprobe(delta=1e-3, T=0.2):
    """Normal (non-SC) probe + SC sample -- the case get_kappa_ratio_jax
    targets: didv routes through the BdG smatrix formula, not Keldysh."""
    g = geometry.chain()
    h = g.get_hamiltonian()
    h.add_swave(0.1)
    lp = LocalProbe(h, delta=delta)
    lp.T = T
    return lp


def _sc_probe_localprobe(delta=1e-3, T=0.3):
    """Both probe and sample superconducting -- routes through
    Floquet-Keldysh dc_current, outside kappa_jax's scope."""
    g = geometry.chain()
    h = g.get_hamiltonian(); h.shift_fermi(1.); h.add_swave(0.1)
    lead = geometry.chain().get_hamiltonian(); lead.shift_fermi(1.); lead.add_swave(0.1)
    lp = LocalProbe(h, lead=lead, delta=delta)
    lp.T = T
    return lp


def test_get_kappa_ratio_jax_matches_numeric_secant():
    """The exact jax.grad derivative should agree with the existing
    2-point log-log secant to within the secant's own finite-difference
    curvature bias -- not exactly (they estimate d(log G)/d(log T)
    differently), but tightly, for a smooth, off-resonance energy."""
    lp = _normal_probe_localprobe()
    energy = 0.05

    ht_sc = kappa_mod.generate_HT(lp, SC=True)
    ht_normal = kappa_mod.generate_HT(lp, SC=False)
    fast = kappa_jax.get_kappa_ratio_jax(ht_sc, ht_normal, energy=energy, T=lp.T)
    assert fast is not None

    ht_sc2 = kappa_mod.generate_HT(lp, SC=True)
    ht_normal2 = kappa_mod.generate_HT(lp, SC=False)
    k1 = kappa_mod.get_kappa(HT=ht_sc2, energy=energy, T=lp.T)
    k2 = kappa_mod.get_kappa(HT=ht_normal2, energy=energy, T=lp.T)
    numeric = k1/k2

    assert abs(fast-numeric) < 1e-2
    assert np.isfinite(fast)


def test_get_kappa_uses_jax_path_by_default_and_agrees_with_forced_fallback():
    """LocalProbe.get_kappa (the public API) must actually take the jax
    branch when applicable, and disabling it must reproduce (up to the
    secant's own bias) the same physics -- guards the wiring in
    transporttk.kappa.get_kappa_ratio, not just kappa_jax in isolation."""
    lp = _normal_probe_localprobe()
    energy = 0.05

    default_val = lp.get_kappa(energy=energy)

    orig = kappa_jax.get_kappa_ratio_jax
    kappa_jax.get_kappa_ratio_jax = lambda *a, **k: None
    try:
        fallback_val = lp.get_kappa(energy=energy)
    finally:
        kappa_jax.get_kappa_ratio_jax = orig

    assert np.isfinite(default_val) and np.isfinite(fallback_val)
    assert abs(default_val-fallback_val) < 1e-2
    assert default_val != fallback_val  # forcing None must actually change the code path taken


def test_applicable_rejects_superconducting_probe():
    """A superconducting probe routes didv through Floquet-Keldysh, not
    the BdG smatrix formula kappa_jax's tail reimplements -- applicable()
    must decline it so get_kappa_ratio_jax falls back cleanly instead of
    silently computing the wrong quantity."""
    lp = _sc_probe_localprobe()
    ht_sc = kappa_mod.generate_HT(lp, SC=True)
    assert kappa_jax.applicable(ht_sc) is False


def test_applicable_rejects_non_localprobe():
    """Heterostructure objects use a different get_central_gmatrix (a
    possibly multi-block hlist, not the simple 2-block LocalProbe case) --
    applicable() must reject them, not assume LocalProbe's structure."""
    g = geometry.chain()
    h1 = g.get_hamiltonian(); h1.add_swave(0.1)
    h2 = g.get_hamiltonian(); h2.add_swave(0.1)
    ht = heterostructures.build(h1, h2)
    assert kappa_jax.applicable(ht) is False


def test_get_kappa_ratio_jax_returns_none_for_keldysh_case():
    lp = _sc_probe_localprobe()
    ht_sc = kappa_mod.generate_HT(lp, SC=True)
    ht_normal = kappa_mod.generate_HT(lp, SC=False)
    assert kappa_jax.get_kappa_ratio_jax(ht_sc, ht_normal, energy=0.25, T=lp.T) is None


def test_get_kappa_ratio_jax_returns_none_for_zero_coupling():
    lp = _normal_probe_localprobe()
    ht_sc = kappa_mod.generate_HT(lp, SC=True)
    ht_normal = kappa_mod.generate_HT(lp, SC=False)
    assert kappa_jax.get_kappa_ratio_jax(ht_sc, ht_normal, energy=0.05, T=0.0) is None


def test_decay_constant_keldysh_case_still_works_end_to_end():
    """Regression guard: the superconducting-probe LocalProbe.get_kappa
    path (examples/transport/decay_constant_keldysh) must still complete
    and return a finite value now that get_kappa_ratio tries the jax path
    first -- it must decline (see test above) and fall through to the
    unmodified Keldysh code, not break it."""
    lp = _sc_probe_localprobe(delta=1e-3, T=0.3)
    D = 0.1
    val = lp.get_kappa(energy=1.3*2*D, nmax=4, nmax_max=12, tol=5e-2)
    assert np.isfinite(val)
