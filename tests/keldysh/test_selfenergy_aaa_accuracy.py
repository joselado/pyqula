import pytest

from pyqula import geometry
from pyqula import heterostructures
from pyqula.keldyshtk.current import build_selfenergy_aaa, dc_current
from pyqula.transporttk.didv import keldysh_didv

# Regression coverage for documentation/keldysh_aaa_selfenergy_accuracy_plan.md:
# selfenergy_method="aaa" used to have a current-error gap that grew with the
# fitting window (nmax_max) -- 0.5% at nmax_max=4 up to 9.8% at nmax_max=40 on
# this exact case -- while SelfenergyAAA's own held-out validation_error
# stayed flat and never detected it (root cause: candidate-grid under-
# resolution, both at the lead's gap-edge singularity and, more importantly
# for the current-error trend, across the fit's broader "bulk" domain --
# NOT RGF-chain error amplification, which was directly ruled out). The fix
# (aaatk/selfenergy_aaa.py's domain-independent validation sampling plus
# curvature-driven adaptive grid refinement, `_refine_grid`) is only
# meaningfully tested by checking AGAINST THE ACTUAL CURRENT across that same
# sweep, per that document's own closing point -- validation_error was shown
# not to be a reliable proxy for it, so asserting on it here would miss
# exactly the failure mode this guards against.


def _sc_sc_junction(delta_sc, transparency, ht_delta=1e-4):
    """Two-lead SC-SC junction, same shape as the cases benchmarked in
    keldysh_sideband_decimation_plan.md's "fixed" quadrature updates and
    keldysh_aaa_selfenergy_accuracy_plan.md's item-3 sweep."""
    h1 = geometry.chain().get_hamiltonian(); h1.shift_fermi(1.); h1.add_swave(delta_sc)
    h2 = geometry.chain().get_hamiltonian(); h2.shift_fermi(1.); h2.add_swave(delta_sc)
    HT = heterostructures.build(h1, h2)
    HT.set_coupling(transparency)
    HT.delta = ht_delta
    return HT


@pytest.mark.parametrize("nmax_max", [8, 24, 40])
def test_aaa_current_matches_direct_across_the_nmax_max_sweep(nmax_max):
    """The case and nmax_max values that exposed the original accuracy gap
    (see module docstring): AAA-vs-direct relative current error must stay
    small at every window size, not just the small ones -- the growing-error
    trend the fix targets only shows up once nmax_max (hence the fitting
    window) gets large, so a single small-window check would not have caught
    the original bug."""
    delta_sc, transparency, voltage = 0.3, 0.5, 0.18
    HT = _sc_sc_junction(delta_sc, transparency, ht_delta=1e-4)

    Idirect = dc_current(HT, voltage, fixed_nmax=nmax_max, nmax_max=nmax_max,
                          selfenergy_method="direct")
    interp = build_selfenergy_aaa(HT, voltage, nmax_max, delta=1e-4)
    assert all(s.converged for s in interp.values())
    Iaaa = dc_current(HT, voltage, fixed_nmax=nmax_max, nmax_max=nmax_max,
                       selfenergy_qtci=interp)

    relerr = abs(Iaaa - Idirect) / max(abs(Idirect), 1e-12)
    assert relerr < 2e-2


def test_aaa_didv_matches_direct_on_the_catastrophic_cancellation_case():
    """keldysh_didv's finite difference divides by Ip-Im, which
    keldysh_sideband_decimation_plan.md found amplifies a per-branch AAA
    self-energy error by ~12x (|Ip|/|Ip-Im|) on this exact case -- the
    pre-fix combination of a loose validation check and an over-tight
    default tolerance produced a 37-60% dI/dV error here even though the
    plain dc_current error looked modest. This is the discriminating check
    the accuracy plan's own review flagged: passing dc_current-level
    agreement alone would not have caught this amplified failure mode."""
    delta_sc, transparency, voltage, nmax_max = 0.3, 0.5, 0.18, 40
    HT = _sc_sc_junction(delta_sc, transparency, ht_delta=1e-4)

    Gdirect = keldysh_didv(HT, voltage=voltage, delta=1e-4, use_aaa=False,
                            nmax_max=nmax_max, fixed_nmax=nmax_max)
    Gaaa = keldysh_didv(HT, voltage=voltage, delta=1e-4, use_aaa=True,
                         nmax_max=nmax_max, fixed_nmax=nmax_max)

    relerr = abs(Gaaa - Gdirect) / max(abs(Gdirect), 1e-12)
    assert relerr < 5e-2
