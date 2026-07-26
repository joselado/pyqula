# Add the root path of the pyqula library
import os ; import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")

import time
import warnings

import numpy as np

from pyqula import geometry
from pyqula.transporttk.localprobe import LocalProbe
from pyqula.keldyshtk.current import dc_current, _prepare_bias_target, build_selfenergy_aaa

## Benchmarks keldyshtk.current_jax.JaxKeldyshCurrent (a JAX-differentiable
## reformulation of the Floquet-Keldysh DC current, see that module's own
## docstring for the full numerical story) against the direct, numba-based
## dc_current/keldysh_didv path this library uses by default, at a FIXED
## number of Floquet sidebands (matching, not adaptively growing, nmax --
## see current_jax.py for why). Requires the optional "jax" extra:
## pip install pyqula[jax]
##
## current() (the DC current value alone, e.g. what an I-V curve needs) is
## a large, consistent win: once JaxKeldyshCurrent is built, warm calls are
## sub-millisecond versus a full dc_current solve. didv() (the differential
## conductance, via jax.grad instead of a finite difference) needs a much
## higher quasienergy quadrature order to be trustworthy for a resonance-
## rich system like the one below, which can erode or reverse that
## advantage depending on how many Floquet sidebands (nmax) the physics
## actually needs -- this script measures both, at two different nmax, to
## show that split directly rather than only reporting a single flattering
## number.
try:
    import jax
except ImportError:
    print("This benchmark needs the optional jax dependency: pip install pyqula[jax]")
    sys.exit(0)
from pyqula.keldyshtk.current_jax import JaxKeldyshCurrent


def make_localprobe():
    """SC probe + SC sample: the case that routes through the expensive
    Floquet-Keldysh MAR path (see the user guide's "Multiple Andreev
    reflection" section), the workload both current_jax.py and this
    benchmark target."""
    h = geometry.chain().get_hamiltonian(); h.shift_fermi(1.); h.add_swave(0.1)
    lead = geometry.chain().get_hamiltonian(); lead.shift_fermi(1.); lead.add_swave(0.1)
    lp = LocalProbe(h, lead=lead, delta=1e-3)
    lp.T = 0.3
    return lp


def direct_didv(ht, voltage, nmax, delta=None, dv=None):
    """Same central finite difference LocalProbe.didv(method="keldysh")
    uses internally, but at a fixed nmax (nmax==nmax_max disables
    dc_current's adaptive sideband growth) so this is directly comparable
    to JaxKeldyshCurrent, which is also fixed-nmax."""
    htb = _prepare_bias_target(ht)
    if delta is None: delta = htb.delta
    if dv is None: dv = max(abs(voltage)*1e-2, 1e-3)
    shared = build_selfenergy_aaa(htb, abs(voltage)+dv, nmax, delta=delta)
    Ip = dc_current(ht, voltage+dv, nmax=nmax, nmax_max=nmax, delta=delta, selfenergy_qtci=shared)
    Im = dc_current(ht, voltage-dv, nmax=nmax, nmax_max=nmax, delta=delta, selfenergy_qtci=shared)
    return (Ip-Im)/(2*dv)


def benchmark(nmax, voltage=0.25, n_repeat=3):
    print(f"\n=== nmax={nmax}, voltage={voltage} ===")
    lp = make_localprobe()

    t0 = time.perf_counter()
    jkc = JaxKeldyshCurrent(lp, nmax=nmax, vmax=voltage)
    t_build = time.perf_counter()-t0
    print(f"JaxKeldyshCurrent build (JIT compile + gl_order search): "
          f"{t_build:.2f}s, chose gl_order={jkc.gl_order}")

    t0 = time.perf_counter()
    for _ in range(n_repeat): jax_val = jkc.current(voltage)
    t_jax_current = (time.perf_counter()-t0)/n_repeat
    t0 = time.perf_counter()
    for _ in range(n_repeat): jax_didv = jkc.didv(voltage)
    t_jax_didv = (time.perf_counter()-t0)/n_repeat

    lp2 = make_localprobe()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # nmax fixed deliberately low for a fast demo
        _ = dc_current(lp2, voltage, nmax=nmax, nmax_max=nmax, delta=lp2.delta)  # warm up numba
        t0 = time.perf_counter()
        for _ in range(n_repeat):
            direct_val = dc_current(lp2, voltage, nmax=nmax, nmax_max=nmax, delta=lp2.delta)
        t_direct_current = (time.perf_counter()-t0)/n_repeat
        t0 = time.perf_counter()
        for _ in range(n_repeat):
            direct_didv_val = direct_didv(lp2, voltage, nmax)
        t_direct_didv = (time.perf_counter()-t0)/n_repeat

    print(f"{'':20s} {'JAX (warm)':>14s} {'direct':>14s} {'speedup':>10s} {'reldiff':>10s}")
    print(f"{'current(V)':20s} {t_jax_current:12.4f}s {t_direct_current:12.4f}s "
          f"{t_direct_current/t_jax_current:9.1f}x "
          f"{abs(jax_val-direct_val)/max(abs(direct_val),1e-8):9.2%}")
    print(f"{'dI/dV(V)':20s} {t_jax_didv:12.4f}s {t_direct_didv:12.4f}s "
          f"{t_direct_didv/t_jax_didv:9.1f}x "
          f"{abs(jax_didv-direct_didv_val)/max(abs(direct_didv_val),1e-8):9.2%}")


if __name__ == "__main__":
    print("keldyshtk.current_jax.JaxKeldyshCurrent vs. the direct dc_current/keldysh_didv path")
    print("(both at the SAME fixed nmax -- see current_jax.py's own docstring for why nmax is")
    print(" not adaptively grown here, and for the full numerical story behind these numbers)")
    for nmax in (8, 16):
        benchmark(nmax)
