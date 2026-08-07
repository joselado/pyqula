"""Shared timing/persistence helpers for benchmark cases.

Every case follows the same contract: for each method being compared, call
`time_cold_warm` (first call, which may pay a numba/jax compile cost, timed
separately from a second steady-state call), compute a numerical value to
compare against a designated reference method, and hand the resulting
records to `save_records`. `benchmarks/report.py` reads them back with
`load_records`.
"""
import json
import os
import platform
import time


def time_cold_warm(fn):
    """Call fn() twice, returning (t_cold, t_warm, result).

    t_cold includes any first-call compilation cost (numba/jax JIT); t_warm
    is the steady-state cost. `result` is the warm call's return value.
    """
    t0 = time.perf_counter()
    fn()
    t_cold = time.perf_counter() - t0
    t0 = time.perf_counter()
    result = fn()
    t_warm = time.perf_counter() - t0
    return t_cold, t_warm, result


def machine_info():
    """Capture enough about the current machine that results from two
    different machines are never silently compared as if equivalent."""
    info = {
        "platform": platform.platform(),
        "processor": platform.processor() or "unknown",
        "cpu_count": os.cpu_count(),
    }
    for name in ("numpy", "scipy", "numba"):
        try:
            mod = __import__(name)
            info[name] = getattr(mod, "__version__", "unknown")
        except Exception:
            info[name] = None
    try:
        import jax
        info["jax"] = jax.__version__
        info["jax_devices"] = [str(d) for d in jax.devices()]
    except Exception:
        info["jax"] = None
        info["jax_devices"] = []
    try:
        from pyqula import parallel
        info["pyqula_cores"] = parallel.cores
    except Exception:
        info["pyqula_cores"] = None
    return info


def save_records(results_dir, case_name, records, machine):
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, f"{case_name}.json")
    with open(path, "w") as f:
        json.dump({"case": case_name, "machine": machine, "records": records}, f, indent=2)
    return path


def load_records(results_dir, case_name):
    path = os.path.join(results_dir, f"{case_name}.json")
    with open(path) as f:
        return json.load(f)
