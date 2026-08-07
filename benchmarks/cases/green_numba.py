"""Smoke-test case: greentk.rg.green_renormalization(numba=True) vs
numba=False (pure-Python) Sancho-Rubio iteration on the same lead. This is
the lowest-risk case in the suite -- same function, one kwarg flips the
backend -- and exists to validate the harness/report pipeline end-to-end
before trusting it on the richer cases.

`size` is the lead's matrix dimension (2 sites per unit cell x n cells).
"""
import numpy as np

from pyqula import geometry
from pyqula.greentk.rg import green_renormalization

from benchmarks.harness import time_cold_warm

CASE_NAME = "green_renormalization_numba"
SIZES_QUICK = [4, 16, 40]
SIZES_FULL = [4, 16, 40, 100, 200]

METHODS = (("python", False), ("numba", True))


def _lead(n):
    h = geometry.chain(n).get_hamiltonian()
    hop = h.get_multihopping().dict
    return hop[(0, 0, 0)], hop[(1, 0, 0)]


def _quantity(g_bulk):
    """-Im[Tr g] / pi, the same scalar DOS-like quantity dos.py's Green
    mode extracts from a Green's function -- a reasonable, cheap check that
    both backends converged to the same physics, not just "ran"."""
    return float(-np.trace(g_bulk).imag / np.pi)


def run(sizes):
    records = []
    for n in sizes:
        intra, inter = _lead(n)
        dim = intra.shape[0]
        ref = None
        batch = []
        for method, numba in METHODS:
            def call(numba=numba):
                return green_renormalization(intra, inter, numba=numba, energy=0.0, delta=1e-2)
            t_cold, t_warm, (g_bulk, g_surf) = time_cold_warm(call)
            value = _quantity(g_bulk)
            if method == "python":
                ref = value
            batch.append(dict(
                case=CASE_NAME, method=method, size=dim,
                t_cold=t_cold, t_warm=t_warm, value=value,
                meta=dict(n_cells=n),
            ))
        for rec in batch:
            rec["reldiff"] = abs(rec["value"] - ref) / (abs(ref) + 1e-300)
        records.extend(batch)
    return records
