"""Benchmark suite comparing the performance of different methods pyqula
offers for computing the same physical quantity (e.g. DOS via exact
diagonalization vs KPM vs Green's function). See
documentation/benchmark_plan.md for the design and rationale.

Entry point: `python -m benchmarks.run_all --quick` (run from the repo
root, same convention as `python -m pytest tests`).
"""
import os
import sys

_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
