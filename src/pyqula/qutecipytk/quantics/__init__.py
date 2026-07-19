"""pyqula.qutecipytk.quantics: Python port of QuanticsGrids.jl.

0-based throughout (grid indices, quantics digit values, bitnumbers) -- see
CLAUDE.md's "QuanticsGrids" section for the full list of deliberate
deviations from the Julia original (always-tuple returns, no overflow
guard, no kwargs convenience API, no base-2 fast path).
"""
from pyqula.qutecipytk.quantics.discretized import DiscretizedGrid, quantics_function
from pyqula.qutecipytk.quantics.grid import InherentDiscreteGrid

__all__ = ["InherentDiscreteGrid", "DiscretizedGrid", "quantics_function"]
