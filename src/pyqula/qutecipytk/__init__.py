"""qutecipytk: vendored copy of qutecipy, a Python port of TensorCrossInterpolation.jl.

Public API mirrors the Julia package's exports (crossinterpolate1,
crossinterpolate2, TensorTrain, ...), adapted to 0-based indexing. Vendored
from https://github.com/joselado/qutecipy so pyqula can use it (e.g.
`topology.chern` with `mode="qtci"`) without an extra install step.

Provenance
----------
Upstream:  https://github.com/joselado/qutecipy
Commit:    6df39e7 ("Fix correctness bugs found by high-effort code review")
License:   MIT (declared in upstream's pyproject.toml; upstream ships no
           LICENSE file of its own). MIT is GPL-compatible, so redistributing
           it inside pyqula (GPLv3) is fine as long as this notice is kept.

This is a VERBATIM vendor: the only edits are import-path rewrites
(`qutecipy.` -> `pyqula.qutecipytk.`). Verified by diffing the whole tree
against upstream at the commit above -- identical file set, and the only
differing lines are those rewrites. Do not fix bugs here; fix them upstream
and re-vendor, or the next re-vendor silently reverts the fix.

Upstream's own 16 test files are vendored at tests/qutecipy/ (marked `slow`)
and pass against this copy, so a re-vendor can be validated in one command.
"""
from pyqula.qutecipytk.contraction import Contraction, contract
from pyqula.qutecipytk.conversion import tci1_from_tci2, tci2_from_tci1
from pyqula.qutecipytk.gausskronrod import kronrod
from pyqula.qutecipytk.integration import integrate
from pyqula.qutecipytk.tci1 import TensorCI1, crossinterpolate1
from pyqula.qutecipytk.tci2 import TensorCI2, crossinterpolate2, optimize
from pyqula.qutecipytk.tensortrain.base import AbstractTensorTrain
from pyqula.qutecipytk.tensortrain.cache import TTCache
from pyqula.qutecipytk.tensortrain.cachedfunction import CachedFunction
from pyqula.qutecipytk.tensortrain.core import TensorTrain, add, subtract, tensortrain
from pyqula.qutecipytk.util import optfirstpivot

__all__ = [
    "AbstractTensorTrain",
    "TensorTrain",
    "tensortrain",
    "add",
    "subtract",
    "TTCache",
    "CachedFunction",
    "TensorCI1",
    "crossinterpolate1",
    "TensorCI2",
    "crossinterpolate2",
    "optimize",
    "optfirstpivot",
    "tci1_from_tci2",
    "tci2_from_tci1",
    "kronrod",
    "integrate",
    "Contraction",
    "contract",
]
