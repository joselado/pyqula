"""qutecipytk: vendored copy of qutecipy, a Python port of TensorCrossInterpolation.jl.

Public API mirrors the Julia package's exports (crossinterpolate1,
crossinterpolate2, TensorTrain, ...), adapted to 0-based indexing. Vendored
from https://github.com/joselado/qutecipy so pyqula can use it (e.g.
`topology.chern` with `mode="qtci"`) without an extra install step.
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
