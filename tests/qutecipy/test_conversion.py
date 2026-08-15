import itertools

import numpy as np

from pyqula.qutecipytk.conversion import tci1_from_tci2, tci2_from_tci1, tci2_from_tensortrain
from pyqula.qutecipytk.tci2 import crossinterpolate2, optimize
from pyqula.qutecipytk.tensortrain.core import TensorTrain


def test_tci1_tci2_roundtrip():
    def f(v):
        return 1.0 / (1.0 + sum(x ** 2 for x in v))

    localdims = [6] * 5
    tci2, ranks, errors = crossinterpolate2(np.float64, f, localdims, tolerance=1e-10)

    tci1 = tci1_from_tci2(tci2, f)
    assert tci1.rank() == tci2.rank()

    for idx in itertools.product(*[range(d) for d in localdims]):
        v = list(idx)
        assert np.isclose(tci1.evaluate(v), tci2.evaluate(v), atol=1e-8)

    tci2b = tci2_from_tci1(tci1)
    for idx in itertools.product(*[range(d) for d in localdims]):
        v = list(idx)
        assert np.isclose(tci2b.evaluate(v), f(v), atol=1e-8)


def test_tt_tci2_roundtrip():
    def f(v):
        return (1.0 + 2.0j) / (sum(x ** 2 for x in v) + 1)

    tci, ranks, errors = crossinterpolate2(np.complex128, f, [4] * 4, tolerance=1e-14, maxbonddim=5)
    tt = TensorTrain.from_tt_like(tci)
    tcib = tci2_from_tensortrain(tt, tolerance=1e-14)

    assert tt.rank() == 5
    assert tt.linkdims() == tci.linkdims()
    assert tt.sitedims() == [[4]] * 4

    assert tcib.rank() == 5
    assert tcib.linkdims() == tt.linkdims()
    assert tcib.sitedims() == [[4]] * 4

    for v in itertools.product(range(4), repeat=4):
        v = list(v)
        assert abs(tt.evaluate(v) - tci.evaluate(v)) < 1e-13
        assert abs(tcib.evaluate(v) - tci.evaluate(v)) < 1e-13

    optimize(tcib, f, tolerance=1e-14)
    for v in itertools.product(range(4), repeat=4):
        v = list(v)
        assert abs(tcib.evaluate(v) - f(v)) < 1e-13
