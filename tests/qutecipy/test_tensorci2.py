import itertools

import numpy as np
import pytest

from pyqula.qutecipytk.globalpivot import AbstractGlobalPivotFinder
from pyqula.qutecipytk.tci2 import (TensorCI2, convergence_criterion, crossinterpolate2,
                          kronecker_left, kronecker_right)
from pyqula.qutecipytk.tensortrain.batcheval import isbatchevaluable
from pyqula.qutecipytk.tensortrain.core import TensorTrain


def _bits_to_x(bits, R):
    idx = 0
    for b in bits:
        idx = idx * 2 + b
    return idx / 2 ** R


def _x_to_bits(x, R):
    idx = int(round(x * 2 ** R))
    idx = max(0, min(2 ** R - 1, idx))
    return [(idx >> i) & 1 for i in range(R - 1, -1, -1)]


def test_kronecker():
    multiset = [tuple(range(5)) for _ in range(5)]
    localdim = 4
    localset = set(range(localdim))

    c = kronecker_left(multiset, localdim)
    for ci in c:
        assert ci[:5] == tuple(range(5))
        assert ci[5] in localset

    d = kronecker_right(localdim, multiset)
    for di in d:
        assert di[0] in localset
        assert di[1:6] == tuple(range(5))


def test_pivoterrors():
    diags = [1.0, 1e-5, 0.0]

    def f(x):
        return diags[x[0]] if x[0] == x[1] else 0.0

    localdims = [3, 3]
    tci, ranks, errors = crossinterpolate2(np.float64, f, localdims, [[0, 0]], tolerance=1e-8)
    assert np.allclose(tci.pivoterrors, diags)


def test_checkbatchevaluatable():
    def f(x):
        return 1.0

    L = 10
    localdims = [2] * L
    with pytest.raises(ValueError):
        crossinterpolate2(np.float64, f, localdims, [[0] * L], checkbatchevaluatable=True)


@pytest.mark.parametrize("pivotsearch", ["full", "rook"])
@pytest.mark.parametrize("strictlynested", [False, True])
def test_trivial_mps_exp(pivotsearch, strictlynested):
    R = 8
    abstol = 1e-4

    def f(bitlist):
        return np.exp(-_bits_to_x(bitlist, R))

    localdims = [2] * R
    firstpivots = [[0] * R, [0] + [1] * (R - 1)]
    tci, ranks, errors = crossinterpolate2(
        np.float64, f, localdims, firstpivots, tolerance=abstol, maxbonddim=1, maxiter=2,
        normalizeerror=False, nsearchglobalpivot=0, pivotsearch=pivotsearch, strictlynested=strictlynested,
    )

    assert all(d == 1 for d in tci.linkdims())

    tt = TensorTrain.from_tt_like(tci)
    for x in [0.1, 0.3, 0.6, 0.9]:
        indexset = _x_to_bits(x, R)
        assert abs(tci.evaluate(indexset) - f(indexset)) < abstol
        assert abs(tt.evaluate(indexset) - f(indexset)) < abstol


class _CustomGlobalPivotFinder(AbstractGlobalPivotFinder):
    def __init__(self, npivots):
        self.npivots = npivots

    def __call__(self, input, f, abstol, verbosity=0, rng=None):
        import random
        rng = rng if rng is not None else random
        L = len(input.localdims)
        return [tuple(rng.randrange(input.localdims[p]) for p in range(L)) for _ in range(self.npivots)]


def test_custom_global_pivot_finder():
    R = 8
    abstol = 1e-4

    def f(bitlist):
        return np.exp(-_bits_to_x(bitlist, R))

    localdims = [2] * R
    firstpivots = [[0] * R, [0] + [1] * (R - 1)]
    tci, ranks, errors = crossinterpolate2(
        np.float64, f, localdims, firstpivots, tolerance=abstol, maxbonddim=1, maxiter=2,
        normalizeerror=False, globalpivotfinder=_CustomGlobalPivotFinder(10), pivotsearch="full",
        strictlynested=False,
    )

    assert all(d == 1 for d in tci.linkdims())
    tt = TensorTrain.from_tt_like(tci)
    for x in [0.1, 0.3, 0.6, 0.9]:
        indexset = _x_to_bits(x, R)
        assert abs(tci.evaluate(indexset) - f(indexset)) < abstol
        assert abs(tt.evaluate(indexset) - f(indexset)) < abstol


def test_lorentz_mps():
    for coeff in [1.0, 0.5 - 1.0j]:
        dtype = np.complex128 if isinstance(coeff, complex) else np.float64
        n = 5

        def f(v, coeff=coeff):
            return coeff / (sum(x ** 2 for x in v) + 1)

        for pivotsearch in ["full", "rook"]:
            tci, ranks, errors = crossinterpolate2(
                dtype, f, [10] * n, [[0] * n], tolerance=1e-10, pivotsearch=pivotsearch
            )
            tt = TensorTrain.from_tt_like(tci)
            for v in itertools.product(range(4), repeat=n):
                assert np.isclose(tt.evaluate(list(v)), f(list(v)), atol=1e-8)


def test_sum_matches_true_value():
    def f(v):
        return 1.0 / (1.0 + sum(x ** 2 for x in v))

    localdims = [6] * 5
    tci, ranks, errors = crossinterpolate2(np.float64, f, localdims, tolerance=1e-10)
    tt = TensorTrain.from_tt_like(tci)

    brute = sum(f(list(idx)) for idx in itertools.product(*[range(d) for d in localdims]))
    assert np.isclose(tt.sum(), brute, rtol=1e-6)


def test_convergencecriterion():
    assert not convergence_criterion([1, 2], [1e-2, 1e-5], [0, 0], 1e-4, 4, 3)
    assert convergence_criterion([1, 2, 2, 2], [1e-2, 1e-5, 1e-5, 1e-5], [0, 0, 0, 0], 1e-4, 4, 3)
    assert not convergence_criterion([1, 2, 2, 2], [1e-2, 1e-2, 1e-5, 1e-5], [0, 0, 0, 0], 1e-4, 4, 3)
    assert convergence_criterion([1, 2, 2, 2], [1e-2, 1e-2, 1e-2, 1e-2], [0, 0, 0, 0], 1e-4, 2, 3)
    assert convergence_criterion([1, 2, 2, 2], [1e-2, 1e-2, 1e-2, 1e-2], [0, 1, 1, 1], 1e-4, 2, 3)


def test_isbatchevaluable_on_tci2():
    def f(v):
        return 1.0

    tci, _, _ = crossinterpolate2(np.float64, f, [2] * 5, tolerance=1e-8)
    assert isbatchevaluable(tci) is False  # TensorCI2 itself is not a BatchEvaluator, unlike TTCache
