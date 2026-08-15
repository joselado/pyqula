import itertools

import numpy as np

from pyqula.qutecipytk.tensortrain.batcheval import BatchEvaluator, batchevaluate_dispatch, isbatchevaluable
from pyqula.qutecipytk.tensortrain.cachedfunction import CachedFunction


def test_cache():
    for dtype in [np.float64, np.complex128]:
        def f(x):
            return 2 * x[0] + x[1]

        cf = CachedFunction(dtype, f, [4, 2])
        assert cf.f is f
        for i in range(4):
            for j in range(2):
                x = [i, j]
                assert cf(x) == f(x)
                assert tuple(x) in cf.cache
                assert cf(x) == f(x)


class _TestFunction(BatchEvaluator):
    def __init__(self, dtype, localdims):
        self.dtype = dtype
        self.localdims = localdims

    def __call__(self, indexset):
        return sum(indexset)

    def batchevaluate(self, leftindexset, rightindexset, ncent):
        nl = len(leftindexset[0])
        center_dims = self.localdims[nl:nl + ncent]
        combos = list(itertools.product(*[range(d) for d in center_dims]))
        result = np.empty((len(leftindexset), len(combos), len(rightindexset)), dtype=self.dtype)
        for i, l in enumerate(leftindexset):
            for c, combo in enumerate(combos):
                for j, r in enumerate(rightindexset):
                    result[i, c, j] = sum(l) + sum(combo) + sum(r)
        return result.reshape((len(leftindexset), *center_dims, len(rightindexset)))


def test_cache_batcheval():
    for dtype in [np.float64, np.complex128]:
        localdims = [2, 2, 2, 2, 2]
        leftindexset = [[0, 0] for _ in range(100)]
        rightindexset = [[0, 0] for _ in range(100)]

        f = CachedFunction(dtype, _TestFunction(dtype, localdims), localdims)
        assert isbatchevaluable(f)
        result = batchevaluate_dispatch(dtype, f, localdims, leftindexset, rightindexset, 1)
        ref = np.array([
            [[sum(l) + c + sum(r) for r in rightindexset] for c in range(localdims[2])]
            for l in leftindexset
        ])
        assert np.allclose(result, ref)


def test_encode_and_decode_cachekey():
    localdims = [2, 3, 4]
    cf = CachedFunction(np.complex128, lambda x: float(sum(x)), localdims)
    for i1 in range(localdims[0]):
        for i2 in range(localdims[1]):
            for i3 in range(localdims[2]):
                x = [i1, i2, i3]
                cf(x)
                key = tuple(x)
                assert key in cf.cache

    cachedata = cf.cachedata()
    for x, v in cachedata.items():
        assert cf(list(x)) == v
