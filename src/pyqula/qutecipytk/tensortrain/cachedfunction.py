"""Port of cachedfunction.jl: CachedFunction, memoizing an arbitrary function.

Per CLAUDE.md, the cache is keyed directly by ``tuple(indexset)`` rather than
Julia's mixed-radix integer encoding (`BitIntegers`/`BigInt`) -- simpler, no
overflow-tracking machinery needed (Python's native ``int`` is
arbitrary-precision anyway, so the overflow concern that motivated
`BitIntegers` in Julia doesn't apply here even if we wanted the integer
encoding), and dict-of-tuple lookups are fast enough.

Julia's ``get!(dict, key) do ... end`` compute-if-absent idiom becomes an
explicit ``if key not in cache: cache[key] = ...`` block.
"""
from __future__ import annotations

import itertools
from typing import Callable, Sequence

import numpy as np

from pyqula.qutecipytk.tensortrain.batcheval import BatchEvaluator, isbatchevaluable


class CachedFunction(BatchEvaluator):
    def __init__(self, dtype, f: Callable, localdims: Sequence[int], cache: dict | None = None):
        self.dtype = dtype
        self.f = f
        self.localdims = list(localdims)
        self.cache: dict[tuple, object] = cache if cache is not None else {}

    def __call__(self, x: Sequence[int]):
        key = tuple(x)
        if len(key) != len(self.localdims):
            raise ValueError("Invalid length of x")
        if key not in self.cache:
            self.cache[key] = self.f(list(key))
        return self.cache[key]

    def __setitem__(self, indexset: Sequence[int], val) -> None:
        self.cache[tuple(indexset)] = val

    def __contains__(self, x: Sequence[int]) -> bool:
        return tuple(x) in self.cache

    def cachedata(self) -> dict[tuple, object]:
        return dict(self.cache)

    def batchevaluate(self, leftindexset: Sequence, rightindexset: Sequence, ncent: int) -> np.ndarray:
        if len(leftindexset) * len(rightindexset) == 0:
            return np.empty((0,) * (ncent + 2), dtype=self.dtype)
        if isbatchevaluable(self.f):
            return self._batcheval_for_batchevaluator(leftindexset, rightindexset, ncent)
        return self._batcheval_default(leftindexset, rightindexset, ncent)

    def _center_combos(self, nl: int, ncent: int):
        center_dims = self.localdims[nl:nl + ncent]
        combos = list(itertools.product(*[range(d) for d in center_dims])) if ncent > 0 else [()]
        return center_dims, combos

    def _batcheval_default(self, leftindexset, rightindexset, ncent) -> np.ndarray:
        nl = len(leftindexset[0])
        center_dims, combos = self._center_combos(nl, ncent)
        lefts = [tuple(l) for l in leftindexset]
        rights = [tuple(r) for r in rightindexset]

        def get(key):
            if key not in self.cache:
                self.cache[key] = self.f(list(key))
            return self.cache[key]

        # Flat comprehension instead of per-cell numpy __setitem__ inside a triple loop --
        # cache population is order-independent, so any traversal order is equivalent; this
        # one matches the (i,c,j) axis order used for the final reshape.
        flat = [get(l + k + r) for l in lefts for k in combos for r in rights]
        result = np.array(flat, dtype=self.dtype).reshape(len(lefts), len(combos), len(rights))
        return result.reshape((len(leftindexset), *center_dims, len(rightindexset)))

    def _batcheval_for_batchevaluator(self, leftindexset, rightindexset, ncent) -> np.ndarray:
        nl = len(leftindexset[0])
        center_dims, combos = self._center_combos(nl, ncent)
        result = np.empty((len(leftindexset), len(combos), len(rightindexset)), dtype=self.dtype)
        filled = np.zeros(result.shape, dtype=bool)

        for j, rightindex in enumerate(rightindexset):
            for c, k in enumerate(combos):
                for i, leftindex in enumerate(leftindexset):
                    key = tuple(leftindex) + k + tuple(rightindex)
                    if key in self.cache:
                        result[i, c, j] = self.cache[key]
                        filled[i, c, j] = True

        left_needs = [i for i in range(len(leftindexset)) if not filled[i, :, :].all()]
        right_needs = [j for j in range(len(rightindexset)) if not filled[:, :, j].all()]
        if left_needs and right_needs:
            leftindexset_ = [leftindexset[i] for i in left_needs]
            rightindexset_ = [rightindexset[j] for j in right_needs]
            result_ = self.f.batchevaluate(leftindexset_, rightindexset_, ncent)
            for jj, j in enumerate(right_needs):
                rightindex = rightindexset[j]
                for c, k in enumerate(combos):
                    for ii, i in enumerate(left_needs):
                        leftindex = leftindexset[i]
                        key = tuple(leftindex) + k + tuple(rightindex)
                        self.cache[key] = result_[(ii, *k, jj)]

        for j, rightindex in enumerate(rightindexset):
            for c, k in enumerate(combos):
                for i, leftindex in enumerate(leftindexset):
                    if filled[i, c, j]:
                        continue
                    key = tuple(leftindex) + k + tuple(rightindex)
                    result[i, c, j] = self.cache[key]

        return result.reshape((len(leftindexset), *center_dims, len(rightindexset)))
