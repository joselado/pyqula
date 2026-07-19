"""Port of batcheval.jl: generic batch-evaluation adapters and dispatch.

Julia's dispatch-on-type-of-f (`_batchevaluate_dispatch` overloads) becomes
an explicit `isinstance(f, BatchEvaluator)` check.
"""
from __future__ import annotations

import itertools
from abc import ABC, abstractmethod
from typing import Callable, Sequence

import numpy as np


class BatchEvaluator(ABC):
    """Marker interface: an object that can evaluate a whole batch of
    (left-index x center-index x right-index) combinations at once, faster
    than looping scalar calls (e.g. TTCache, CachedFunction, Contraction)."""

    @abstractmethod
    def batchevaluate(self, leftindexset: Sequence, rightindexset: Sequence, ncent: int) -> np.ndarray: ...


def isbatchevaluable(f) -> bool:
    return isinstance(f, BatchEvaluator)


def _generic_batchevaluate(
    dtype, f: Callable, localdims: Sequence[int], leftindexset: Sequence, rightindexset: Sequence, M: int
) -> np.ndarray:
    """Fallback: evaluate f once per (left, center, right) combination.

    Builds the whole batch as a flat list comprehension (in the exact same
    i,c,j iteration order the equivalent triple loop would use) and casts to
    an array in one shot, rather than assigning into a numpy array element
    by element inside the innermost loop -- per-cell `result[i,c,j] = ...`
    numpy scalar __setitem__ dominates this function's own overhead (more
    than the cost of calling f itself, in profiling), so this is a pure
    library-overhead fix with no change to which values get computed.
    """
    if len(leftindexset) * len(rightindexset) == 0:
        return np.empty((0,) * (M + 2), dtype=dtype)

    nl = len(leftindexset[0])
    nr = len(rightindexset[0])
    center_dims = list(localdims[nl:nl + M])
    center_combos = list(itertools.product(*[range(d) for d in center_dims])) if M > 0 else [()]

    lefts = [list(lindex) for lindex in leftindexset]
    centers = [list(cindex) for cindex in center_combos]
    rights = [list(rindex) for rindex in rightindexset]

    flat = [f(l + c + r) for l in lefts for c in centers for r in rights]
    result = np.array(flat, dtype=dtype).reshape(len(lefts), len(centers), len(rights))
    return result.reshape((len(leftindexset), *center_dims, len(rightindexset)))


def batchevaluate_dispatch(
    dtype, f, localdims: Sequence[int], leftindexset: Sequence, rightindexset: Sequence, M: int
) -> np.ndarray:
    if len(leftindexset) * len(rightindexset) == 0:
        return np.empty((0,) * (M + 2), dtype=dtype)
    if isinstance(f, BatchEvaluator):
        N = len(localdims)
        nl = len(leftindexset[0])
        nr = len(rightindexset[0])
        ncent = N - nl - nr
        return f.batchevaluate(leftindexset, rightindexset, ncent)
    return _generic_batchevaluate(dtype, f, localdims, leftindexset, rightindexset, M)


class BatchEvaluatorAdapter(BatchEvaluator):
    """Wraps any plain scalar function to support batch evaluation (via the
    generic loop-based fallback)."""

    def __init__(self, dtype, f: Callable, localdims: Sequence[int]):
        self.dtype = dtype
        self.f = f
        self.localdims = list(localdims)

    def __call__(self, indexset):
        return self.f(indexset)

    def batchevaluate(self, leftindexset, rightindexset, ncent: int) -> np.ndarray:
        return _generic_batchevaluate(self.dtype, self.f, self.localdims, leftindexset, rightindexset, ncent)


def make_batch_evaluatable(dtype, f: Callable, localdims: Sequence[int]) -> BatchEvaluatorAdapter:
    return BatchEvaluatorAdapter(dtype, f, localdims)


class ThreadedBatchEvaluator(BatchEvaluatorAdapter):
    """Julia's ThreadedBatchEvaluator parallelizes the batch loop with
    Threads.@threads. Ported as a sequential fallback for now (see
    CLAUDE.md: naive Python threading gives no benefit under the GIL unless
    the callback itself releases it) -- revisit with joblib/multiprocessing
    if profiling shows this loop is a bottleneck."""
