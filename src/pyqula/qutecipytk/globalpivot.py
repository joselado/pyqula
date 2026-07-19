"""Port of globalpivotfinder.jl + globalsearch.jl.

Two genuinely different greedy search variants (per CLAUDE.md, kept separate
rather than merged): ``DefaultGlobalPivotFinder`` (random restarts, single
one-pass greedy coordinate scan, keep-best-if-above-margin) vs.
``_floatingzone``/``estimate_true_error`` (unconditional coordinate ascent to
a true local error maximum, used for diagnostics and by the older
``search_global_pivots``).
"""
from __future__ import annotations

import random as _random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from pyqula.qutecipytk.tensortrain.cache import TTCache
from pyqula.qutecipytk.tensortrain.core import TensorTrain


@dataclass
class GlobalPivotSearchInput:
    localdims: list[int]
    current_tt: TensorTrain
    maxsamplevalue: float
    Iset: list[list[tuple]]
    Jset: list[list[tuple]]


class AbstractGlobalPivotFinder(ABC):
    @abstractmethod
    def __call__(
        self, input: GlobalPivotSearchInput, f: Callable, abstol: float,
        verbosity: int = 0, rng: _random.Random | None = None,
    ) -> list[tuple]: ...


class DefaultGlobalPivotFinder(AbstractGlobalPivotFinder):
    def __init__(self, nsearch: int = 5, maxnglobalpivot: int = 5, tolmarginglobalsearch: float = 10.0):
        self.nsearch = nsearch
        self.maxnglobalpivot = maxnglobalpivot
        self.tolmarginglobalsearch = tolmarginglobalsearch

    def __call__(
        self, input: GlobalPivotSearchInput, f: Callable, abstol: float,
        verbosity: int = 0, rng: _random.Random | None = None,
    ) -> list[tuple]:
        rng = rng if rng is not None else _random
        L = len(input.localdims)

        initial_points = [[rng.randrange(input.localdims[p]) for p in range(L)] for _ in range(self.nsearch)]

        found_pivots: list[tuple] = []
        for point in initial_points:
            current_point = list(point)
            best_error = 0.0
            best_point = list(point)

            for p in range(L):
                for v in range(input.localdims[p]):
                    current_point[p] = v
                    error = abs(f(current_point) - input.current_tt(current_point))
                    if error > best_error:
                        best_error = error
                        best_point = list(current_point)
                current_point[p] = point[p]

            if best_error > abstol * self.tolmarginglobalsearch:
                found_pivots.append(tuple(best_point))

        if len(found_pivots) > self.maxnglobalpivot:
            found_pivots = found_pivots[: self.maxnglobalpivot]

        if verbosity > 0:
            print(f"Found {len(found_pivots)} global pivots")

        return found_pivots


def _floatingzone(
    ttcache: TTCache, f: Callable, earlystoptol: float = float("inf"), nsweeps: int = 10**9,
    initp: Sequence[int] | None = None,
) -> tuple[tuple, float]:
    if nsweeps <= 0:
        raise ValueError("nsweeps should be positive!")

    localdims = [d[0] for d in ttcache.sitedims()]
    n = len(ttcache)

    pivot = list(initp) if initp is not None else [_random.randrange(d) for d in localdims]

    dtype = ttcache.dtype
    maxerror = abs(f(pivot) - ttcache(pivot))

    for _ in range(nsweeps):
        prev_maxerror = maxerror
        for ipos in range(n):
            from pyqula.qutecipytk.tci2 import filltensor  # local import: avoids a tci2<->globalpivot import cycle

            left = [tuple(pivot[:ipos])]
            right = [tuple(pivot[ipos + 1:])]
            exactdata = np.asarray(filltensor(dtype, f, localdims, left, right, 1))
            prediction = np.asarray(filltensor(dtype, ttcache, localdims, left, right, 1))
            err = np.abs(exactdata - prediction).reshape(-1)
            pivot[ipos] = int(np.argmax(err))
            maxerror = max(float(np.max(err)), maxerror)

        if maxerror == prev_maxerror or maxerror > earlystoptol:
            break

    return tuple(pivot), maxerror


def estimate_true_error(
    tt: TensorTrain, f: Callable, nsearch: int = 100, initialpoints: Sequence[Sequence[int]] | None = None,
) -> list[tuple[tuple, float]]:
    if nsearch <= 0 and initialpoints is None:
        raise ValueError("No search is performed")
    if nsearch < 0:
        raise ValueError("nsearch must be non-negative")

    if nsearch > 0 and initialpoints is None:
        initialpoints = [[_random.randrange(d[0]) for d in tt.sitedims()] for _ in range(nsearch)]

    ttcache = TTCache.from_tt(tt)
    pivoterror = [_floatingzone(ttcache, f, initp=initp) for initp in initialpoints]
    pivoterror.sort(key=lambda pe: pe[1], reverse=True)

    seen = set()
    result = []
    for p, e in pivoterror:
        if p not in seen:
            seen.add(p)
            result.append((p, e))
    return result
