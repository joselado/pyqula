"""Port of util.jl + sweepstrategies.jl.

Indexing convention: this whole package is 0-based internally (see CLAUDE.md),
unlike the 1-based Julia original. Pivot values (``MultiIndex`` entries) range
over ``0 .. localdim-1`` instead of ``1 .. localdim``.
"""
from __future__ import annotations

import random
from typing import Callable, Iterable, Sequence, TypeVar

import numpy as np

MultiIndex = tuple[int, ...]

T = TypeVar("T")


def maxabs(maxval: float, updates) -> float:
    """Running max of |x| over an array, seeded with maxval."""
    arr = np.asarray(updates)
    if arr.size == 0:
        return maxval
    return max(maxval, float(np.max(np.abs(arr))))


def pushunique(collection: list[T], *items: T) -> None:
    """Append each item to collection only if not already present."""
    for item in items:
        if item not in collection:
            collection.append(item)


def isconstant(collection: Sequence) -> bool:
    if len(collection) == 0:
        return True
    c = collection[0]
    return all(x == c for x in collection)


def randomsubset(items: Sequence[T], n: int) -> list[T]:
    """Draw n unique random elements from items without replacement."""
    n = min(n, len(items))
    if n <= 0:
        return []
    return random.sample(list(items), n)


def pushrandomsubset(subset: list[T], items: Iterable[T], n: int) -> None:
    """Extend subset in place with n random elements from items \\ subset."""
    remaining = [x for x in items if x not in subset]
    subset.extend(randomsubset(remaining, n))


def optfirstpivot(
    f: Callable[[list[int]], float],
    localdims: Sequence[int],
    firstpivot: Sequence[int] | None = None,
    maxsweep: int = 1000,
) -> list[int]:
    """Greedy coordinate-ascent search for a good starting pivot.

    0-based: each coordinate i is searched over range(localdims[i]).
    """
    n = len(localdims)
    pivot = list(firstpivot) if firstpivot is not None else [0] * n
    valf = abs(f(pivot))

    for _ in range(maxsweep):
        valf_prev = valf
        for i in range(n):
            for d in range(localdims[i]):
                bak = pivot[i]
                pivot[i] = d
                newval = abs(f(pivot))
                if newval > valf:
                    valf = newval
                else:
                    pivot[i] = bak
        if valf_prev == valf:
            break

    return pivot


def projector_to_slice(p: Sequence[int]) -> tuple[list, list[int]]:
    """Build slice objects + shape for a "projected" (partially fixed) index.

    A 0 entry means "keep as a full slice"; a nonzero entry (using the same
    0-based convention, so we reserve None instead of 0 to mean "free") is
    fixed to that value.

    To keep this unambiguous under 0-based indexing, ``None`` marks a free
    (unfixed) axis instead of Julia's ``0`` sentinel.
    """
    slices = [slice(None) if x is None else x for x in p]
    shape = [slice(None) if x is None else 1 for x in p]
    return slices, shape


def forwardsweep(sweepstrategy: str, iteration: int) -> bool:
    return sweepstrategy == "forward" or (
        sweepstrategy == "backandforth" and iteration % 2 == 1
    )
