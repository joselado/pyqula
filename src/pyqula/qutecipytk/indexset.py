"""Port of indexset.jl.

``IndexSet`` is a bijective index<->position container: a list (``fromint``)
plus a dict (``toint``) for O(1) reverse lookup. Positions are 0-based here
(Julia's ``fromint``/``toint`` are 1-based).
"""
from __future__ import annotations

from typing import Generic, Iterator, Sequence, TypeVar

T = TypeVar("T")


class IndexSet(Generic[T]):
    __slots__ = ("toint", "fromint")

    def __init__(self, items: Sequence[T] | None = None):
        if items is None:
            self.fromint: list[T] = []
            self.toint: dict[T, int] = {}
        else:
            self.fromint = list(items)
            self.toint = {x: i for i, x in enumerate(self.fromint)}

    def __getitem__(self, i: int) -> T:
        return self.fromint[i]

    def __setitem__(self, i: int, x: T) -> None:
        old = self.fromint[i]
        del self.toint[old]
        self.fromint[i] = x
        self.toint[x] = i

    def __iter__(self) -> Iterator[T]:
        return iter(self.fromint)

    def __len__(self) -> int:
        return len(self.fromint)

    def __bool__(self) -> bool:
        return len(self.fromint) > 0

    def __eq__(self, other) -> bool:
        if not isinstance(other, IndexSet):
            return NotImplemented
        return self.fromint == other.fromint

    def append(self, x: T) -> None:
        self.fromint.append(x)
        self.toint[x] = len(self.fromint) - 1

    def pos(self, indices):
        if isinstance(indices, list):
            return [self.toint[i] for i in indices]
        return self.toint[indices]


def isnested(a: Sequence[T], b: Sequence[T], row_or_col: str = "row") -> bool:
    """Whether every element of b, with its last (row) or first (col)
    component dropped, is contained in the set a."""
    aset = set(a)
    for b_ in b:
        if len(b_) == 0:
            return False
        if row_or_col == "row" and tuple(b_[:-1]) not in aset:
            return False
        if row_or_col == "col" and tuple(b_[1:]) not in aset:
            return False
    return True
