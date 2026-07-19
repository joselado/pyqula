"""Port of QuanticsGrids.jl's src/grid_discretized.jl -> DiscretizedGrid.

A Float64 wrapper composing an internal InherentDiscreteGrid (always
origin=0, step=1 -- 0-based here, matching this port's global convention)
plus lower_bound/upper_bound/includeendpoint for the float affine map. All
quantics<->grididx logic delegates to the wrapped discrete grid; only
grididx_to_origcoord/origcoord_to_grididx are reimplemented for floats.

Numerical details preserved exactly (see CLAUDE.md): float64 throughout,
same arithmetic operation order as the Julia source so round-trip
(grididx -> origcoord -> grididx) exact-reproducibility holds; rounds to
nearest (Python's round()/banker's-rounding matches Julia's default
`round`); bounds-checks against the closed interval [lower_bound,
upper_bound] even though the grid itself is half-open; clamps the rounded
index into [0, base^R); includeendpoint=True rewrites the *stored*
upper_bound at construction time (a linspace(..., endpoint=True)-style
trick), not just the final step-size formula.
"""
from __future__ import annotations

from typing import Sequence

from pyqula.qutecipytk.quantics.grid import InherentDiscreteGrid, _build_indextable, _to_tuple


class DiscretizedGrid:
    def __init__(
        self, Rs: Sequence[int], lower_bound: Sequence[float], upper_bound: Sequence[float],
        variablenames: Sequence[str], base: Sequence[int], indextable: list[list[tuple[str, int]]],
        includeendpoint: Sequence[bool] = False,
    ):
        D = len(Rs)
        lower_bound = tuple(float(x) for x in _to_tuple(lower_bound, D))
        upper_bound = tuple(float(x) for x in _to_tuple(upper_bound, D))
        base = _to_tuple(base, D)
        includeendpoint = _to_tuple(includeendpoint, D)

        for d in range(D):
            if not (lower_bound[d] < upper_bound[d]):
                raise ValueError(
                    f"(lower_bound[{d}], upper_bound[{d}]) = "
                    f"({lower_bound[d]}, {upper_bound[d]}). lower_bound must be < upper_bound."
                )
            if Rs[d] == 0 and includeendpoint[d]:
                raise ValueError(f"Rs[{d}] = 0 and includeendpoint[{d}] = True is not allowed.")

        self.discretegrid = InherentDiscreteGrid(
            Rs, tuple([0] * D), tuple([1] * D), variablenames, base, indextable
        )
        self.lower_bound = lower_bound
        self.upper_bound = tuple(
            upper_bound[d] + (upper_bound[d] - lower_bound[d]) / (base[d] ** Rs[d] - 1)
            if includeendpoint[d] else upper_bound[d]
            for d in range(D)
        )
        self.includeendpoint = includeendpoint

    # -- constructors --------------------------------------------------------

    @classmethod
    def from_resolutions(
        cls, variablenames: Sequence[str], Rs: Sequence[int], *, lower_bound=None, upper_bound=None,
        base=2, unfoldingscheme: str = "fused", includeendpoint=False,
    ) -> "DiscretizedGrid":
        D = len(Rs)
        Rs = tuple(Rs)
        lower_bound = lower_bound if lower_bound is not None else (0.0,) * D
        upper_bound = upper_bound if upper_bound is not None else (1.0,) * D
        base = _to_tuple(base, D)
        indextable = _build_indextable(variablenames, Rs, unfoldingscheme)
        return cls(Rs, lower_bound, upper_bound, variablenames, base, indextable, includeendpoint)

    @classmethod
    def from_indextable(
        cls, variablenames: Sequence[str], indextable: list[list[tuple[str, int]]], *, lower_bound=None,
        upper_bound=None, base=2, includeendpoint=False,
    ) -> "DiscretizedGrid":
        D = len(variablenames)
        Rs = tuple(
            sum(1 for site in indextable for (vn, _) in site if vn == variablename) for variablename in variablenames
        )
        lower_bound = lower_bound if lower_bound is not None else (0.0,) * D
        upper_bound = upper_bound if upper_bound is not None else (1.0,) * D
        base = _to_tuple(base, D)
        return cls(Rs, lower_bound, upper_bound, variablenames, base, indextable, includeendpoint)

    # -- basic accessors -------------------------------------------------

    def ndims(self) -> int:
        return self.discretegrid.ndims()

    def __len__(self) -> int:
        return len(self.discretegrid)

    def grid_Rs(self) -> tuple[int, ...]:
        return self.discretegrid.Rs

    def grid_indextable(self) -> list[list[tuple[str, int]]]:
        return self.discretegrid.indextable

    def grid_bases(self) -> tuple[int, ...]:
        return self.discretegrid.base

    def grid_variablenames(self) -> tuple[str, ...]:
        return self.discretegrid.variablenames

    def sitedim(self, site: int) -> int:
        return self.discretegrid.sitedim(site)

    def localdimensions(self) -> list[int]:
        return self.discretegrid.localdimensions()

    # -- grid coordinate functions -------------------------------------------

    def grid_min(self) -> tuple[float, ...]:
        return self.lower_bound

    def grid_step(self) -> tuple[float, ...]:
        Rs = self.grid_Rs()
        base = self.grid_bases()
        return tuple(
            (self.upper_bound[d] - self.lower_bound[d]) / (base[d] ** Rs[d]) for d in range(self.ndims())
        )

    def grid_max(self) -> tuple[float, ...]:
        step = self.grid_step()
        return tuple(u - s for u, s in zip(self.upper_bound, step))

    def grid_origcoords(self, d: int) -> list[float]:
        if not (0 <= d < self.ndims()):
            raise IndexError(f"Dimension {d} out of bounds [0, {self.ndims()}).")
        start = self.grid_min()[d]
        stop = self.grid_max()[d]
        n = self.grid_bases()[d] ** self.grid_Rs()[d]
        if n == 1:
            return [start]
        return [start + i * (stop - start) / (n - 1) for i in range(n)]

    # -- core conversions --------------------------------------------------

    def quantics_to_grididx(self, quantics: Sequence[int]) -> tuple[int, ...]:
        return self.discretegrid.quantics_to_grididx(quantics)

    def grididx_to_quantics(self, grididx) -> list[int]:
        return self.discretegrid.grididx_to_quantics(grididx)

    def grididx_to_origcoord(self, index) -> tuple[float, ...]:
        D = self.ndims()
        index = _to_tuple(index, D)
        base = self.grid_bases()
        Rs = self.grid_Rs()
        for d in range(D):
            if not (0 <= index[d] < base[d] ** Rs[d]):
                raise ValueError(f"Grid index out of bounds [0, {base[d] ** Rs[d]}).")

        result = []
        for d in range(D):
            step_d = (self.upper_bound[d] - self.lower_bound[d]) / (base[d] ** Rs[d])
            result.append(self.lower_bound[d] + index[d] * step_d)
        return tuple(result)

    def origcoord_to_grididx(self, coordinate) -> tuple[int, ...]:
        D = self.ndims()
        coord = _to_tuple(coordinate, D)
        lower = self.lower_bound
        upper = self.upper_bound
        for d in range(D):
            if not (lower[d] <= coord[d] <= upper[d]):
                raise ValueError(f"Coordinate out of bounds [{lower[d]}, {upper[d]}] (dimension {d}).")

        steps = self.grid_step()
        base = self.grid_bases()
        Rs = self.grid_Rs()
        result = []
        for d in range(D):
            continuous_idx = (coord[d] - lower[d]) / steps[d]
            discrete_idx = round(continuous_idx)
            discrete_idx = max(0, min(base[d] ** Rs[d] - 1, discrete_idx))
            result.append(discrete_idx)
        return tuple(result)

    def origcoord_to_quantics(self, coordinate) -> list[int]:
        return self.grididx_to_quantics(self.origcoord_to_grididx(coordinate))

    def quantics_to_origcoord(self, quantics: Sequence[int]) -> tuple[float, ...]:
        return self.grididx_to_origcoord(self.quantics_to_grididx(quantics))


def quantics_function(dtype, g, f):
    """Wrap a coordinate-space function f into a quantics-space function
    q -> f(*quantics_to_origcoord(g, q)); the key adapter for feeding a
    quantics grid into TCI's crossinterpolate2."""
    def wrapped(quantics):
        coords = g.quantics_to_origcoord(quantics)
        return dtype(f(*coords))
    return wrapped
