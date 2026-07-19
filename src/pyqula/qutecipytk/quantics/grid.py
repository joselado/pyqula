"""Port of QuanticsGrids.jl's src/grid.jl -> InherentDiscreteGrid.

0-based throughout (grid indices 0..base^R-1, quantics digit values
0..base-1, bitnumber 0..R-1 with bitnumber 0 = most significant digit) --
see CLAUDE.md's "QuanticsGrids" section for the full list of deliberate
deviations from the Julia original:
- always returns tuples (not the Julia scalar-vs-tuple-for-D=1 polymorphism)
- no `base^R <= typemax(Int)` overflow guard (Python ints are arbitrary precision)
- only the general mixed-radix conversion path (no base-2 bit-shift fast path)
- a handful of classmethod factories instead of one constructor with
  argument-type-sniffing overloads
"""
from __future__ import annotations

import math
from typing import Sequence


def _trunc_div(a: int, b: int) -> int:
    """Integer division truncating toward zero (matches Julia's `div`, b > 0)."""
    return -((-a) // b) if a < 0 else a // b


def _to_tuple(x, d: int) -> tuple:
    if isinstance(x, (tuple, list)) and len(x) == d:
        return tuple(x)
    return tuple(x for _ in range(d))


class InherentDiscreteGrid:
    def __init__(
        self, Rs: Sequence[int], origin: Sequence[int], step: Sequence[int],
        variablenames: Sequence[str], base: Sequence[int], indextable: list[list[tuple[str, int]]],
    ):
        D = len(Rs)
        origin = _to_tuple(origin, D)
        step = _to_tuple(step, D)
        base = _to_tuple(base, D)
        variablenames = tuple(variablenames)

        for d in range(D):
            if base[d] <= 1:
                raise ValueError(f"base[{d}] = {base[d]}. base must be at least 2.")
            if step[d] < 1:
                raise ValueError(f"step[{d}] = {step[d]}. step must be at least 1.")
        if len(set(variablenames)) != len(variablenames):
            raise ValueError(f"variablenames = {variablenames} must be unique.")

        self.Rs = tuple(Rs)
        self.origin = origin
        self.step = step
        self.variablenames = variablenames
        self.base = base
        self.indextable = [list(site) for site in indextable]

        self._lookup_table = _build_lookup_table(self.Rs, self.indextable, self.variablenames)
        self._maxgrididx = tuple(b ** R for b, R in zip(self.base, self.Rs))
        self._site_radices = _build_site_radices(self.indextable, self.variablenames, self.base)
        self._site_placevalues = _build_site_placevalues(self._site_radices)
        self._sitedims_list = _build_site_dims(self._site_radices)

    # -- constructors --------------------------------------------------------

    @classmethod
    def from_resolutions(
        cls, Rs, origin=None, *, unfoldingscheme: str = "fused", step=None, base=2, variablenames=None,
    ) -> "InherentDiscreteGrid":
        D = len(Rs) if isinstance(Rs, (tuple, list)) else 1
        Rs = _to_tuple(Rs, D)
        origin = _to_tuple(origin if origin is not None else (0,) * D, D)
        step = _to_tuple(step if step is not None else (1,) * D, D)
        base = _to_tuple(base, D)
        variablenames = tuple(variablenames) if variablenames is not None else tuple(str(i) for i in range(D))
        indextable = _build_indextable(variablenames, Rs, unfoldingscheme)
        return cls(Rs, origin, step, variablenames, base, indextable)

    @classmethod
    def from_indextable(
        cls, variablenames: Sequence[str], indextable: list[list[tuple[str, int]]], *, origin=None, step=None,
        base=2,
    ) -> "InherentDiscreteGrid":
        D = len(variablenames)
        base = _to_tuple(base, D)
        Rs = tuple(
            sum(1 for site in indextable for (vn, _) in site if vn == variablename) for variablename in variablenames
        )
        origin = _to_tuple(origin if origin is not None else (0,) * D, D)
        step = _to_tuple(step if step is not None else (1,) * D, D)
        return cls(Rs, origin, step, variablenames, base, indextable)

    # -- basic accessors -------------------------------------------------

    def ndims(self) -> int:
        return len(self.Rs)

    def __len__(self) -> int:
        return len(self.indextable)

    def sitedim(self, site: int) -> int:
        if not (0 <= site < len(self.indextable)):
            raise IndexError(f"Site index {site} out of bounds [0, {len(self.indextable)}).")
        return self._sitedims_list[site]

    def localdimensions(self) -> list[int]:
        return list(self._sitedims_list)

    def grid_min(self) -> tuple[int, ...]:
        return self.origin

    def grid_max(self) -> tuple[int, ...]:
        return tuple(o + s * (b ** R - 1) for o, s, b, R in zip(self.origin, self.step, self.base, self.Rs))

    # -- core conversions --------------------------------------------------

    def quantics_to_grididx(self, quantics: Sequence[int]) -> tuple[int, ...]:
        if len(quantics) != len(self):
            raise ValueError(f"Quantics vector must have length {len(self)}, got {len(quantics)}.")
        for site, q in enumerate(quantics):
            if not (0 <= q < self.sitedim(site)):
                raise ValueError(f"Quantics value for site {site} out of range [0, {self.sitedim(site)}).")

        result = []
        for d in range(self.ndims()):
            grididx = 0
            R_d = self.Rs[d]
            for bitnumber in range(R_d):
                site_idx, pos_in_site = self._lookup_table[d][bitnumber]
                placevalue = self._site_placevalues[site_idx][pos_in_site]
                base_pos = self._site_radices[site_idx][pos_in_site]
                digit = (quantics[site_idx] // placevalue) % base_pos
                grididx += digit * self.base[d] ** (R_d - 1 - bitnumber)
            result.append(grididx)
        return tuple(result)

    def grididx_to_quantics(self, grididx) -> list[int]:
        grididx = _to_tuple(grididx, self.ndims())
        for d in range(self.ndims()):
            if not (0 <= grididx[d] < self._maxgrididx[d]):
                raise ValueError(f"Grid index out of bounds [0, {self._maxgrididx[d]}).")

        result = [0] * len(self.indextable)
        for d in range(self.ndims()):
            idx = grididx[d]
            R_d = self.Rs[d]
            base_d = self.base[d]
            for bitnumber in range(R_d):
                site_idx, pos_in_site = self._lookup_table[d][bitnumber]
                bit_position = R_d - 1 - bitnumber
                digit = (idx // (base_d ** bit_position)) % base_d
                placevalue = self._site_placevalues[site_idx][pos_in_site]
                result[site_idx] += digit * placevalue
        return result

    def grididx_to_origcoord(self, grididx) -> tuple[int, ...]:
        grididx = _to_tuple(grididx, self.ndims())
        for d in range(self.ndims()):
            if not (0 <= grididx[d] < self._maxgrididx[d]):
                raise ValueError(f"Grid index out of bounds [0, {self._maxgrididx[d]}).")
        return tuple(o + g * s for o, g, s in zip(self.origin, grididx, self.step))

    def origcoord_to_grididx(self, coordinate) -> tuple[int, ...]:
        coord = _to_tuple(coordinate, self.ndims())
        lower = self.grid_min()
        upper = self.grid_max()
        for d in range(self.ndims()):
            if not (lower[d] <= coord[d] <= upper[d]):
                raise ValueError(f"Coordinate out of bounds [{lower[d]}, {upper[d]}].")
        result = []
        for d in range(self.ndims()):
            idx = _trunc_div(coord[d] - lower[d], self.step[d])
            idx = max(0, min(self.base[d] ** self.Rs[d] - 1, idx))
            result.append(idx)
        return tuple(result)

    def origcoord_to_quantics(self, coordinate) -> list[int]:
        return self.grididx_to_quantics(self.origcoord_to_grididx(coordinate))

    def quantics_to_origcoord(self, quantics: Sequence[int]) -> tuple[int, ...]:
        return self.grididx_to_origcoord(self.quantics_to_grididx(quantics))


def _to_tuple_local(x, d):
    return _to_tuple(x, d)


def _build_lookup_table(Rs, indextable, variablenames) -> list[list[tuple[int, int]]]:
    D = len(Rs)
    for d in range(D):
        if Rs[d] < 0:
            raise ValueError(f"Rs[{d}] = {Rs[d]}. Rs must be non-negative.")
    lookup_table = [[None] * Rs[d] for d in range(D)]
    var_index = {v: i for i, v in enumerate(variablenames)}
    visited = [[False] * Rs[d] for d in range(D)]

    for site_idx, site in enumerate(indextable):
        for pos_in_site, (variablename, bitnumber) in enumerate(site):
            if variablename not in var_index:
                raise ValueError(f"Index table contains unknown variable {variablename!r}.")
            var_idx = var_index[variablename]
            if not (0 <= bitnumber < Rs[var_idx]):
                raise ValueError(
                    f"Index table contains quantics bitnumber {bitnumber} of variable {variablename!r}, "
                    f"must be in [0, {Rs[var_idx]})."
                )
            if visited[var_idx][bitnumber]:
                raise ValueError(f"Index table contains quantics bitnumber {bitnumber} of {variablename!r} twice.")
            lookup_table[var_idx][bitnumber] = (site_idx, pos_in_site)
            visited[var_idx][bitnumber] = True

    for var_idx, v in enumerate(visited):
        if not all(v):
            missing = v.index(False)
            raise ValueError(f"Index table contains no site for bitnumber {missing} of {variablenames[var_idx]!r}.")

    return lookup_table


def _build_site_radices(indextable, variablenames, base) -> list[list[int]]:
    var_index = {v: i for i, v in enumerate(variablenames)}
    return [[base[var_index[vn]] for vn, _ in site] for site in indextable]


def _build_site_placevalues(site_radices: list[list[int]]) -> list[list[int]]:
    result = []
    for radices in site_radices:
        placevalues = [1] * len(radices)
        mult = 1
        for pos in range(len(radices) - 1, -1, -1):
            placevalues[pos] = mult
            mult *= radices[pos]
        result.append(placevalues)
    return result


def _build_site_dims(site_radices: list[list[int]]) -> list[int]:
    return [math.prod(radices) if radices else 1 for radices in site_radices]


def _build_indextable(variablenames, Rs, unfoldingscheme: str) -> list[list[tuple[str, int]]]:
    if unfoldingscheme not in ("interleaved", "fused", "grouped"):
        raise ValueError(f"unfoldingscheme = {unfoldingscheme!r}. Supported: interleaved, fused, grouped.")

    D = len(variablenames)
    indextable: list[list[tuple[str, int]]] = []

    if unfoldingscheme == "grouped":
        for d in range(D):
            for bitnumber in range(Rs[d]):
                indextable.append([(variablenames[d], bitnumber)])
    else:
        maxR = max(Rs) if Rs else 0
        for bitnumber in range(maxR):
            if unfoldingscheme == "interleaved":
                for d in range(D):
                    if bitnumber < Rs[d]:
                        indextable.append([(variablenames[d], bitnumber)])
            else:  # fused: dimensions in reverse order (first dim varies fastest)
                site = []
                for d in range(D - 1, -1, -1):
                    if bitnumber < Rs[d]:
                        site.append((variablenames[d], bitnumber))
                if site:
                    indextable.append(site)

    return indextable
