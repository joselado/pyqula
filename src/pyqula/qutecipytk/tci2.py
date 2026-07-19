"""Port of tensorci2.jl: the improved TCI2 algorithm.

0-based indexing throughout. Unlike TCI1's IndexSet-based bookkeeping,
Iset/Jset here are plain Python lists of tuples with "array position == site
index" convention uniformly (no bond-array offset subtlety like TCI1 had).
"""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

from pyqula.qutecipytk.globalpivot import DefaultGlobalPivotFinder, GlobalPivotSearchInput, _floatingzone
from pyqula.qutecipytk.matrix.luci import MatrixLUCI
from pyqula.qutecipytk.tensortrain.base import AbstractTensorTrain
from pyqula.qutecipytk.tensortrain.batcheval import batchevaluate_dispatch, isbatchevaluable
from pyqula.qutecipytk.tensortrain.cache import TTCache
from pyqula.qutecipytk.tensortrain.core import TensorTrain
from pyqula.qutecipytk.util import forwardsweep, maxabs, pushunique

_UNBOUNDED_RANK = 2**62


def kronecker_left(Iset: Sequence[tuple], localdim: int) -> list[tuple]:
    return [tuple(i) + (j,) for i in Iset for j in range(localdim)]


def kronecker_right(localdim: int, Jset: Sequence[tuple]) -> list[tuple]:
    return [(i,) + tuple(j) for i in range(localdim) for j in Jset]


def filltensor(dtype, f, localdims: Sequence[int], Iset: Sequence, Jset: Sequence, M: int) -> np.ndarray:
    if len(Iset) * len(Jset) == 0:
        return np.empty((0,) * (M + 2), dtype=dtype)
    N = len(localdims)
    nl = len(Iset[0])
    nr = len(Jset[0])
    ncent = N - nl - nr
    if M != ncent:
        raise ValueError("Invalid number of central indices")
    expected_size = (len(Iset), *localdims[nl:nl + ncent], len(Jset))
    result = batchevaluate_dispatch(dtype, f, localdims, Iset, Jset, ncent)
    return result.reshape(expected_size)


def _union_preserve(a: Sequence, b: Sequence) -> list:
    seen = set()
    result = []
    for x in list(a) + list(b):
        if x not in seen:
            seen.add(x)
            result.append(x)
    return result


def reconstruct_global_pivots_from_ijset(localdims, Isets, Jsets) -> list[tuple]:
    pivots: list[tuple] = []
    for i in range(len(Isets)):
        for iset in Isets[i]:
            for jset in Jsets[i]:
                for j in range(localdims[i]):
                    pushunique(pivots, tuple(iset) + (j,) + tuple(jset))
    return pivots


class SubMatrix:
    """Lazy submatrix evaluator for rook pivoting: wraps f + fixed row/col
    label lists, evaluating arbitrary (irow, icol) subsets on demand."""

    def __init__(self, f: Callable, rows: Sequence[tuple], cols: Sequence[tuple]):
        self.f = f
        self.rows = rows
        self.cols = cols
        self.maxsamplevalue = 0.0

    def __call__(self, irows: Sequence[int], icols: Sequence[int]) -> np.ndarray:
        if isbatchevaluable(self.f):
            Iset = [self.rows[i] for i in irows]
            Jset = [self.cols[j] for j in icols]
            res = np.asarray(self.f.batchevaluate(Iset, Jset, 0)).reshape(len(irows), len(icols))
        else:
            # Precompute each row's/col's list() once instead of re-building it on every
            # inner-loop iteration (list(self.rows[i]) only depends on i, not j).
            rowlists = [list(self.rows[i]) for i in irows]
            collists = [list(self.cols[j]) for j in icols]
            res = np.array([[self.f(rl + cl) for cl in collists] for rl in rowlists])
        if res.size:
            self.maxsamplevalue = max(self.maxsamplevalue, float(np.max(np.abs(res))))
        return res


class TensorCI2(AbstractTensorTrain):
    def __init__(self, dtype, localdims: Sequence[int]):
        if len(localdims) <= 1:
            raise ValueError("localdims should have at least 2 elements!")
        n = len(localdims)
        self.dtype = dtype
        self.localdims = list(localdims)
        self.Iset: list[list[tuple]] = [[] for _ in range(n)]
        self.Jset: list[list[tuple]] = [[] for _ in range(n)]
        self._sitetensors = [np.zeros((0, d, 0), dtype=dtype) for d in localdims]
        self.pivoterrors: list[float] = []
        self.bonderrors = [0.0] * (n - 1)
        self.maxsamplevalue = 0.0
        self.Iset_history: list[list[list[tuple]]] = []
        self.Jset_history: list[list[list[tuple]]] = []

    @classmethod
    def from_function(
        cls, dtype, func: Callable, localdims: Sequence[int], initialpivots: Sequence[Sequence[int]] | None = None
    ) -> "TensorCI2":
        if initialpivots is None:
            initialpivots = [tuple([0] * len(localdims))]
        initialpivots = [tuple(p) for p in initialpivots]
        tci = cls(dtype, localdims)
        tci.add_global_pivots(initialpivots)
        tci.maxsamplevalue = max(abs(func(list(x))) for x in initialpivots)
        if abs(tci.maxsamplevalue) == 0.0:
            raise ValueError("maxsamplevalue is zero!")
        tci.invalidate_sitetensors()
        return tci

    @classmethod
    def from_index_sets(cls, dtype, func: Callable, localdims: Sequence[int], Iset, Jset) -> "TensorCI2":
        tci = cls(dtype, localdims)
        tci.Iset = [list(s) for s in Iset]
        tci.Jset = [list(s) for s in Jset]
        pivots = reconstruct_global_pivots_from_ijset(localdims, tci.Iset, tci.Jset)
        tci.maxsamplevalue = max(abs(func(list(x))) for x in pivots)
        if abs(tci.maxsamplevalue) == 0.0:
            raise ValueError("maxsamplevalue is zero!")
        tci.invalidate_sitetensors()
        return tci

    def sitetensors(self) -> list[np.ndarray]:
        return self._sitetensors

    def __len__(self) -> int:
        return len(self.localdims)

    def linkdims(self) -> list[int]:
        return [len(self.Iset[b + 1]) for b in range(len(self) - 1)]

    def invalidate_sitetensors(self) -> None:
        for b in range(len(self)):
            self._sitetensors[b] = np.zeros((0, 0, 0), dtype=self.dtype)

    def sitetensors_available(self) -> bool:
        return all(t.size != 0 for t in self._sitetensors)

    def update_bond_error(self, b: int, error: float) -> None:
        self.bonderrors[b] = error

    def max_bond_error(self) -> float:
        return max(self.bonderrors)

    def update_pivot_error(self, errors: Sequence[float]) -> None:
        n = max(len(self.pivoterrors), len(errors))
        a = list(self.pivoterrors) + [0.0] * (n - len(self.pivoterrors))
        b = list(errors) + [0.0] * (n - len(errors))
        self.pivoterrors = [max(x, y) for x, y in zip(a, b)]

    def flush_pivot_error(self) -> None:
        self.pivoterrors = []

    def pivot_error(self) -> float:
        return self.max_bond_error()

    def update_errors(self, b: int, errors: Sequence[float]) -> None:
        self.update_bond_error(b, errors[-1])
        self.update_pivot_error(errors)

    def update_max_sample(self, samples) -> None:
        self.maxsamplevalue = maxabs(self.maxsamplevalue, samples)

    # -- global pivots ---------------------------------------------------

    def add_global_pivots(self, pivots: Sequence[Sequence[int]]) -> None:
        if any(len(self) != len(p) for p in pivots):
            raise ValueError("Please specify a pivot as one index per leg of the MPS.")
        for pivot in pivots:
            for b in range(len(self)):
                pushunique(self.Iset[b], tuple(pivot[:b]))
                pushunique(self.Jset[b], tuple(pivot[b + 1:]))
        if len(pivots) > 0:
            self.invalidate_sitetensors()

    def add_global_pivots_1site_sweep(
        self, f: Callable, pivots: Sequence[Sequence[int]], reltol: float = 1e-14, abstol: float = 0.0,
        maxbonddim: int | None = None,
    ) -> None:
        self.add_global_pivots(pivots)
        self.make_canonical(f, reltol=reltol, abstol=abstol, maxbonddim=maxbonddim)

    def exist_as_pivot(self, indexset: Sequence[int]) -> list[bool]:
        return [
            tuple(indexset[:b]) in set(self.Iset[b]) and tuple(indexset[b + 1:]) in set(self.Jset[b])
            for b in range(len(self))
        ]

    def add_global_pivots_2site_sweep(
        self, f: Callable, pivots: Sequence[Sequence[int]], tolerance: float = 1e-8, normalizeerror: bool = True,
        maxbonddim: int | None = None, pivotsearch: str = "full", verbosity: int = 0, ntry: int = 10,
        strictlynested: bool = False,
    ) -> int:
        if any(len(self) != len(p) for p in pivots):
            raise ValueError("Please specify a pivot as one index per leg of the MPS.")
        pivots = [tuple(p) for p in pivots]
        pivots_ = pivots

        for _ in range(ntry):
            errornormalization = self.maxsamplevalue if normalizeerror else 1.0
            abstol = tolerance * errornormalization
            self.add_global_pivots(pivots_)
            self.sweep2site(
                f, 2, abstol=abstol, maxbonddim=maxbonddim, pivotsearch=pivotsearch,
                strictlynested=strictlynested, verbosity=verbosity,
            )
            newpivots = [p for p in pivots if abs(self.evaluate(list(p)) - f(list(p))) > abstol]
            if verbosity > 0:
                print(f"Trying to add {len(pivots_)} global pivots, {len(newpivots)} still remain.")
            if len(newpivots) == 0 or set(newpivots) == set(pivots_):
                return len(newpivots)
            pivots_ = newpivots
        return len(pivots_)

    # -- site tensors ------------------------------------------------------

    def set_sitetensor(self, b: int, T) -> np.ndarray:
        self._sitetensors[b] = np.asarray(T, dtype=self.dtype).reshape(
            len(self.Iset[b]), self.localdims[b], len(self.Jset[b])
        )
        return self._sitetensors[b]

    def compute_sitetensor(self, f: Callable, b: int, leftorthogonal: bool = True) -> np.ndarray:
        if not leftorthogonal:
            raise NotImplementedError("leftorthogonal=False is not supported!")

        Is = kronecker_left(self.Iset[b], self.localdims[b])
        Js = self.Jset[b]
        Pi1 = filltensor(self.dtype, f, self.localdims, self.Iset[b], self.Jset[b], 1).reshape(len(Is), len(Js))
        self.update_max_sample(Pi1)

        n = len(self)
        if b == n - 1:
            return self.set_sitetensor(b, Pi1)

        P = filltensor(self.dtype, f, self.localdims, self.Iset[b + 1], self.Jset[b], 0).reshape(
            len(self.Iset[b + 1]), len(self.Jset[b])
        )
        if len(self.Iset[b + 1]) != len(self.Jset[b]):
            raise ValueError(f"Pivot matrix at bond {b} is not square!")

        Tmat = np.linalg.solve(P.T, Pi1.T).T
        return self.set_sitetensor(b, Tmat.reshape(len(self.Iset[b]), self.localdims[b], len(self.Iset[b + 1])))

    def fill_sitetensors(self, f: Callable) -> None:
        for b in range(len(self)):
            self.compute_sitetensor(f, b)

    def sweep0site(self, f: Callable, b: int, reltol: float = 1e-14, abstol: float = 0.0) -> None:
        """AKA rmbadpivots!: shrink Iset[b+1]/Jset[b] to only pivots with a
        numerically significant diagonal, without adding new ones."""
        self.invalidate_sitetensors()
        P = filltensor(self.dtype, f, self.localdims, self.Iset[b + 1], self.Jset[b], 0).reshape(
            len(self.Iset[b + 1]), len(self.Jset[b])
        )
        self.update_max_sample(P)
        F = MatrixLUCI.from_matrix(P, reltol=reltol, abstol=abstol, leftorthogonal=True)

        diagU = F.lu.diag()
        ndiag = sum(
            1 for i in range(len(diagU))
            if abs(diagU[i]) > abstol and (len(diagU) > 0 and abs(diagU[i] / diagU[0]) > reltol)
        )
        rowidx = F.rowindices()[:ndiag]
        colidx = F.colindices()[:ndiag]
        self.Iset[b + 1] = [self.Iset[b + 1][i] for i in rowidx]
        self.Jset[b] = [self.Jset[b][j] for j in colidx]

    def sweep1site(
        self, f: Callable, sweepdirection: str = "forward", reltol: float = 1e-14, abstol: float = 0.0,
        maxbonddim: int | None = None, updatetensors: bool = True,
    ) -> None:
        self.flush_pivot_error()
        self.invalidate_sitetensors()
        if sweepdirection not in ("forward", "backward"):
            raise ValueError(f"Unknown sweep direction {sweepdirection}: choose 'forward' or 'backward'.")
        forward = sweepdirection == "forward"
        n = len(self)
        maxbonddim = maxbonddim if maxbonddim is not None else _UNBOUNDED_RANK
        bond_range = range(n - 1) if forward else range(n - 1, 0, -1)

        for b in bond_range:
            Is = kronecker_left(self.Iset[b], self.localdims[b]) if forward else list(self.Iset[b])
            Js = list(self.Jset[b]) if forward else kronecker_right(self.localdims[b], self.Jset[b])
            Pi = filltensor(self.dtype, f, self.localdims, self.Iset[b], self.Jset[b], 1).reshape(len(Is), len(Js))
            self.update_max_sample(Pi)
            luci = MatrixLUCI.from_matrix(
                Pi, reltol=reltol, abstol=abstol, maxrank=maxbonddim, leftorthogonal=forward
            )
            if forward:
                self.Iset[b + 1] = [Is[i] for i in luci.rowindices()]
                self.Jset[b] = [Js[j] for j in luci.colindices()]
            else:
                self.Iset[b] = [Is[i] for i in luci.rowindices()]
                self.Jset[b - 1] = [Js[j] for j in luci.colindices()]

            if updatetensors:
                self.set_sitetensor(b, luci.left() if forward else luci.right())
                if np.any(np.isnan(self._sitetensors[b])):
                    raise FloatingPointError(f"Error: NaN in tensor T[{b}]")

            errbond = b if forward else b - 1
            self.update_errors(errbond, luci.pivoterrors())

        if updatetensors:
            lastupdateindex = n - 1 if forward else 0
            localtensor = filltensor(
                self.dtype, f, self.localdims, self.Iset[lastupdateindex], self.Jset[lastupdateindex], 1
            )
            self.set_sitetensor(lastupdateindex, localtensor)

    def make_canonical(
        self, f: Callable, reltol: float = 1e-14, abstol: float = 0.0, maxbonddim: int | None = None
    ) -> None:
        self.sweep1site(f, "forward", reltol=0.0, abstol=0.0, maxbonddim=_UNBOUNDED_RANK, updatetensors=False)
        self.sweep1site(f, "backward", reltol=reltol, abstol=abstol, maxbonddim=maxbonddim, updatetensors=False)
        self.sweep1site(f, "forward", reltol=reltol, abstol=abstol, maxbonddim=maxbonddim, updatetensors=True)

    def _full_pivot_search(
        self, f: Callable, Icombined: list[tuple], Jcombined: list[tuple], reltol: float, abstol: float,
        maxbonddim: int, leftorthogonal: bool,
    ) -> MatrixLUCI:
        Pi = filltensor(self.dtype, f, self.localdims, Icombined, Jcombined, 0).reshape(
            len(Icombined), len(Jcombined)
        )
        self.update_max_sample(Pi)
        return MatrixLUCI.from_matrix(
            Pi, reltol=reltol, abstol=abstol, maxrank=maxbonddim, leftorthogonal=leftorthogonal
        )

    def update_pivots(
        self, b: int, f: Callable, leftorthogonal: bool, reltol: float = 1e-14, abstol: float = 0.0,
        maxbonddim: int | None = None, sweepdirection: str = "forward", pivotsearch: str = "full",
        verbosity: int = 0, extraIset: Sequence[tuple] | None = None, extraJset: Sequence[tuple] | None = None,
    ) -> None:
        """Core TCI2 pivot search: rank-revealing LU (full or rook-pivoted)
        over the genuine 2-site combined Pi domain."""
        self.invalidate_sitetensors()
        maxbonddim = maxbonddim if maxbonddim is not None else _UNBOUNDED_RANK
        extraIset = list(extraIset) if extraIset else []
        extraJset = list(extraJset) if extraJset else []

        Icombined = _union_preserve(kronecker_left(self.Iset[b], self.localdims[b]), extraIset)
        Jcombined = _union_preserve(kronecker_right(self.localdims[b + 1], self.Jset[b + 1]), extraJset)

        if pivotsearch == "full":
            luci = self._full_pivot_search(f, Icombined, Jcombined, reltol, abstol, maxbonddim, leftorthogonal)
        elif pivotsearch == "rook":
            icombined_pos = {v: i for i, v in enumerate(Icombined)}
            jcombined_pos = {v: i for i, v in enumerate(Jcombined)}
            I0 = [icombined_pos[i] for i in self.Iset[b + 1] if i in icombined_pos]
            J0 = [jcombined_pos[j] for j in self.Jset[b] if j in jcombined_pos]
            Pif = SubMatrix(f, Icombined, Jcombined)
            luci = MatrixLUCI.from_function(
                self.dtype, Pif, (len(Icombined), len(Jcombined)), I0, J0,
                reltol=reltol, abstol=abstol, maxrank=maxbonddim, leftorthogonal=leftorthogonal,
                pivotsearch="rook", usebatcheval=True,
            )
            self.update_max_sample(np.array([Pif.maxsamplevalue], dtype=self.dtype))

            if luci.npivots() == 0:  # fall back to full search if rook search fails
                luci = self._full_pivot_search(f, Icombined, Jcombined, reltol, abstol, maxbonddim, leftorthogonal)
        else:
            raise ValueError(f"Unknown pivot search strategy {pivotsearch}. Choose from 'rook', 'full'.")

        self.Iset[b + 1] = [Icombined[i] for i in luci.rowindices()]
        self.Jset[b] = [Jcombined[j] for j in luci.colindices()]
        if len(extraIset) == 0 and len(extraJset) == 0:
            self.set_sitetensor(b, luci.left())
            self.set_sitetensor(b + 1, luci.right())
        self.update_errors(b, luci.pivoterrors())

    def sweep2site(
        self, f: Callable, niter: int, iter1: int = 1, abstol: float = 1e-8, maxbonddim: int | None = None,
        sweepstrategy: str = "backandforth", pivotsearch: str = "full", verbosity: int = 0,
        strictlynested: bool = False, fillsitetensors: bool = True,
    ) -> None:
        self.invalidate_sitetensors()
        n = len(self)

        for iteration in range(iter1, iter1 + niter):
            extraIset: list[list[tuple]] = [[] for _ in range(n)]
            extraJset: list[list[tuple]] = [[] for _ in range(n)]
            if not strictlynested and len(self.Iset_history) > 0:
                extraIset = self.Iset_history[-1]
                extraJset = self.Jset_history[-1]

            # Shallow-per-site copy suffices (and is far cheaper than deepcopy): the tuples
            # inside each site's list are immutable, and Iset[b]/Jset[b] are always replaced
            # wholesale elsewhere (update_pivots) or grown via list.append (add_global_pivots'
            # pushunique) -- either way `list(s)` gives an independent list object per site.
            self.Iset_history.append([list(s) for s in self.Iset])
            self.Jset_history.append([list(s) for s in self.Jset])

            self.flush_pivot_error()
            if forwardsweep(sweepstrategy, iteration):
                for bondindex in range(n - 1):
                    self.update_pivots(
                        bondindex, f, True, abstol=abstol, maxbonddim=maxbonddim, sweepdirection="forward",
                        pivotsearch=pivotsearch, verbosity=verbosity,
                        extraIset=extraIset[bondindex + 1], extraJset=extraJset[bondindex],
                    )
            else:
                for bondindex in range(n - 2, -1, -1):
                    self.update_pivots(
                        bondindex, f, False, abstol=abstol, maxbonddim=maxbonddim, sweepdirection="backward",
                        pivotsearch=pivotsearch, verbosity=verbosity,
                        extraIset=extraIset[bondindex + 1], extraJset=extraJset[bondindex],
                    )

        if fillsitetensors:
            self.fill_sitetensors(f)

    def sanity_check(self) -> bool:
        for b in range(len(self) - 1):
            if len(self.Iset[b + 1]) != len(self.Jset[b]):
                raise ValueError(f"Pivot matrix at bond {b} is not square!")
        return True

    def search_global_pivots(
        self, f: Callable, abstol: float, verbosity: int = 0, nsearch: int = 100, maxnglobalpivot: int = 5
    ) -> list[tuple]:
        if nsearch == 0 or maxnglobalpivot == 0:
            return []
        if not self.sitetensors_available():
            self.fill_sitetensors(f)

        pivots: dict[float, tuple] = {}
        ttcache = TTCache.from_tt(self)
        for _ in range(nsearch):
            pivot, error = _floatingzone(ttcache, f, earlystoptol=10 * abstol, nsweeps=100)
            if error > abstol:
                pivots[error] = pivot
            if len(pivots) == maxnglobalpivot:
                break

        if len(pivots) == 0:
            if verbosity > 1:
                print("  No global pivot found")
            return []
        if verbosity > 1:
            print(f"  Found {len(pivots)} global pivots: max error {max(pivots)}")
        return list(pivots.values())


def convergence_criterion(
    ranks: Sequence[int], errors: Sequence[float], nglobalpivots: Sequence[int], tolerance: float,
    maxbonddim: int, ncheckhistory: int, checkconvglobalpivot: bool = True,
) -> bool:
    if len(errors) < ncheckhistory:
        return False
    lastranks = ranks[-ncheckhistory:]
    lastngpivots = nglobalpivots[-ncheckhistory:]
    lasterrors = errors[-ncheckhistory:]
    converged = (
        all(e < tolerance for e in lasterrors)
        and (all(g == 0 for g in lastngpivots) if checkconvglobalpivot else True)
        and min(lastranks) == lastranks[-1]
    )
    saturated = all(r >= maxbonddim for r in lastranks)
    return converged or saturated


def optimize(
    tci: TensorCI2, f: Callable, tolerance: float | None = None, maxbonddim: int | None = None, maxiter: int = 20,
    sweepstrategy: str = "backandforth", pivotsearch: str = "full", verbosity: int = 0, loginterval: int = 10,
    normalizeerror: bool = True, ncheckhistory: int = 3, globalpivotfinder=None, maxnglobalpivot: int = 5,
    nsearchglobalpivot: int = 5, tolmarginglobalsearch: float = 10.0, strictlynested: bool = False,
    checkbatchevaluatable: bool = False, checkconvglobalpivot: bool = True,
) -> tuple[list[int], list[float]]:
    errors: list[float] = []
    ranks: list[int] = []
    nglobalpivots: list[int] = []

    if checkbatchevaluatable and not isbatchevaluable(f):
        raise ValueError("Function `f` is not batch evaluatable")
    if 0 < nsearchglobalpivot < maxnglobalpivot:
        raise ValueError("nsearchglobalpivot < maxnglobalpivot!")

    tol = tolerance if tolerance is not None else 1e-8
    maxbonddim_eff = maxbonddim if maxbonddim is not None else _UNBOUNDED_RANK
    if maxbonddim is None and tol <= 0:
        raise ValueError(
            "Specify either tolerance > 0 or some maxbonddim; otherwise, the convergence "
            "criterion is not reachable!"
        )

    finder = globalpivotfinder if globalpivotfinder is not None else DefaultGlobalPivotFinder(
        nsearch=nsearchglobalpivot, maxnglobalpivot=maxnglobalpivot, tolmarginglobalsearch=tolmarginglobalsearch
    )

    globalpivots: list[tuple] = []
    for iteration in range(1, maxiter + 1):
        errornormalization = tci.maxsamplevalue if normalizeerror else 1.0
        abstol = tol * errornormalization

        tci.sweep2site(
            f, 2, iter1=1, abstol=abstol, maxbonddim=maxbonddim_eff, pivotsearch=pivotsearch,
            strictlynested=strictlynested, verbosity=verbosity, sweepstrategy=sweepstrategy, fillsitetensors=True,
        )
        errors.append(tci.pivot_error())

        input_ = GlobalPivotSearchInput(
            tci.localdims, TensorTrain.from_tt_like(tci), tci.maxsamplevalue, tci.Iset, tci.Jset
        )
        globalpivots = finder(input_, f, abstol, verbosity=verbosity)
        tci.add_global_pivots(globalpivots)
        nglobalpivots.append(len(globalpivots))

        ranks.append(tci.rank())
        if verbosity > 0 and iteration % loginterval == 0:
            print(
                f"iteration = {iteration}, rank = {ranks[-1]}, error= {errors[-1]}, "
                f"maxsamplevalue= {tci.maxsamplevalue}, nglobalpivot={len(globalpivots)}"
            )
        if convergence_criterion(
            ranks, errors, nglobalpivots, abstol, maxbonddim_eff, ncheckhistory,
            checkconvglobalpivot=checkconvglobalpivot,
        ):
            break

    errornormalization = tci.maxsamplevalue if normalizeerror else 1.0
    abstol = tol * errornormalization
    tci.sweep1site(f, abstol=abstol, maxbonddim=maxbonddim_eff)

    tci.sanity_check()

    return ranks, [e / errornormalization for e in errors]


def crossinterpolate2(
    dtype, f: Callable, localdims: Sequence[int], initialpivots: Sequence[Sequence[int]] | None = None, **kwargs
) -> tuple[TensorCI2, list[int], list[float]]:
    """Cross interpolate f using the TCI2 algorithm.

    By default, no caching takes place (matching the Julia original). If f
    is expensive, or has any pivot-search redundancy (rook/full pivot search
    and the global pivot finder frequently re-visit the same points --
    measured 5-15x redundant calls on typical problems), wrap it first:

        from pyqula.qutecipytk.tensortrain.cachedfunction import CachedFunction
        f_cached = CachedFunction(dtype, f, localdims)
        tci, ranks, errors = crossinterpolate2(dtype, f_cached, localdims, ...)

    This is the same pattern the Julia library documents, and cuts wall time
    substantially whenever f costs more than a dict-of-tuple lookup.
    """
    if initialpivots is None:
        initialpivots = [tuple([0] * len(localdims))]
    tci = TensorCI2.from_function(dtype, f, localdims, initialpivots)
    ranks, errors = optimize(tci, f, **kwargs)
    return tci, ranks, errors
