"""Port of tensorci1.jl: the original TCI1 algorithm.

0-based indexing throughout. Bookkeeping note: Julia's ``Iset``/``Jset``/
``newI``/``newJ`` arrays use an "array position == prefix/suffix length"
convention (position p holds length-p entries), while ``Pi``/``aca``/``P``/
``PiIset``/``PiJset``/``T`` use a "array position == bond index" convention
(1-based bond p in Julia). Converting both conventions to 0-based shifts
them differently relative to each other (position-as-length needs no shift;
position-as-bond-index shifts by one) -- this is the trickiest bookkeeping
in the whole port; each cross-reference below is commented with which
convention it uses.
"""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

from pyqula.qutecipytk.indexset import IndexSet
from pyqula.qutecipytk.matrix.aca import MatrixACA, MatrixCI, a_times_binv
from pyqula.qutecipytk.tensortrain.base import AbstractTensorTrain
from pyqula.qutecipytk.util import maxabs, forwardsweep


class TensorCI1(AbstractTensorTrain):
    def __init__(self, dtype, localdims: Sequence[int]):
        n = len(localdims)
        self.dtype = dtype
        self.localdims = list(localdims)
        self.Iset: list[IndexSet] = [IndexSet() for _ in range(n)]
        self.Jset: list[IndexSet] = [IndexSet() for _ in range(n)]
        self.T = [np.zeros((0, d, 0), dtype=dtype) for d in localdims]
        self.P = [np.zeros((0, 0), dtype=dtype) for _ in range(n)]
        self.aca = [MatrixACA.empty(dtype, 0, 0) for _ in range(n)]
        self.Pi = [np.zeros((0, 0), dtype=dtype) for _ in range(n)]
        self.PiIset: list[IndexSet] = [IndexSet() for _ in range(n)]
        self.PiJset: list[IndexSet] = [IndexSet() for _ in range(n)]
        self.pivoterrors = [np.inf] * max(n - 1, 0)
        self.maxsamplevalue = 0.0

    @classmethod
    def from_function(cls, dtype, func: Callable, localdims: Sequence[int], firstpivot: Sequence[int]) -> "TensorCI1":
        tci = cls(dtype, localdims)
        f = lambda x: dtype(func(x))

        tci.maxsamplevalue = abs(f(list(firstpivot)))
        if tci.maxsamplevalue == 0:
            raise ValueError("Please provide a first pivot where f(pivot) != 0.")
        if len(localdims) != len(firstpivot):
            raise ValueError("firstpivot and localdims must have the same length.")

        n = len(localdims)
        firstpivot = list(firstpivot)
        tci.Iset = [IndexSet([tuple(firstpivot[:p])]) for p in range(n)]
        tci.Jset = [IndexSet([tuple(firstpivot[p + 1:])]) for p in range(n)]
        tci.PiIset = [tci._get_pi_iset(p) for p in range(n)]
        tci.PiJset = [tci._get_pi_jset(p) for p in range(n)]
        tci.Pi = [tci._get_pi(p, f) for p in range(n - 1)]

        for p in range(n - 1):
            localpivot = (
                tci.PiIset[p].pos(tci.Iset[p + 1][0]),
                tci.PiJset[p + 1].pos(tci.Jset[p][0]),
            )
            tci.aca[p] = MatrixACA.from_matrix(tci.Pi[p], localpivot)
            if p == 0:
                tci._update_T(0, tci.Pi[p][:, [localpivot[1]]])
            tci._update_T(p + 1, tci.Pi[p][[localpivot[0]], :])
            tci.P[p] = tci.Pi[p][np.ix_([localpivot[0]], [localpivot[1]])]
        tci.P[-1] = np.ones((1, 1), dtype=dtype)

        return tci

    # -- AbstractTensorTrain contract --------------------------------------

    def _t_times_pinv(self, p: int) -> np.ndarray:
        T = self.T[p]
        shape = T.shape
        tpinv = a_times_binv(T.reshape(shape[0] * shape[1], shape[2]), self.P[p])
        return tpinv.reshape(shape)

    def sitetensor(self, p: int) -> np.ndarray:
        return self._t_times_pinv(p)

    def sitetensors(self) -> list[np.ndarray]:
        return [self.sitetensor(p) for p in range(len(self.T))]

    def __len__(self) -> int:
        return len(self.T)

    # Overridden (matching the Julia original) to read bond dimensions directly off
    # the raw T tensors, rather than going through sitetensor()'s T*P^-1 computation
    # -- P can be numerically fragile/near-singular mid-sweep, long before pivots
    # settle, so linkdims/rank must not depend on inverting it.
    def linkdims(self) -> list[int]:
        return [T.shape[0] for T in self.T[1:]]

    def linkdim(self, i: int) -> int:
        return self.T[i + 1].shape[0]

    def sitedims(self) -> list[list[int]]:
        return [list(T.shape[1:-1]) for T in self.T]

    def sitedim(self, i: int) -> list[int]:
        return list(self.T[i].shape[1:-1])

    def evaluate(self, indexset):
        result = None
        for p in range(len(self)):
            mat = a_times_binv(self.T[p][:, indexset[p], :], self.P[p])
            result = mat if result is None else result @ mat
        return result[0, 0]

    # -- internal Pi/T bookkeeping ------------------------------------------

    def _get_pi_iset(self, p: int) -> IndexSet:
        items = [i + (u,) for i in self.Iset[p].fromint for u in range(self.localdims[p])]
        return IndexSet(items)

    def _get_pi_jset(self, p: int) -> IndexSet:
        items = [(u,) + j for u in range(self.localdims[p]) for j in self.Jset[p].fromint]
        return IndexSet(items)

    def _get_pi(self, p: int, f: Callable) -> np.ndarray:
        iset = self.PiIset[p]
        jset = self.PiJset[p + 1]
        res = np.array([[f(list(i) + list(j)) for j in jset.fromint] for i in iset.fromint], dtype=self.dtype)
        self._update_max_sample(res)
        return res

    def _update_max_sample(self, samples) -> None:
        self.maxsamplevalue = maxabs(self.maxsamplevalue, samples)

    def _get_cross(self, p: int) -> MatrixCI:
        iset = self.PiIset[p].pos(list(self.Iset[p + 1].fromint))
        jset = self.PiJset[p + 1].pos(list(self.Jset[p].fromint))
        shape = self.T[p].shape
        Tp = self.T[p].reshape(shape[0] * shape[1], shape[2])
        shape1 = self.T[p + 1].shape
        Tp1 = self.T[p + 1].reshape(shape1[0], shape1[1] * shape1[2])
        return MatrixCI(iset, jset, Tp, Tp1)

    def _update_T(self, p: int, new_T) -> None:
        self.T[p] = np.asarray(new_T, dtype=self.dtype).reshape(
            len(self.Iset[p]), self.localdims[p], len(self.Jset[p])
        )

    def _update_pi_rows(self, p: int, f: Callable) -> None:
        newIset = self._get_pi_iset(p)
        old_set = set(self.PiIset[p].fromint)
        diffIset = [x for x in newIset.fromint if x not in old_set]
        newPi = np.empty((len(newIset), self.Pi[p].shape[1]), dtype=self.dtype)

        permutation = np.array([newIset.pos(i) for i in self.PiIset[p].fromint], dtype=np.int64)
        newPi[permutation, :] = self.Pi[p]

        for imulti in diffIset:
            newi = newIset.pos(imulti)
            row = np.array([f(list(imulti) + list(j)) for j in self.PiJset[p + 1].fromint], dtype=self.dtype)
            newPi[newi, :] = row
            self._update_max_sample(row)

        self.Pi[p] = newPi
        self.PiIset[p] = newIset

        Tshape = self.T[p].shape
        Tp = self.T[p].reshape(Tshape[0] * Tshape[1], Tshape[2])
        self.aca[p].set_rows(Tp, permutation)

    def _update_pi_cols(self, p: int, f: Callable) -> None:
        newJset = self._get_pi_jset(p + 1)
        old_set = set(self.PiJset[p + 1].fromint)
        diffJset = [x for x in newJset.fromint if x not in old_set]
        newPi = np.empty((self.Pi[p].shape[0], len(newJset)), dtype=self.dtype)

        permutation = np.array([newJset.pos(j) for j in self.PiJset[p + 1].fromint], dtype=np.int64)
        newPi[:, permutation] = self.Pi[p]

        for jmulti in diffJset:
            newj = newJset.pos(jmulti)
            col = np.array([f(list(i) + list(jmulti)) for i in self.PiIset[p].fromint], dtype=self.dtype)
            newPi[:, newj] = col
            self._update_max_sample(col)

        self.Pi[p] = newPi
        self.PiJset[p + 1] = newJset

        Tshape = self.T[p + 1].shape
        Tp = self.T[p + 1].reshape(Tshape[0], Tshape[1] * Tshape[2])
        self.aca[p].set_cols(Tp, permutation)

    def _add_pivot_row(self, cross: MatrixCI, p: int, newi: int, f: Callable) -> None:
        self.aca[p].add_pivot_row(self.Pi[p], newi)
        cross.add_pivot_row(self.Pi[p], newi)
        self.Iset[p + 1].append(self.PiIset[p][newi])
        self._update_T(p + 1, cross.pivotrows)
        self.P[p] = cross.pivotmatrix()

        if p < len(self) - 2:
            self._update_pi_rows(p + 1, f)

    def _add_pivot_col(self, cross: MatrixCI, p: int, newj: int, f: Callable) -> None:
        self.aca[p].add_pivot_col(self.Pi[p], newj)
        cross.add_pivot_col(self.Pi[p], newj)
        self.Jset[p].append(self.PiJset[p + 1][newj])
        self._update_T(p, cross.pivotcols)
        self.P[p] = cross.pivotmatrix()

        if p > 0:
            self._update_pi_cols(p - 1, f)

    def add_pivot(self, p: int, f: Callable, tolerance: float = 1e-12) -> None:
        n = len(self)
        if not (0 <= p <= n - 2):
            raise IndexError(f"Pi tensors can only be built at sites 0 to {n - 2}.")

        if self.aca[p].rank() >= min(self.Pi[p].shape):
            self.pivoterrors[p] = 0.0
            return

        newpivot, newerror = self.aca[p].find_new_pivot(self.Pi[p])
        self.pivoterrors[p] = newerror
        if newerror < tolerance:
            return

        cross = self._get_cross(p)
        self._add_pivot_col(cross, p, newpivot[1], f)
        self._add_pivot_row(cross, p, newpivot[0], f)

    # -- global pivot insertion ----------------------------------------------

    def _cross_error(self, f: Callable, x: tuple, y: tuple) -> float:
        if len(x) == 0 or len(y) == 0:
            return 0.0
        b = len(x)  # bond index (Julia 1-based "bondindex" = length(x))
        if x in self.Iset[b].toint or y in self.Jset[b - 1].toint:
            return 0.0
        if len(self.Jset[b - 1]) == 0:
            return abs(f(list(x) + list(y)))

        fx = np.array([f(list(x) + list(j)) for j in self.Jset[b - 1].fromint], dtype=self.dtype)
        fy = np.array([f(list(i) + list(y)) for i in self.Iset[b].fromint], dtype=self.dtype)
        self._update_max_sample(fx)
        self._update_max_sample(fy)
        val = (a_times_binv(fx.reshape(1, -1), self.P[b - 1]) @ fy)[0]
        return abs(val - f(list(x) + list(y)))

    def _update_I_proposal(self, f, newpivot, newI, newJ, abstol) -> list:
        n = len(self)
        error = np.inf
        for b in range(1, n):
            if len(newI[b]) == 0:
                error = 0.0
                continue
            if error > abstol:
                newI[b] = newI[b - 1] + (newpivot[b - 1],)
                error = self._cross_error(f, newI[b], newJ[b - 1])
            elif tuple(newpivot[:b]) in self.Iset[b - 1].toint:
                # Julia's `newpivot[1:bondindex] in tci.Iset[bondindex]` compares a
                # length-bondindex vector against length-(bondindex-1) entries, so this
                # branch is provably unreachable there; mirror that length mismatch here
                # (newpivot[:b] has length b, self.Iset[b - 1] entries have length b - 1)
                # rather than diverging from the reference algorithm's actual behavior.
                newI[b] = tuple(newpivot[:b])
                error = self._cross_error(f, newI[b], newJ[b - 1])
            else:
                xset = [i + (newpivot[b - 1],) for i in self.Iset[b - 1].fromint]
                errors = [self._cross_error(f, x, newJ[b - 1]) for x in xset]
                maxindex = int(np.argmax(errors))
                newI[b] = xset[maxindex]
                error = errors[maxindex]
            if error < abstol:
                newI[b] = ()
        return newI

    def _update_J_proposal(self, f, newpivot, newI, newJ, abstol) -> list:
        n = len(self)
        error = np.inf
        for b in range(n - 1, 0, -1):
            if len(newJ[b - 1]) == 0:
                error = 0.0
                continue
            if error > abstol:
                newJ[b - 1] = (newpivot[b],) + newJ[b]
                error = self._cross_error(f, newI[b], newJ[b - 1])
            elif tuple(newpivot[b + 1:]) in self.Jset[b].toint:
                newJ[b - 1] = tuple(newpivot[b:])
                error = self._cross_error(f, newI[b], newJ[b - 1])
            else:
                yset = [(newpivot[b],) + j for j in self.Jset[b].fromint]
                errors = [self._cross_error(f, newI[b], y) for y in yset]
                maxindex = int(np.argmax(errors))
                newJ[b - 1] = yset[maxindex]
                error = errors[maxindex]
            if error < abstol:
                newJ[b - 1] = ()
        return newJ

    def add_global_pivot(self, f: Callable, newpivot: Sequence[int], abstol: float) -> None:
        n = len(self)
        if len(newpivot) != n:
            raise ValueError(f"New global pivot {newpivot} should have exactly {n} entries.")
        newpivot = list(newpivot)

        newI = [tuple(newpivot[:p]) for p in range(n)]
        newJ = [tuple(newpivot[p + 1:]) for p in range(n)]
        newI = self._update_I_proposal(f, newpivot, newI, newJ, abstol)

        for _ in range(n):
            newJ = self._update_J_proposal(f, newpivot, newI, newJ, abstol)
            newI = self._update_I_proposal(f, newpivot, newI, newJ, abstol)
            empties_I = [len(x) == 0 for x in newI[1:]]
            empties_J = [len(x) == 0 for x in newJ[:-1]]
            if empties_I == empties_J:
                break

        for p in range(1, n):
            if len(newI[p]) > 0:
                pb = p - 1
                cross = self._get_cross(pb)
                newi_pos = self.PiIset[pb].pos(newI[p])
                self._add_pivot_row(cross, pb, newi_pos, f)

        for p in range(n - 1, 0, -1):
            if len(newJ[p - 1]) > 0:
                pb = p - 1
                cross = self._get_cross(pb)
                # PiJset access is offset by +1 relative to the bond index (unlike PiIset) --
                # matches the same PiJset[p+1] convention used in _get_pi/_get_cross/_update_pi_cols.
                newj_pos = self.PiJset[pb + 1].pos(newJ[p - 1])
                self._add_pivot_col(cross, pb, newj_pos, f)

    def last_sweep_pivot_error(self) -> float:
        return max(self.pivoterrors) if self.pivoterrors else 0.0


def crossinterpolate1(
    dtype,
    f: Callable,
    localdims: Sequence[int],
    firstpivot: Sequence[int] | None = None,
    tolerance: float = 1e-8,
    maxiter: int = 200,
    sweepstrategy: str = "backandforth",
    pivottolerance: float = 1e-12,
    verbosity: int = 0,
    additionalpivots: Sequence[Sequence[int]] = (),
    normalizeerror: bool = True,
) -> tuple[TensorCI1, list[int], list[float]]:
    """Cross interpolate f using the TCI1 algorithm. No caching takes place
    by default; wrap f in qutecipy.tensortrain.cachedfunction.CachedFunction
    first if it's expensive to evaluate (see crossinterpolate2's docstring)."""
    if firstpivot is None:
        firstpivot = [0] * len(localdims)
    tci = TensorCI1.from_function(dtype, f, localdims, firstpivot)
    n = len(tci)
    errors: list[float] = []
    ranks: list[int] = []

    for pivot in additionalpivots:
        tci.add_global_pivot(f, pivot, tolerance)

    for iteration in range(tci.rank() + 1, maxiter + 1):
        if forwardsweep(sweepstrategy, iteration):
            for bondindex in range(n - 1):
                tci.add_pivot(bondindex, f, pivottolerance)
        else:
            for bondindex in range(n - 2, -1, -1):
                tci.add_pivot(bondindex, f, pivottolerance)

        errornormalization = tci.maxsamplevalue if normalizeerror else 1.0
        errors.append(tci.last_sweep_pivot_error())
        ranks.append(tci.rank())
        if verbosity > 0 and iteration % 10 == 0:
            print(f"iteration = {iteration}, rank = {ranks[-1]}, error = {errors[-1]}")
        if errors[-1] < tolerance * errornormalization:
            break

    errornormalization = tci.maxsamplevalue if normalizeerror else 1.0
    return tci, ranks, [e / errornormalization for e in errors]
