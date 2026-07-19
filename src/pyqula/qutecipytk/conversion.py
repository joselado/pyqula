"""Port of conversion.jl: TCI1<->TCI2 interconversion, and
sweep1sitegetindices!/TensorCI2(tt::TensorTrain), which re-derives a
canonical pivot-based TCI2 representation from an arbitrary dense
TensorTrain (needed after contract_naive/contract_zipup produce raw cores).
"""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

from pyqula.qutecipytk.indexset import IndexSet
from pyqula.qutecipytk.matrix.aca import MatrixACA
from pyqula.qutecipytk.matrix.luci import MatrixLUCI
from pyqula.qutecipytk.matrix.rrlu import rrLU
from pyqula.qutecipytk.tci1 import TensorCI1
from pyqula.qutecipytk.tci2 import TensorCI2, kronecker_left, kronecker_right

_UNBOUNDED_RANK = 2**62


def matrixaca_from_rrlu(lu: rrLU) -> MatrixACA:
    aca = MatrixACA.empty(lu.L.dtype, *lu.size())
    aca.rowindices = lu.rowindices()
    aca.colindices = lu.colindices()
    aca.u = lu.left()
    aca.v = lu.right()
    diagvals = lu.diag()
    aca.alpha = list(1.0 / diagvals)

    if lu.leftorthogonal:
        for j in range(aca.u.shape[1]):
            aca.u[:, j] *= diagvals[j]
    else:
        for i in range(aca.v.shape[0]):
            aca.v[i, :] *= diagvals[i]
    return aca


def tci1_from_tci2(tci2: TensorCI2, f: Callable) -> TensorCI1:
    L = len(tci2)
    dtype = tci2.dtype
    tci1 = TensorCI1(dtype, tci2.localdims)
    tci1.Iset = [IndexSet(s) for s in tci2.Iset]
    tci1.Jset = [IndexSet(s) for s in tci2.Jset]
    tci1.PiIset = [tci1._get_pi_iset(p) for p in range(L)]
    tci1.PiJset = [tci1._get_pi_jset(p) for p in range(L)]
    tci1.Pi = [tci1._get_pi(p, f) for p in range(L - 1)]

    for ell in range(L - 1):
        iset = tci1.PiIset[ell].pos(list(tci1.Iset[ell + 1].fromint))
        jset = tci1.PiJset[ell + 1].pos(list(tci1.Jset[ell].fromint))
        tci1._update_T(ell, tci1.Pi[ell][:, jset])
        if ell == L - 2:
            tci1._update_T(L - 1, tci1.Pi[ell][iset, :])
        tci1.P[ell] = tci1.Pi[ell][np.ix_(iset, jset)]

        tci1.aca[ell] = MatrixACA.from_matrix(tci1.Pi[ell], (iset[0], jset[0]))
        for rowindex, colindex in zip(iset[1:], jset[1:]):
            tci1.aca[ell].add_pivot_col(tci1.Pi[ell], colindex)
            tci1.aca[ell].add_pivot_row(tci1.Pi[ell], rowindex)

    tci1.P[-1] = np.ones((1, 1), dtype=dtype)
    tci1.pivoterrors = list(tci2.bonderrors)
    tci1.maxsamplevalue = tci2.maxsamplevalue
    return tci1


def tci2_from_tci1(tci1: TensorCI1) -> TensorCI2:
    tci2 = TensorCI2(tci1.dtype, tci1.localdims)
    tci2.Iset = [list(i.fromint) for i in tci1.Iset]
    tci2.Jset = [list(j.fromint) for j in tci1.Jset]
    tci2.localdims = list(tci1.localdims)
    L = len(tci1)
    for p in range(L - 1):
        tci2._sitetensors[p] = tci1._t_times_pinv(p)
    tci2._sitetensors[L - 1] = tci1.T[L - 1]
    tci2.pivoterrors = []
    tci2.bonderrors = list(tci1.pivoterrors)
    tci2.maxsamplevalue = tci1.maxsamplevalue
    return tci2


def sweep1site_get_indices(
    tt, forwardsweep: bool, spectatorindices: list[list[tuple]] | None = None,
    maxbonddim: int | None = None, tolerance: float = 0.0,
) -> tuple[list[list[tuple]], np.ndarray]:
    """Single 1-site sweep (forward or backward) over an already fully-dense
    TensorTrain: factorizes each site via rank-revealing LU, building up
    nested pivot index sets. Mutates tt's site tensors in place (matching
    the Julia original's `tt.sitetensors[ell] = ...` direct mutation).

    If spectatorindices is given (e.g. a Jset being refined against a newly
    swept Iset), it is mutated in place too -- matches Julia's aliasing:
    arrays passed by reference, so `spectatorindices[ell] = ...` inside also
    mutates the caller's list. This is relied on by tci2_from_tensortrain's
    refinement loop, ported faithfully as-is (see its docstring).
    """
    maxbonddim = maxbonddim if maxbonddim is not None else _UNBOUNDED_RANK
    indexset: list[list[tuple]] = [[()]]
    pivoterrorsarray = np.zeros(tt.rank() + 1)

    def group_indices(T: np.ndarray, next_: bool) -> np.ndarray:
        shape_ = T.shape
        if forwardsweep != next_:
            return T.reshape(-1, shape_[-1])
        return T.reshape(shape_[0], -1)

    def split_indices(T: np.ndarray, shape_: tuple, newbonddim: int, next_: bool) -> np.ndarray:
        if forwardsweep != next_:
            newshape = shape_[:-1] + (newbonddim,)
        else:
            newshape = (newbonddim,) + shape_[1:]
        return T.reshape(newshape)

    L = len(tt)
    for i in range(1, L):
        if forwardsweep:
            ell, ellnext = i - 1, i
        else:
            ell, ellnext = L - i, L - i - 1

        shape = tt.sitetensor(ell).shape
        shapenext = tt.sitetensor(ellnext).shape

        luci = MatrixLUCI.from_matrix(
            group_indices(tt.sitetensor(ell), False), leftorthogonal=forwardsweep,
            abstol=tolerance, maxrank=maxbonddim,
        )

        if forwardsweep:
            candidates = kronecker_left(indexset[-1], shape[1])
            indexset.append([candidates[i2] for i2 in luci.rowindices()])
            if spectatorindices:
                spectatorindices[ell] = [spectatorindices[ell][j] for j in luci.colindices()]
        else:
            candidates = kronecker_right(shape[1], indexset[-1])
            indexset.append([candidates[j] for j in luci.colindices()])
            if spectatorindices:
                spectatorindices[ell] = [spectatorindices[ell][i2] for i2 in luci.rowindices()]

        tt._sitetensors[ell] = split_indices(
            luci.left() if forwardsweep else luci.right(), shape, luci.npivots(), False
        )

        if forwardsweep:
            nexttensor = luci.right() @ group_indices(tt.sitetensor(ellnext), True)
        else:
            nexttensor = group_indices(tt.sitetensor(ellnext), True) @ luci.left()
        tt._sitetensors[ellnext] = split_indices(nexttensor, shapenext, luci.npivots(), True)

        n_upto = luci.npivots() + 1
        pivoterrorsarray[:n_upto] = np.maximum(pivoterrorsarray[:n_upto], luci.pivoterrors())

    if forwardsweep:
        return indexset, pivoterrorsarray
    return list(reversed(indexset)), pivoterrorsarray


def tci2_from_tensortrain(tt, tolerance: float = 1e-12, maxbonddim: int | None = None, maxiter: int = 3) -> TensorCI2:
    maxbonddim = maxbonddim if maxbonddim is not None else _UNBOUNDED_RANK

    Iset, _ = sweep1site_get_indices(tt, True, maxbonddim=maxbonddim, tolerance=tolerance)
    Jset, pivoterrors = sweep1site_get_indices(tt, False, maxbonddim=maxbonddim, tolerance=tolerance)

    for iteration in range(3, maxiter + 1):
        if iteration % 2 == 1:
            Isetnew, pivoterrors = sweep1site_get_indices(tt, True, Jset)
            if Isetnew == Iset:
                break
        else:
            Jsetnew, pivoterrors = sweep1site_get_indices(tt, False, Iset)
            if Jsetnew == Jset:
                break

    localdims = [d[0] for d in tt.sitedims()]
    tci2 = TensorCI2(tt.sitetensor(0).dtype, localdims)
    tci2.Iset = Iset
    tci2.Jset = Jset
    tci2._sitetensors = list(tt.sitetensors())
    tci2.pivoterrors = list(pivoterrors)
    tci2.maxsamplevalue = max(float(np.max(np.abs(T))) for T in tci2._sitetensors)
    return tci2
