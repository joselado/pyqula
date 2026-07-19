"""Optional numba-accelerated kernel for rrLU's dense pivoting loop.

Soft dependency: if numba isn't installed, ``HAVE_NUMBA`` is False and
``rrlu.py`` falls back to the pure-Python implementation (which stays the
tested reference -- this module exists purely to go faster when numba is
available, never to change behavior).

Scope, per the profiling-driven optimization pass this was built from: numba
can only accelerate the *pure-numeric* part of rank-revealing LU (the
argmax-scan + rank-1 update + row/col swaps within ``rrLU._optimize``). It
cannot touch anything that calls the user's arbitrary Python ``f`` (the
``_generic_batchevaluate`` loop, ``arrlu``'s lazy submatrix evaluation, etc.)
-- those stay pure Python by necessity.

The kernel below is a direct, tie-break-faithful port of
``rrLU._optimize``/``_add_pivot``/``_swap_row``/``_swap_col``: the argmax
scan is an explicit column-major linear scan (columns outer, rows inner) with
strict ``>`` comparison, matching Julia's own ``submatrixargmax`` loop order
(and, correspondingly, ``pyqula.qutecipytk.matrix.rrlu.submatrixargmax``'s transposed
``np.argmax`` call) -- required so the numba path picks bit-for-bit the same
pivots as the Python path on exact ties (verified by the full test suite
passing identically with numba installed vs. not).
"""
from __future__ import annotations

try:
    from numba import njit
    HAVE_NUMBA = True
except ImportError:
    HAVE_NUMBA = False


if HAVE_NUMBA:
    @njit(cache=True)
    def _rrlu_pivot_kernel(A, rowperm, colperm, npivot, maxrank, reltol, abstol, leftorthogonal, error):
        # `error` is passed in (rather than reset to 0.0 here) so that, when the while
        # loop body never runs (e.g. maxrank <= npivot on entry), this kernel leaves it
        # unchanged -- matching both the pure-Python fallback (which only ever assigns
        # self.error inside the loop body) and the Julia reference (same: lu.error keeps
        # whatever value it had before _optimizerrlu! if the loop doesn't execute).
        maxerror = 0.0
        nrows, ncols = A.shape

        while npivot < maxrank:
            k = npivot

            # argmax(|A[k:, k:]|), first-occurrence-in-column-major-order tie-break
            # (columns outer, rows inner -- matches Julia's submatrixargmax).
            best_val = -1.0
            best_i = k
            best_j = k
            for j in range(k, ncols):
                for i in range(k, nrows):
                    v = abs(A[i, j])
                    if v > best_val:
                        best_val = v
                        best_i = i
                        best_j = j

            error = abs(A[best_i, best_j])
            if (error < reltol * maxerror or error < abstol) and npivot > 0:
                break
            if error > maxerror:
                maxerror = error

            if best_i != k:
                for col in range(ncols):
                    tmp = A[k, col]
                    A[k, col] = A[best_i, col]
                    A[best_i, col] = tmp
                tmp_p = rowperm[k]
                rowperm[k] = rowperm[best_i]
                rowperm[best_i] = tmp_p
            if best_j != k:
                for row in range(nrows):
                    tmp = A[row, k]
                    A[row, k] = A[row, best_j]
                    A[row, best_j] = tmp
                tmp_p = colperm[k]
                colperm[k] = colperm[best_j]
                colperm[best_j] = tmp_p

            npivot += 1

            pivotval = A[k, k]
            if leftorthogonal:
                for i in range(k + 1, nrows):
                    A[i, k] = A[i, k] / pivotval
            else:
                for j in range(k + 1, ncols):
                    A[k, j] = A[k, j] / pivotval

            for i in range(k + 1, nrows):
                xi = A[i, k]
                for j in range(k + 1, ncols):
                    A[i, j] = A[i, j] - xi * A[k, j]

        return npivot, error, maxerror
else:
    _rrlu_pivot_kernel = None
