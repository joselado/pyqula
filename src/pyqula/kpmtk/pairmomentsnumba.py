# Chebyshev moments <e_i|T_n(H)|e_j> of many (i,j) pairs and k-points, for
# the KPM mean field on the CPU (kpmtk/densitymatrix_kpm.py). The GPU
# counterpart, with the same two entry points, is kpmtk/pairmomentsjax.py.
#
# The recursion runs on a dense block whose columns are starting vectors
# e_j, V_{n+1} = 2 H V_n - V_{n-1}, with H in CSR form and the block stored
# row by row, so that H@V is, for every row, a sum of whole rows of V; the
# rows are split among numba's threads (or, for a small cell on a k-mesh,
# the k-points are), and V_{n-1} is overwritten in place, so two blocks
# are live at a time. The real and imaginary parts of
# the block are kept as two real arrays, which numba vectorizes and which
# took less than half the time of a complex array on a 20,000-orbital
# island (future_development/sparse_kpm_mean_field.md), and a Hamiltonian
# whose entries are exactly real at a k-point runs on the real part only.
#
# A block runs in one of two ways. Read on the rows: the moment of a pair
# is the entry of its row in the column of its starting vector, read at
# every step, 2*npol steps. Doubled: when both orbitals of a pair are
# columns of the block, T_{2n} = 2 T_n T_n - T_0 and
# T_{2n+1} = 2 T_{n+1} T_n - T_1 give two moments per step from inner
# products of the columns, npol steps. The orbitals of the pairs that are
# not starting columns are added to the block when this pays, which for a
# local interaction is a few columns, and for every pair of a small cell
# (the dense engine on a k-mesh) is never, since the inner products of
# N^2 pairs cost N^3 per step.

import numpy as np
import numba
from numba import jit, prange
from scipy.sparse import csr_matrix

_DTYPES = {"single": np.float32, "double": np.float64}

# the most entries (orbitals x starting columns) of the two blocks live at
# a time, together; each entry is a real and an imaginary part, so this is
# 128 MB in double precision. Below _MIN_COLUMNS columns a step costs more
# per column (on six desktop cores at 20,000 orbitals, 1.6 times more at 16
# columns and 2.4 at 8, and the same from 64 to 256), so a block never has
# fewer, and beyond 65,536 orbitals the memory of the blocks grows linearly,
# at 2 kB per orbital in double precision
_MAX_BLOCK = 2**23
_MIN_COLUMNS = 64

# a block of at most this many entries runs its rows in one thread, since
# splitting them costs more than they do
_SERIAL_BLOCK = 2**12

# the cost of the inner products of a doubled pair, per row and step, in
# units of one stored entry of H times one column of the block: a diagonal
# pair is the norm of its column, taken for every column at once, and the
# others are taken one at a time
_DIAGONAL_COST = 1.
_PAIR_COST = 4.


def get_precision_names():
    """Every precision the kernels of this module accept"""
    return sorted(_DTYPES)


def _plan(pairs, N, nnz):
    """Split the pairs (i,j) by their starting column j into blocks of at
    most _MAX_BLOCK entries, and decide for each block whether it is
    doubled. Returns a list of (columns, sel, doubled, ci, cj): the
    orbitals of the block columns, the indexes of its pairs in pairs, and
    for every pair the block column of its row (doubled) or its row (read
    on the rows), and the block column of its starting vector"""
    z = max(1., nnz/N) # stored entries per row
    order = np.argsort(pairs[:, 1], kind="stable")
    pj = pairs[order, 1]
    starts = np.unique(pj)
    ncol = min(len(starts), max(_MIN_COLUMNS, _MAX_BLOCK//(2*N)))
    blocks = []
    for c0 in range(0, len(starts), ncol):
        chunk = starts[c0:c0+ncol]
        lo = np.searchsorted(pj, chunk[0], side="left")
        hi = np.searchsorted(pj, chunk[-1], side="right")
        sel = order[lo:hi]
        rows, cols = pairs[sel, 0], pairs[sel, 1]
        both = np.union1d(chunk, rows) # every orbital the doubling needs
        # per two moments: one step of the wider block and the inner
        # products, against two steps of the starting columns alone
        ndiag = np.count_nonzero(rows == cols)
        cost = (z*len(both) + _DIAGONAL_COST*len(both)
                + _PAIR_COST*(len(sel) - ndiag))
        doubled = cost < 2*z*len(chunk) and len(both) <= 2*ncol
        if doubled:
            blocks.append((both, sel, True, np.searchsorted(both, rows),
                np.searchsorted(both, cols)))
        else:
            blocks.append((chunk, sel, False, rows,
                np.searchsorted(chunk, cols)))
    return blocks


# The kernels below are written once for a complex and a real H: with
# cplx=False the imaginary parts are never read, and the arrays passed for
# them can be empty. The work of one row is in _step_row and
# _products_row, shared by the drivers, which run the rows of one block
# split among the threads (_doubled_rows, _read_rows), or one after the
# other (_doubled_serial, _read_serial) for a block too small to split,
# or for many independent members at once with one member per thread
# (_doubled_members, _read_members): the k-points of a small cell on a
# k-mesh, or the tiles of the truncated recursion, where splitting their
# few rows costs more than the rows themselves.

@jit(nopython=True, cache=True)
def _step_row(r, cplx, indptr, indices, d2r, d2i, Vr, Vi, Wr, Wi):
    """Row r of W = 2 H V - W, with d2 = 2H"""
    ncol = Vr.shape[1]
    wr = Wr[r]
    for c in range(ncol): wr[c] = -wr[c]
    if cplx:
        wi = Wi[r]
        for c in range(ncol): wi[c] = -wi[c]
        for k in range(indptr[r], indptr[r+1]):
            xr = Vr[indices[k]]; xi = Vi[indices[k]]
            a = d2r[k]; b = d2i[k]
            for c in range(ncol):
                wr[c] += a*xr[c] - b*xi[c]
                wi[c] += a*xi[c] + b*xr[c]
    else:
        for k in range(indptr[r], indptr[r+1]):
            xr = Vr[indices[k]]
            a = d2r[k]
            for c in range(ncol): wr[c] += a*xr[c]


@jit(nopython=True, cache=True)
def _products_row(r, cplx, Vr, Vi, Wr, Wi, oi, oj, dacc, oacc):
    """Add row r to the inner products of the doubling, V holding V_n and
    W V_{n+1}: dacc[0] and dacc[1] the norms <V_n e_c|V_n e_c> and
    <V_{n+1} e_c|V_n e_c> of every column c, which are real, taken in a
    loop numba vectorizes, and oacc[0:2] and oacc[2:4] the real and
    imaginary parts of <V_n e_i|V_n e_j> and <V_{n+1} e_i|V_n e_j> of the
    pairs (oi,oj) that are not diagonal, one pair at a time"""
    ncol = Vr.shape[1]
    vr = Vr[r]; wr = Wr[r]
    d0 = dacc[0]; d1 = dacc[1]
    if cplx:
        vi = Vi[r]; wi = Wi[r]
        for c in range(ncol):
            d0[c] += vr[c]*vr[c] + vi[c]*vi[c]
            d1[c] += wr[c]*vr[c] + wi[c]*vi[c]
        for q in range(len(oi)):
            i = oi[q]; j = oj[q]
            oacc[0, q] += vr[i]*vr[j] + vi[i]*vi[j]
            oacc[1, q] += vr[i]*vi[j] - vi[i]*vr[j]
            oacc[2, q] += wr[i]*vr[j] + wi[i]*vi[j]
            oacc[3, q] += wr[i]*vi[j] - wi[i]*vr[j]
    else:
        for c in range(ncol):
            d0[c] += vr[c]*vr[c]
            d1[c] += wr[c]*vr[c]
        for q in range(len(oi)):
            i = oi[q]; j = oj[q]
            oacc[0, q] += vr[i]*vr[j]
            oacc[2, q] += wr[i]*vr[j]


@jit(nopython=True, cache=True)
def _start(cplx, indptr, indices, d2r, d2i, columns):
    """The blocks V_0 (in W), the starting vectors e_c of the block columns,
    and V_1 = H V_0 (in V), as planes of the dtype of d2"""
    N = len(indptr) - 1
    ncol = len(columns)
    Wr = np.zeros((N, ncol), dtype=d2r.dtype)
    Vr = np.zeros((N, ncol), dtype=d2r.dtype)
    if cplx:
        Wi = np.zeros((N, ncol), dtype=d2r.dtype)
        Vi = np.zeros((N, ncol), dtype=d2r.dtype)
    else:
        Wi = np.zeros((0, 0), dtype=d2r.dtype)
        Vi = np.zeros((0, 0), dtype=d2r.dtype)
    for c in range(ncol): Wr[columns[c], c] = 1.
    for r in range(N): _step_row(r, cplx, indptr, indices, d2r, d2i, Wr, Wi, Vr, Vi)
    for r in range(N): # 2 H V_0 - 0, halved, which is exact
        for c in range(ncol): Vr[r, c] = Vr[r, c]/2
        if cplx:
            for c in range(ncol): Vi[r, c] = Vi[r, c]/2
    return Vr, Vi, Wr, Wi


@jit(nopython=True, cache=True)
def _entry(cplx, Ar, Ai, r, c):
    """The entry (r,c) of a block given as planes, in double precision"""
    if cplx: return complex(Ar[r, c]) + 1j*complex(Ai[r, c])
    return complex(Ar[r, c])


@jit(nopython=True, cache=True)
def _record(mu, n, p, diag, coef, out, tr, state):
    """Add the moment mu_n of pair p to its contracted values, one per
    column of coef, to the trace when the pair is diagonal, and to the
    largest modulus in state[0], with state[1] set by a moment that is not
    finite"""
    for c in range(coef.shape[1]): out[p, c] += coef[n, c]*mu
    if diag[p]: tr[n] += mu
    m = abs(mu)
    if not np.isfinite(m): state[1] = 1.
    elif m > state[0]: state[0] = m


@jit(nopython=True, cache=True)
def _doubled_moments(n, dsum, osum, ci, diag, mu1, coef, out, tr, state):
    """Moments 2n and 2n+1 of every pair from the inner products of step n:
    T_{2n} = 2 T_n T_n - T_0 and T_{2n+1} = 2 T_{n+1} T_n - T_1"""
    q = 0
    for p in range(len(ci)):
        if diag[p]:
            m2 = complex(2.*dsum[0, ci[p]] - 1.)
            m3 = 2.*dsum[1, ci[p]] - mu1[p]
        else:
            m2 = 2.*(osum[0, q] + 1j*osum[1, q])
            m3 = 2.*(osum[2, q] + 1j*osum[3, q]) - mu1[p]
            q += 1
        _record(m2, 2*n, p, diag, coef, out, tr, state)
        if 2*n + 1 < len(coef): _record(m3, 2*n + 1, p, diag, coef, out, tr, state)


@jit(nopython=True, cache=True)
def _doubled_init(cplx, Vr, Vi, columns, ci, cj, diag, coef):
    """The output arrays of a doubled block, the first moment of every pair,
    <e_i|H|e_j>, read on V_1, and the moments 0 and 1 recorded"""
    P = len(ci)
    out = np.zeros((P, coef.shape[1]), dtype=np.complex128)
    tr = np.zeros(len(coef), dtype=np.complex128)
    state = np.zeros(2)
    mu1 = np.zeros(P, dtype=np.complex128)
    for p in range(P):
        mu1[p] = _entry(cplx, Vr, Vi, columns[ci[p]], cj[p])
        _record(complex(1. if diag[p] else 0.), 0, p, diag, coef, out, tr, state)
        if len(coef) > 1: _record(mu1[p], 1, p, diag, coef, out, tr, state)
    off = np.nonzero(~diag)[0]
    return out, tr, state, mu1, ci[off], cj[off]


@jit(nopython=True, parallel=True, cache=True)
def _doubled_rows(cplx, indptr, indices, d2r, d2i, columns, ci, cj, diag,
        coef, nchunk):
    """The doubled recursion of one block, its rows split among the
    threads in nchunk pieces: columns are the orbitals of the block
    columns, ci and cj the block columns of the two orbitals of every
    pair. Returns sum_n coef[n,c] mu_n for every pair and column c of
    coef, the moments summed over the pairs flagged in diag, and the
    largest modulus of a moment"""
    Vr, Vi, Wr, Wi = _start(cplx, indptr, indices, d2r, d2i, columns)
    out, tr, state, mu1, oi, oj = _doubled_init(cplx, Vr, Vi, columns, ci,
            cj, diag, coef)
    N, ncol = Vr.shape
    size = (N + nchunk - 1)//nchunk
    D = np.zeros((nchunk, 2, ncol))
    O = np.zeros((nchunk, 4, len(oi)))
    for n in range(1, (len(coef) + 1)//2):
        for t in prange(nchunk):
            dacc = np.zeros((2, ncol))
            oacc = np.zeros((4, len(oi)))
            for r in range(t*size, min(N, (t + 1)*size)):
                _step_row(r, cplx, indptr, indices, d2r, d2i, Vr, Vi, Wr, Wi)
                _products_row(r, cplx, Vr, Vi, Wr, Wi, oi, oj, dacc, oacc)
            D[t] = dacc
            O[t] = oacc
        _doubled_moments(n, D.sum(axis=0), O.sum(axis=0), ci, diag, mu1, coef,
                out, tr, state)
        Vr, Wr = Wr, Vr # V_{n+1} is now in W
        Vi, Wi = Wi, Vi
    return out, tr, (np.inf if state[1] else state[0])


@jit(nopython=True, cache=True)
def _doubled_serial(cplx, indptr, indices, d2r, d2i, columns, ci, cj, diag,
        coef):
    """_doubled_rows with the rows one after the other"""
    Vr, Vi, Wr, Wi = _start(cplx, indptr, indices, d2r, d2i, columns)
    out, tr, state, mu1, oi, oj = _doubled_init(cplx, Vr, Vi, columns, ci,
            cj, diag, coef)
    N, ncol = Vr.shape
    for n in range(1, (len(coef) + 1)//2):
        dacc = np.zeros((2, ncol))
        oacc = np.zeros((4, len(oi)))
        for r in range(N):
            _step_row(r, cplx, indptr, indices, d2r, d2i, Vr, Vi, Wr, Wi)
            _products_row(r, cplx, Vr, Vi, Wr, Wi, oi, oj, dacc, oacc)
        _doubled_moments(n, dacc, oacc, ci, diag, mu1, coef, out, tr, state)
        Vr, Wr = Wr, Vr
        Vi, Wi = Wi, Vi
    return out, tr, (np.inf if state[1] else state[0])


@jit(nopython=True, cache=True)
def _read_moments(n, cplx, Vr, Vi, rows, cidx, diag, coef, out, tr, state):
    """Moment n of every pair, the entry of its row in the column of its
    starting vector"""
    for p in range(len(rows)):
        _record(_entry(cplx, Vr, Vi, rows[p], cidx[p]), n, p, diag, coef,
                out, tr, state)


@jit(nopython=True, parallel=True, cache=True)
def _read_rows(cplx, indptr, indices, d2r, d2i, columns, rows, cidx, diag,
        coef):
    """The recursion of one block read on the rows, its rows split among
    the threads: the moment of a pair is the entry of its row in the
    column of its starting vector. Returns what _doubled_rows does"""
    Vr, Vi, Wr, Wi = _start(cplx, indptr, indices, d2r, d2i, columns)
    N = Vr.shape[0]
    out = np.zeros((len(rows), coef.shape[1]), dtype=np.complex128)
    tr = np.zeros(len(coef), dtype=np.complex128)
    state = np.zeros(2)
    _read_moments(0, cplx, Wr, Wi, rows, cidx, diag, coef, out, tr, state)
    if len(coef) > 1:
        _read_moments(1, cplx, Vr, Vi, rows, cidx, diag, coef, out, tr, state)
    for n in range(2, len(coef)):
        for r in prange(N):
            _step_row(r, cplx, indptr, indices, d2r, d2i, Vr, Vi, Wr, Wi)
        Vr, Wr = Wr, Vr
        Vi, Wi = Wi, Vi
        _read_moments(n, cplx, Vr, Vi, rows, cidx, diag, coef, out, tr, state)
    return out, tr, (np.inf if state[1] else state[0])


@jit(nopython=True, cache=True)
def _read_serial(cplx, indptr, indices, d2r, d2i, columns, rows, cidx, diag,
        coef):
    """_read_rows with the rows one after the other"""
    Vr, Vi, Wr, Wi = _start(cplx, indptr, indices, d2r, d2i, columns)
    N = Vr.shape[0]
    out = np.zeros((len(rows), coef.shape[1]), dtype=np.complex128)
    tr = np.zeros(len(coef), dtype=np.complex128)
    state = np.zeros(2)
    _read_moments(0, cplx, Wr, Wi, rows, cidx, diag, coef, out, tr, state)
    if len(coef) > 1:
        _read_moments(1, cplx, Vr, Vi, rows, cidx, diag, coef, out, tr, state)
    for n in range(2, len(coef)):
        for r in range(N):
            _step_row(r, cplx, indptr, indices, d2r, d2i, Vr, Vi, Wr, Wi)
        Vr, Wr = Wr, Vr
        Vi, Wi = Wi, Vi
        _read_moments(n, cplx, Vr, Vi, rows, cidx, diag, coef, out, tr, state)
    return out, tr, (np.inf if state[1] else state[0])


@jit(nopython=True, parallel=True, cache=True)
def _doubled_members(cplx, indptr, indices, d2r, d2i, nrow, columns, ncol,
        ci, cj, diag, npair, coef):
    """_doubled_serial for every member of a batch at once, one member per
    thread: the k-points of a small cell, or the tiles of the truncated
    recursion (kpmtk/truncation.py). Each member has arrays of its own,
    stacked and padded with zeros, and its own number of rows, block
    columns and pairs (nrow, ncol, npair)"""
    M = indptr.shape[0]
    out = np.zeros((M, ci.shape[1], coef.shape[1]), dtype=np.complex128)
    tr = np.zeros((M, len(coef)), dtype=np.complex128)
    mumax = np.zeros(M)
    for im in prange(M):
        n, c, q = nrow[im], ncol[im], npair[im]
        o, t, m = _doubled_serial(cplx[im], indptr[im, :n+1], indices[im],
                d2r[im], d2i[im], columns[im, :c], ci[im, :q], cj[im, :q],
                diag[im, :q], coef)
        out[im, :q] = o
        tr[im] = t
        mumax[im] = m
    return out, tr, mumax


@jit(nopython=True, parallel=True, cache=True)
def _read_members(cplx, indptr, indices, d2r, d2i, nrow, columns, ncol,
        rows, cidx, diag, npair, coef):
    """_read_serial for every member of a batch at once, see
    _doubled_members"""
    M = indptr.shape[0]
    out = np.zeros((M, rows.shape[1], coef.shape[1]), dtype=np.complex128)
    tr = np.zeros((M, len(coef)), dtype=np.complex128)
    mumax = np.zeros(M)
    for im in prange(M):
        n, c, q = nrow[im], ncol[im], npair[im]
        o, t, m = _read_serial(cplx[im], indptr[im, :n+1], indices[im],
                d2r[im], d2i[im], columns[im, :c], rows[im, :q],
                cidx[im, :q], diag[im, :q], coef)
        out[im, :q] = o
        tr[im] = t
        mumax[im] = m
    return out, tr, mumax


def _csr_arrays(m, dtype):
    """The CSR arrays of m for the kernels, with 2H split into real and
    imaginary parts, and whether the imaginary part is needed. A
    Hamiltonian runs on the real part only when its imaginary part is
    exactly zero: a tolerance would drop the small imaginary seed of a
    chiral or flux state in a mean-field loop and keep it from growing"""
    cplx = bool(np.any(m.data.imag))
    return (cplx, m.indptr.astype(np.int64), m.indices.astype(np.int64),
            (2.*m.data.real).astype(dtype), (2.*m.data.imag).astype(dtype))


def _run(m, block, diag, coef, dtype):
    """The contracted values, the diagonal moments summed, and the largest
    moment of the pairs of one block at one k-point, the rows split among
    the threads unless the block is too small for that to pay"""
    columns, sel, doubled, a, b = block
    cplx, indptr, indices, d2r, d2i = _csr_arrays(m, dtype)
    args = (cplx, indptr, indices, d2r, d2i, columns, a, b, diag, coef)
    small = m.shape[0]*len(columns) <= _SERIAL_BLOCK
    if doubled:
        if small: return _doubled_serial(*args)
        # the rows go to the threads in pieces, as many as the threads numba
        # has now (one with parallel.set_enabled(False)) times four, for balance
        return _doubled_rows(*args, 4*numba.get_num_threads())
    if small: return _read_serial(*args)
    return _read_rows(*args)


def _run_members(members, coef, dtype):
    """_run for every member of a batch at once, one member per thread:
    members is a list of (matrix, block, diag), and the results come back
    as a list of (values, trace, largest moment)"""
    out = []
    for doubled in (False, True):
        group = [i for (i, (_, b, _)) in enumerate(members) if b[2] == doubled]
        if not group: continue
        arrays = [_csr_arrays(members[i][0], dtype) for i in group]
        blocks = [members[i][1] for i in group]
        diags = [members[i][2] for i in group]
        def stacked(xs, kind):
            a = np.zeros((len(xs), max(1, max(len(x) for x in xs))), dtype=kind)
            for j, x in enumerate(xs): a[j, :len(x)] = x
            return a
        nrow = np.array([len(x[1]) - 1 for x in arrays])
        indptr = np.zeros((len(group), nrow.max() + 1), dtype=np.int64)
        for j, x in enumerate(arrays):
            indptr[j, :len(x[1])] = x[1]
            indptr[j, len(x[1]):] = x[1][-1] # empty padding rows
        args = (np.array([x[0] for x in arrays]), indptr,
                stacked([x[2] for x in arrays], np.int64),
                stacked([x[3] for x in arrays], dtype),
                stacked([x[4] for x in arrays], dtype), nrow,
                stacked([b[0] for b in blocks], np.int64),
                np.array([len(b[0]) for b in blocks]),
                stacked([b[3] for b in blocks], np.int64),
                stacked([b[4] for b in blocks], np.int64),
                stacked(diags, np.bool_),
                np.array([len(b[1]) for b in blocks]), coef)
        driver = _doubled_members if doubled else _read_members
        vals, tr, mm = driver(*args)
        for j, i in enumerate(group):
            out.append((i, (vals[j, :len(blocks[j][1])], tr[j], mm[j])))
    return [r for (_, r) in sorted(out, key=lambda x: x[0])]


def _batch_fits(members_n_ncol):
    """Whether one block per thread of these (rows, columns) fits in
    _MAX_BLOCK"""
    return numba.get_num_threads()*max(n*c for (n, c) in members_n_ncol) \
            <= _MAX_BLOCK


def pair_values(ms, pairs, coef, kpm_prec="double", trace=False):
    """For every k-point (ms, the matrices H(k)/scale, spectrum inside
    [-1,1]) and every pair (i,j) of pairs, sum_n coef[n] <e_i|T_n(H)|e_j>,
    as an (nk,npairs) array, with the largest modulus of any moment. A
    coef of shape (nm,nc) gives nc contractions of the same moments, as
    an (nk,npairs,nc) array, for instance a local DOS at nc energies. With
    trace=True, also the moments of the pairs with i=j summed over them,
    as an (nk,len(coef)) array, which is the trace of T_n(H) when every
    orbital has its diagonal pair.

    A cell small enough that every k-point's blocks fit in memory once per
    thread runs one k-point per thread; otherwise the k-points go one
    after the other, with the rows of each block split among the threads."""
    pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    shape = np.shape(coef)[1:] # the contractions of every pair
    coef = np.asarray(coef, dtype=np.float64).reshape(len(coef), -1)
    dtype = _DTYPES[kpm_prec]
    ms = [csr_matrix(m) for m in ms]
    N = ms[0].shape[0]
    out = np.zeros((len(ms), len(pairs), coef.shape[1]), dtype=np.complex128)
    tr = np.zeros((len(ms), len(coef)), dtype=np.complex128)
    mumax = 0.
    blocks = _plan(pairs, N, max(m.nnz for m in ms)) if len(pairs) else []
    per_k = (len(ms) > 1 and len(blocks) == 1
            and _batch_fits([(N, len(blocks[0][0]))]))
    for block in blocks:
        sel = block[1]
        diag = pairs[sel, 0] == pairs[sel, 1]
        if per_k:
            results = _run_members([(m, block, diag) for m in ms], coef, dtype)
        else: results = [_run(m, block, diag, coef, dtype) for m in ms]
        for ik, (vals, t, mm) in enumerate(results):
            out[ik, sel] = vals
            tr[ik] += t
            mumax = max(mumax, float(mm))
    out = out.reshape((len(ms), len(pairs)) + shape)
    if trace: return out, mumax, tr
    return out, mumax


def pair_values_batch(problems, coef, kpm_prec="double", pad=None):
    """pair_values with trace=True for several independent problems, each
    a list of matrices (one per k-point, the same number for every
    problem) and its pairs: the tiles of the truncated recursion
    (kpmtk/truncation.py). Every block of every problem and k-point runs in
    a thread of its own when they fit in memory once per thread, which
    saves the problems the cost of splitting their few rows among the
    threads; otherwise each problem goes through pair_values. Returns a
    list of (values, largest moment, trace), one per problem, the values
    of the shape pair_values gives them. pad, the sizes the card pads to,
    is not needed here"""
    shape = np.shape(coef)[1:]
    coef = np.asarray(coef, dtype=np.float64).reshape(len(coef), -1)
    dtype = _DTYPES[kpm_prec]
    members, where = [], []
    plans = []
    for ip, (ms, pairs) in enumerate(problems):
        pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
        ms = [csr_matrix(m) for m in ms]
        blocks = _plan(pairs, ms[0].shape[0], max(m.nnz for m in ms)) \
                if len(pairs) else []
        plans.append((ms, pairs, blocks))
        for block in blocks:
            diag = pairs[block[1], 0] == pairs[block[1], 1]
            for ik, m in enumerate(ms):
                members.append((m, block, diag))
                where.append((ip, ik, block[1]))
    if not _batch_fits([(m.shape[0], len(b[0])) for (m, b, _) in members]
            or [(0, 0)]):
        return [pair_values(ms, pairs, coef.reshape((len(coef),) + shape),
            kpm_prec=kpm_prec, trace=True) for (ms, pairs, _) in plans]
    results = [[np.zeros((len(ms), len(pairs)) + shape, dtype=np.complex128),
        0., np.zeros((len(ms), len(coef)), dtype=np.complex128)]
        for (ms, pairs, _) in plans]
    if members:
        for (ip, ik, sel), (vals, t, mm) in zip(where,
                _run_members(members, coef, dtype)):
            results[ip][0][ik, sel] = vals.reshape((len(sel),) + shape)
            results[ip][2][ik] += t
            results[ip][1] = max(results[ip][1], float(mm))
    return [tuple(r) for r in results]


def trace_moments(ms, nm, kpm_prec="double"):
    """For every k-point (ms, the matrices H(k)/scale), the first nm
    Chebyshev moments averaged over the orbitals, (1/N) Tr T_n(H), as an
    (nk,nm) array, with the largest modulus of any moment"""
    N = ms[0].shape[0]
    pairs = np.stack([np.arange(N), np.arange(N)], axis=1)
    _, mumax, tr = pair_values(ms, pairs, np.zeros(nm), kpm_prec=kpm_prec,
            trace=True)
    return tr/N, mumax
