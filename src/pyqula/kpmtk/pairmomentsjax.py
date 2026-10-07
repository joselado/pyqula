from .. import gpu
# the package-wide CPU/GPU switch decides where this module's kernels run
# (pyqula/gpu.py); applying it here means importing this module does not
# leave the placement to jax's own default
gpu.apply()

# Chebyshev moments <e_i|T_n(H)|e_j> of many (i,j) pairs and k-points at
# once, for the KPM mean field on the GPU (kpmtk/densitymatrix_kpm.py); on
# the CPU the same two entry points are kpmtk/pairmomentsnumba.py.
#
# The recursion runs on a dense block whose columns are the distinct
# starting vectors e_j, V_{n+1} = 2 H V_n - V_{n-1}, every k-point of the
# mesh at once, and each step reads the entries of every requested pair
# with one gather, so a pair costs a gather instead of a recursion of its
# own; in double precision a block whose pairs are local is doubled
# instead, two moments per step from inner products of its columns, as in
# pairmomentsnumba. The tiles of the truncated recursion
# (kpmtk/truncation.py) go in one batch (pair_values_batch). H is stored
# in ELL form, the entries of each row padded to a common
# width, which makes H@V a sum of whole-row gathers. jax's BCOO product with
# a dense block lowers to a scatter-add instead, whose atomics cost the same
# in either precision, and was 3 to 6 times slower on a consumer card; the
# measurements are in future_development/gpu_kpm_mean_field.md.

from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
from scipy.sparse import issparse, csr_matrix

_DTYPES = {"single": jnp.complex64, "double": jnp.complex128}

# the most block entries (k-points x orbitals x starting columns, plus the
# moments kept per pair) handled in one call; three blocks are live at a
# time, so this is about 1.6 GB in double precision
_MAX_BLOCK = 2**25

# the ELL width last used for a matrix of each dimension, see _ell
_ELL_WIDTH = dict()


def get_precision_names():
    """Every precision the kernels of this module accept"""
    return sorted(_DTYPES)


def _stable_width(key, K):
    """The ELL width for a pattern of width K, which keeps the one last
    used for the same key when that is slightly wider, see _ell"""
    last = _ELL_WIDTH.get(key, 0)
    if K <= last <= K + max(2, K//4): K = last
    _ELL_WIDTH[key] = K
    return K


def _ell(ms, memo=True):
    """ELL form of the matrices ms (one per k-point, all N x N): the
    pattern is the union over k, so that its width, and with it the
    compiled kernel, does not change with k. Returns data (nk,N,K) and
    cols (N,K); a padding slot points at its own row with zero weight.

    In a mean-field loop the Hamiltonian changes every iteration, and an
    entry that passes through zero drops out of its sparse form, which
    would narrow the width by one and compile the kernel again; so a
    width slightly below the last one used for the same dimension takes
    that one instead, and only a much narrower pattern, a different
    Hamiltonian, gets its own; memo=False leaves the width to the pattern,
    for a caller that pads it itself."""
    N = ms[0].shape[0]
    keys = []
    for m in ms:
        if issparse(m):
            c = m.tocoo()
            nz = c.data != 0
            r, cc = c.row[nz], c.col[nz]
        else: r, cc = np.nonzero(np.asarray(m))
        keys.append(r.astype(np.int64)*N + cc)
    keys = np.unique(np.concatenate(keys))
    rows, cols_nz = keys//N, keys % N # sorted by row, then column
    counts = np.bincount(rows, minlength=N)
    K = max(1, int(counts.max()) if len(rows) else 1)
    if memo: K = _stable_width(N, K)
    first = np.concatenate([[0], np.cumsum(counts)[:-1]])
    slot = np.arange(len(rows)) - first[rows]
    cols = np.tile(np.arange(N)[:, None], (1, K))
    cols[rows, slot] = cols_nz
    data = np.zeros((len(ms), N, K), dtype=np.complex128)
    for ik, m in enumerate(ms):
        if issparse(m): vals = np.asarray(csr_matrix(m)[rows, cols_nz]).ravel()
        else: vals = np.asarray(m)[rows, cols_nz]
        data[ik, rows, slot] = vals
    return data, cols


def _hv(data, cols, a):
    """H a for H in ELL form, a sum of whole-row gathers"""
    out = data[:, 0, None]*a[cols[:, 0]]
    for s in range(1, cols.shape[1]):
        out = out + data[:, s, None]*a[cols[:, s]]
    return out


def _read(data, cols, V0, rows, cj, nm):
    """Moments (nm,npairs) of one k-point read on the rows: data/cols the
    ELL form of H, V0 the block of starting columns, rows and cj the row
    and the block column of the starting vector of every pair"""
    V1 = _hv(data, cols, V0)
    mus = jnp.zeros((nm, rows.shape[0]), V0.dtype)
    mus = mus.at[0].set(V0[rows, cj]).at[1].set(V1[rows, cj])
    def body(i, c):
        am, a, mus = c
        ap = 2*_hv(data, cols, a) - am
        return a, ap, mus.at[i].set(ap[rows, cj])
    _, _, mus = jax.lax.fori_loop(2, nm, body, (V0, V1, mus))
    return mus


def _doubled(data, cols, V0, rows, ci, cj, diag, nm):
    """Moments (nm,npairs) of one k-point from the doubling,
    T_{2n} = 2 T_n T_n - T_0 and T_{2n+1} = 2 T_{n+1} T_n - T_1: ci and
    cj are the block columns of the two orbitals of every pair, rows the
    orbital of the first, and diag says which pairs are diagonal. The
    inner products are summed in the precision of the recursion, by
    XLA's tree reduction"""
    V1 = _hv(data, cols, V0)
    mu1 = V1[rows, cj] # <e_i|H|e_j>
    d = diag.astype(V0.dtype)
    nh = (nm + 1)//2
    mus = jnp.zeros((2*nh, rows.shape[0]), V0.dtype)
    mus = mus.at[0].set(d).at[1].set(mu1)
    def body(n, c):
        am, a, mus = c
        ap = 2*_hv(data, cols, a) - am
        x = a[:, cj]
        A = jnp.sum(jnp.conj(a[:, ci])*x, axis=0)
        B = jnp.sum(jnp.conj(ap[:, ci])*x, axis=0)
        mus = mus.at[2*n].set(2*A - d).at[2*n+1].set(2*B - mu1)
        return a, ap, mus
    _, _, mus = jax.lax.fori_loop(1, nh, body, (V0, V1, mus))
    return mus[:nm]


def _one(d, cols, V0, rows, ci, cj, diag, weight, nm, doubled, coef):
    """sum_n coef[n] mu_n of every pair (for every column of coef when it
    has two axes), the largest modulus of a moment, and the moments
    weighted by weight and summed over the pairs (the trace, for the
    Fermi level); the sums in double precision whatever the precision of
    the recursion"""
    if doubled: mus = _doubled(d, cols, V0, rows, ci, cj, diag, nm)
    else: mus = _read(d, cols, V0, rows, cj, nm)
    mus = mus.astype(jnp.complex128)
    return (jnp.tensordot(mus, coef, axes=(0, 0)), jnp.max(jnp.abs(mus)),
            mus @ weight)


@partial(jax.jit, static_argnums=(8, 9))
def _contracted(data, cols, V0, rows, ci, cj, diag, weight, nm, doubled,
        coef):
    """_one for every k-point of data (nk,N,K), the rest shared"""
    return jax.vmap(lambda d: _one(d, cols, V0, rows, ci, cj, diag, weight,
        nm, doubled, coef))(data)


@partial(jax.jit, static_argnums=(8, 9))
def _contracted_each(data, cols, V0, rows, ci, cj, diag, weight, nm,
        doubled, coef):
    """_one for every member of a batch, each with arrays of its own, all
    of the same shapes: the tiles of the truncated recursion"""
    return jax.vmap(lambda *a: _one(*a, nm, doubled, coef))(data, cols, V0,
            rows, ci, cj, diag, weight)


# the cost of the inner products of a doubled pair, per row and step, in
# units of one stored entry of H times one column of the block, the same
# constants as pairmomentsnumba's; doubling is left out in single
# precision, see _plan
_DIAGONAL_COST = 1.
_PAIR_COST = 4.


def _plan(pairs, N, K, nm, kpm_prec):
    """The blocks of starting columns: their orbitals, the indexes of
    their pairs in pairs, and whether they are doubled, with for every
    pair its row, the block columns of its two orbitals, and whether it is
    diagonal. The blocks are as equal as _MAX_BLOCK allows, so that their
    padding is small (padding a last block of 220 columns to a first one of
    3236 made a 3456-orbital Nambu island 1.7 times slower). A block is
    doubled when the cost model of pairmomentsnumba says it pays, with the
    orbitals of its pairs that are not starting columns added to it, and
    only in double precision, where it took 0.68 to 0.73 of the time on a
    consumer card: in single precision it was no faster there, and its
    inner products, summed over every orbital in single precision, moved
    the density matrix by 5e-8 against 1e-8 read on the rows
    (future_development/sparse_kpm_mean_field.md)"""
    order = np.argsort(pairs[:, 1], kind="stable")
    pj = pairs[order, 1]
    starts = np.unique(pj)
    ncol = max(1, min(len(starts), _MAX_BLOCK//(3*N)))
    ncol = -(-len(starts)//(-(-len(starts)//ncol))) # equal blocks
    blocks = []
    for c0 in range(0, len(starts), ncol):
        chunk = starts[c0:c0+ncol]
        sel = order[np.searchsorted(pj, chunk[0], side="left"):
                np.searchsorted(pj, chunk[-1], side="right")]
        rows, cols = pairs[sel, 0], pairs[sel, 1]
        diag = rows == cols
        both = np.union1d(chunk, rows)
        cost = (K*len(both) + _DIAGONAL_COST*len(both)
                + _PAIR_COST*(len(sel) - np.count_nonzero(diag)))
        if (kpm_prec == "double" and cost < 2*K*len(chunk)
                and len(both) <= 2*ncol):
            blocks.append(dict(columns=both, sel=sel, doubled=True,
                rows=rows, ci=np.searchsorted(both, rows),
                cj=np.searchsorted(both, cols), diag=diag))
        else:
            blocks.append(dict(columns=chunk, sel=sel, doubled=False,
                rows=rows, ci=np.zeros(len(sel), dtype=np.int64),
                cj=np.searchsorted(chunk, cols), diag=diag))
    return blocks


def _padded(block, width, npad, N):
    """The arrays of a block padded to width columns and npad pairs: the
    padding columns are zero vectors, which stay zero, and a padding pair
    is a diagonal one on the first column with no weight in the trace, so
    that its moments are bounded as a real one's and are dropped"""
    n = len(block["sel"])
    rows = np.zeros(npad, dtype=np.int32); rows[:n] = block["rows"]
    ci = np.zeros(npad, dtype=np.int32); ci[:n] = block["ci"]
    cj = np.zeros(npad, dtype=np.int32); cj[:n] = block["cj"]
    diag = np.ones(npad, dtype=bool); diag[:n] = block["diag"]
    weight = np.zeros(npad); weight[:n] = block["diag"]
    rows[n:] = block["columns"][0]
    V0 = np.zeros((N, width))
    V0[block["columns"], np.arange(len(block["columns"]))] = 1.
    return V0, rows, ci, cj, diag, weight


def _groups(n, size):
    """n items in groups of at most size, as equal as possible"""
    size = max(1, min(n, size))
    size = -(-n//(-(-n//size)))
    return size, -(-n//size)*size


def pair_values(ms, pairs, coef, kpm_prec="double", trace=False):
    """For every k-point (ms, the matrices H(k)/scale, spectrum inside
    [-1,1]) and every pair (i,j) of pairs, sum_n coef[n] <e_i|T_n(H)|e_j>,
    as an (nk,npairs) array, with the largest modulus of any moment. A
    coef of shape (nm,nc) gives nc contractions of the same moments, as
    an (nk,npairs,nc) array. With trace=True, also the moments of the
    pairs with i=j summed over them, as an (nk,len(coef)) array, which is
    the trace of T_n(H) when every orbital has its diagonal pair.

    The blocks of starting columns and the k-points are split into calls
    of at most _MAX_BLOCK entries, and every call of a kind (doubled, or
    read on the rows) has the same shapes, so that the kernel compiles once
    per Hamiltonian rather than once per call: the blocks are padded to the
    widest and the pairs to the most of their kind, and the k-points to a
    whole number of calls with zero matrices, and what the padding computes
    is dropped. Without it the number of pairs changed from one block to
    the next, and 400 orbitals spent 23 s in 19 compilations, at 3200
    orbitals 2 min each."""
    pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    nm = len(coef)
    shape = np.shape(coef)[1:] # the contractions of every pair
    nc = int(np.prod(shape))
    coef = jnp.asarray(coef, dtype=jnp.float64)
    dtype = _DTYPES[kpm_prec]
    nk = len(ms)
    out = np.zeros((nk, len(pairs)) + shape, dtype=np.complex128)
    tr = np.zeros((nk, nm), dtype=np.complex128)
    mumax = [0.] # reduced with np.max, which keeps a NaN where max() may not
    if len(pairs) == 0:
        return (out, 0., tr) if trace else (out, 0.)
    N = ms[0].shape[0]
    data, cols = _ell(ms)
    blocks = _plan(pairs, N, cols.shape[1], nm, kpm_prec)
    cols = jnp.asarray(cols, dtype=jnp.int32)
    for doubled in (False, True):
        group = [b for b in blocks if b["doubled"] == doubled]
        if not group: continue
        width = max(len(b["columns"]) for b in group)
        npad = max(len(b["sel"]) for b in group)
        nkc, nkpad = _groups(nk, _MAX_BLOCK//(3*N*width + (nm + nc)*npad))
        d = data
        if nkpad > nk: d = np.concatenate([data, np.zeros((nkpad - nk,)
            + data.shape[1:], dtype=data.dtype)])
        for b in group:
            V0, rows, ci, cj, diag, weight = _padded(b, width, npad, N)
            args = [jnp.asarray(V0, dtype=dtype)] + [jnp.asarray(x) for x
                    in (rows, ci, cj, diag)] + [jnp.asarray(weight,
                    dtype=jnp.complex128)]
            n = len(b["sel"])
            for k0 in range(0, nkpad, nkc):
                vals, m, t = _contracted(jnp.asarray(d[k0:k0+nkc],
                    dtype=dtype), cols, *args, nm, doubled, coef)
                nreal = min(nk, k0 + nkc) - k0
                out[k0:k0+nreal, b["sel"]] = np.asarray(vals)[:nreal, :n]
                tr[k0:k0+nreal] += np.asarray(t)[:nreal]
                mumax.append(np.max(np.asarray(m)[:nreal]))
    if trace: return out, float(np.max(mumax)), tr
    return out, float(np.max(mumax))


def pair_values_batch(problems, coef, kpm_prec="double", pad=None):
    """pair_values with trace=True for several independent problems at
    once, each a list of matrices (one per k-point, the same number for
    every problem) and its pairs: the tiles of the truncated recursion
    (kpmtk/truncation.py). Every block of every problem and k-point is a
    member of one batch, padded to the largest dimension, ELL width, block
    and number of pairs of its kind, so that the kernel compiles once
    rather than once per tile. pad, a dictionary of the least dimension N,
    ELL width K, block width and number of pairs npairs, makes the shapes
    the same across calls, for problems handed over in several groups.
    Returns a list of (values, largest moment, trace), one per problem"""
    pad = dict() if pad is None else pad
    nm = len(coef)
    shape = np.shape(coef)[1:] # the contractions of every pair
    nc = int(np.prod(shape))
    coef = jnp.asarray(coef, dtype=jnp.float64)
    dtype = _DTYPES[kpm_prec]
    members = [] # (problem, block, k-point)
    ells, plans = [], []
    for ip, (ms, pairs) in enumerate(problems):
        pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
        data, cols = _ell(ms, memo=False)
        ells.append((data, cols))
        plans.append(_plan(pairs, ms[0].shape[0], cols.shape[1], nm,
            kpm_prec) if len(pairs) else [])
        for ib in range(len(plans[-1])):
            for ik in range(len(ms)): members.append((ip, ib, ik))
    nk = len(problems[0][0]) if problems else 0
    results = [[np.zeros((nk, len(np.asarray(p).reshape(-1, 2))) + shape,
        dtype=np.complex128), 0., np.zeros((nk, nm), dtype=np.complex128)]
        for (_, p) in problems]
    if not members: return [tuple(r) for r in results]
    # every member padded to the largest dimension and ELL width, rounded
    # up so that small changes from one iteration to the next keep the shapes
    N = max(pad.get("N", 0), max(e[0].shape[1] for e in ells))
    N = -(-N//32)*32
    K = _stable_width(("batch", N), max(pad.get("K", 0),
        max(e[1].shape[1] for e in ells)))
    for doubled in (False, True):
        group = [m for m in members if plans[m[0]][m[1]]["doubled"] == doubled]
        if not group: continue
        blocks = [plans[ip][ib] for (ip, ib, _) in group]
        width = max([len(b["columns"]) for b in blocks]
                + [pad.get("width", 0)])
        npad = max([len(b["sel"]) for b in blocks] + [pad.get("npairs", 0)])
        size, total = _groups(len(group),
                _MAX_BLOCK//(3*N*width + (nm + nc)*npad))
        group = group + [None]*(total - len(group))
        for g0 in range(0, total, size):
            arrays = []
            for m in group[g0:g0+size]:
                if m is None: m = group[0] # padding member, dropped
                ip, ib, ik = m
                data, cols = ells[ip]
                n0, k0 = cols.shape
                d = np.zeros((N, K), dtype=np.complex128)
                d[:n0, :k0] = data[ik]
                c = np.tile(np.arange(N)[:, None], (1, K))
                c[:n0, :k0] = cols
                arrays.append((d, c) + _padded(plans[ip][ib], width, npad, N))
            stacked = [np.stack([a[i] for a in arrays]) for i in range(8)]
            vals, mm, t = _contracted_each(
                    jnp.asarray(stacked[0], dtype=dtype),
                    jnp.asarray(stacked[1], dtype=jnp.int32),
                    jnp.asarray(stacked[2], dtype=dtype),
                    *[jnp.asarray(x) for x in stacked[3:7]],
                    jnp.asarray(stacked[7], dtype=jnp.complex128),
                    nm, doubled, coef)
            vals, mm, t = np.asarray(vals), np.asarray(mm), np.asarray(t)
            for j, m in enumerate(group[g0:g0+size]):
                if m is None: continue
                ip, ib, ik = m
                b = plans[ip][ib]
                results[ip][0][ik, b["sel"]] = vals[j, :len(b["sel"])]
                results[ip][2][ik] += t[j]
                results[ip][1] = max(results[ip][1], float(mm[j]))
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
