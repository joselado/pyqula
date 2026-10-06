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
# own. H is stored in ELL form, the entries of each row padded to a common
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


def _ell(ms):
    """ELL form of the matrices ms (one per k-point, all N x N): the
    pattern is the union over k, so that its width, and with it the
    compiled kernel, does not change with k. Returns data (nk,N,K) and
    cols (N,K); a padding slot points at its own row with zero weight.

    In a mean-field loop the Hamiltonian changes every iteration, and an
    entry that passes through zero drops out of its sparse form, which
    would narrow the width by one and compile the kernel again; so a
    width slightly below the last one used for the same dimension takes
    that one instead, and only a much narrower pattern, a different
    Hamiltonian, gets its own."""
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
    last = _ELL_WIDTH.get(N, 0)
    if K <= last <= K + max(2, K//4): K = last # see above
    _ELL_WIDTH[N] = K
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


def _recursion(data, cols, V0, rows, cidx, nm):
    """Moments (nm,npairs) of one k-point: data/cols the ELL form of H,
    V0 the block of starting columns, rows/cidx the row and the block
    column of every pair"""
    def hv(a):
        out = data[:, 0, None]*a[cols[:, 0]]
        for s in range(1, cols.shape[1]):
            out = out + data[:, s, None]*a[cols[:, s]]
        return out
    V1 = hv(V0)
    mus = jnp.zeros((nm, rows.shape[0]), V0.dtype)
    mus = mus.at[0].set(V0[rows, cidx]).at[1].set(V1[rows, cidx])
    def body(i, c):
        am, a, mus = c
        ap = 2*hv(a) - am
        return a, ap, mus.at[i].set(ap[rows, cidx])
    _, _, mus = jax.lax.fori_loop(2, nm, body, (V0, V1, mus))
    return mus


@partial(jax.jit, static_argnums=(5,))
def _contracted(data, cols, V0, rows, cidx, nm, coef, diag):
    """Per k-point, sum_n coef[n]*mus[n] for every pair, the largest
    modulus of a moment (for the scale guard), and the moments summed over
    the pairs whose weight in diag is one, the diagonal ones (the trace,
    for the Fermi level). The sums are done in double precision whatever
    the precision of the recursion, so that a single precision run loses
    only what the recursion itself loses."""
    def one(d):
        mus = _recursion(d, cols, V0, rows, cidx, nm).astype(jnp.complex128)
        return coef @ mus, jnp.max(jnp.abs(mus)), mus @ diag
    return jax.vmap(one)(data)


def _blocks(ms, pairs, nm, kpm_prec):
    """Split the pairs by starting column and the k-points so that no call
    exceeds _MAX_BLOCK, and yield, per call, the device arrays, the weight
    of every pair in the trace, and where its results go: the k-points,
    how many of them are real, and the pairs.

    Every call has the same shapes, so that the kernel compiles once per
    Hamiltonian rather than once per call: the last block of starting
    columns is padded with zero columns, which stay zero, the pairs of
    every block are padded to the most any block has with pairs on the
    first row and column, and the k-points to a whole number of calls with
    zero matrices, and what the padding computes is dropped. The blocks
    and the groups of k-points are made as equal as the budget allows, so
    that the padding is at most one column or k-point per call: padding a
    last block of 220 columns to a first one of 3236 made a 3456-orbital
    Nambu island 1.7 times slower on the card. Without it the
    number of pairs changes from one block to the next, and 400 orbitals
    spent 23 s in 19 compilations, at 3200 orbitals 2 min each."""
    dtype = _DTYPES[kpm_prec]
    N = ms[0].shape[0]
    data, cols = _ell(ms)
    cols = jnp.asarray(cols, dtype=jnp.int32)
    order = np.argsort(pairs[:, 1], kind="stable")
    pj = pairs[order, 1]
    starts = np.unique(pj)
    ncol = max(1, min(len(starts), _MAX_BLOCK//(3*N)))
    ncol = -(-len(starts)//(-(-len(starts)//ncol))) # equal blocks, little padding
    chunks = [starts[c0:c0+ncol] for c0 in range(0, len(starts), ncol)]
    sels = [order[np.searchsorted(pj, c[0], side="left"):
        np.searchsorted(pj, c[-1], side="right")] for c in chunks]
    npad = max(len(sel) for sel in sels)
    nk = len(ms)
    nkc = min(nk, max(1, _MAX_BLOCK//(3*N*ncol + nm*npad)))
    nkc = -(-nk//(-(-nk//nkc))) # equal groups of k-points, see ncol
    nkpad = -(-nk//nkc)*nkc
    if nkpad > nk:
        data = np.concatenate([data, np.zeros((nkpad - nk,) + data.shape[1:],
            dtype=data.dtype)])
    for chunk, sel in zip(chunks, sels):
        rows = np.zeros(npad, dtype=np.int32)
        cidx = np.zeros(npad, dtype=np.int32)
        diag = np.zeros(npad)
        rows[:len(sel)] = pairs[sel, 0]
        cidx[:len(sel)] = np.searchsorted(chunk, pairs[sel, 1])
        diag[:len(sel)] = pairs[sel, 0] == pairs[sel, 1]
        V0 = jnp.zeros((N, ncol), dtype)
        V0 = V0.at[jnp.asarray(chunk), jnp.arange(len(chunk))].set(1.)
        rows, cidx = jnp.asarray(rows), jnp.asarray(cidx)
        diag = jnp.asarray(diag, dtype=jnp.complex128)
        for k0 in range(0, nkpad, nkc):
            d = jnp.asarray(data[k0:k0+nkc], dtype=dtype)
            yield (d, cols, V0, rows, cidx), diag, \
                    slice(k0, min(nk, k0+nkc)), min(nk, k0+nkc) - k0, sel


def pair_values(ms, pairs, coef, kpm_prec="double", trace=False):
    """For every k-point (ms, the matrices H(k)/scale, spectrum inside
    [-1,1]) and every pair (i,j) of pairs, sum_n coef[n] <e_i|T_n(H)|e_j>,
    as an (nk,npairs) array, with the largest modulus of any moment. With
    trace=True, also the moments of the pairs with i=j summed over them,
    as an (nk,len(coef)) array, which is the trace of T_n(H) when every
    orbital has its diagonal pair"""
    pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    coef = jnp.asarray(coef, dtype=jnp.float64)
    out = np.zeros((len(ms), len(pairs)), dtype=np.complex128)
    tr = np.zeros((len(ms), len(coef)), dtype=np.complex128)
    mumax = [0.] # reduced with np.max, which keeps a NaN where max() may not
    if len(pairs):
        for args, diag, ks, nreal, sel in _blocks(ms, pairs, len(coef), kpm_prec):
            vals, m, t = _contracted(*args, len(coef), coef, diag)
            out[ks, sel] = np.asarray(vals)[:nreal, :len(sel)]
            tr[ks] += np.asarray(t)[:nreal]
            mumax.append(np.max(np.asarray(m)[:nreal]))
    if trace: return out, float(np.max(mumax)), tr
    return out, float(np.max(mumax))


def trace_moments(ms, nm, kpm_prec="double"):
    """For every k-point (ms, the matrices H(k)/scale), the first nm
    Chebyshev moments averaged over the orbitals, (1/N) Tr T_n(H), as an
    (nk,nm) array, with the largest modulus of any moment"""
    N = ms[0].shape[0]
    pairs = np.stack([np.arange(N), np.arange(N)], axis=1)
    _, mumax, tr = pair_values(ms, pairs, np.zeros(nm), kpm_prec=kpm_prec,
            trace=True)
    return tr/N, mumax
