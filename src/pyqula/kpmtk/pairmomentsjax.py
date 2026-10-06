from .. import gpu
# the package-wide CPU/GPU switch decides where this module's kernels run
# (pyqula/gpu.py); applying it here means importing this module does not
# leave the placement to jax's own default
gpu.apply()

# Chebyshev moments <e_i|T_n(H)|e_j> of many (i,j) pairs and k-points at
# once, for the KPM mean field (kpmtk/densitymatrix_kpm.py).
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


def get_precision_names():
    """Every precision the kernels of this module accept"""
    return sorted(_DTYPES)


def _ell(ms):
    """ELL form of the matrices ms (one per k-point, all N x N): the
    pattern is the union over k, so that its width, and with it the
    compiled kernel, does not change with k. Returns data (nk,N,K) and
    cols (N,K); a padding slot points at its own row with zero weight."""
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
def _contracted(data, cols, V0, rows, cidx, nm, coef):
    """Per k-point, sum_n coef[n]*mus[n] for every pair, and the largest
    modulus of a moment (for the scale guard). The sum is done in double
    precision whatever the precision of the recursion, so that a single
    precision run loses only what the recursion itself loses."""
    def one(d):
        mus = _recursion(d, cols, V0, rows, cidx, nm)
        return coef @ mus.astype(jnp.complex128), jnp.max(jnp.abs(mus))
    return jax.vmap(one)(data)


@partial(jax.jit, static_argnums=(5,))
def _summed(data, cols, V0, rows, cidx, nm):
    """Per k-point, the moments summed over the pairs (in double
    precision), and the largest modulus of a moment"""
    def one(d):
        mus = _recursion(d, cols, V0, rows, cidx, nm)
        return jnp.sum(mus.astype(jnp.complex128), axis=1), jnp.max(jnp.abs(mus))
    return jax.vmap(one)(data)


def _blocks(ms, pairs, nm, kpm_prec):
    """Split the pairs by starting column and the k-points so that no call
    exceeds _MAX_BLOCK, and yield, per call, the device arrays and where
    its results go"""
    dtype = _DTYPES[kpm_prec]
    N = ms[0].shape[0]
    data, cols = _ell(ms)
    cols = jnp.asarray(cols, dtype=jnp.int32)
    starts = np.unique(pairs[:, 1])
    ncol = max(1, min(len(starts), _MAX_BLOCK//(3*N)))
    for c0 in range(0, len(starts), ncol):
        chunk = starts[c0:c0+ncol]
        sel = np.nonzero(np.isin(pairs[:, 1], chunk))[0]
        cidx = jnp.asarray(np.searchsorted(chunk, pairs[sel, 1]), dtype=jnp.int32)
        rows = jnp.asarray(pairs[sel, 0], dtype=jnp.int32)
        V0 = jnp.zeros((N, len(chunk)), dtype)
        V0 = V0.at[jnp.asarray(chunk), jnp.arange(len(chunk))].set(1.)
        nkc = max(1, _MAX_BLOCK//(3*N*len(chunk) + nm*len(sel)))
        for k0 in range(0, len(ms), nkc):
            d = jnp.asarray(data[k0:k0+nkc], dtype=dtype)
            yield (d, cols, V0, rows, cidx), slice(k0, k0+nkc), sel


def pair_values(ms, pairs, coef, kpm_prec="double"):
    """For every k-point (ms, the matrices H(k)/scale, spectrum inside
    [-1,1]) and every pair (i,j) of pairs, sum_n coef[n] <e_i|T_n(H)|e_j>,
    as an (nk,npairs) array, with the largest modulus of any moment"""
    pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    coef = jnp.asarray(coef, dtype=jnp.float64)
    out = np.zeros((len(ms), len(pairs)), dtype=np.complex128)
    mumax = [] # reduced with np.max, which keeps a NaN where max() may not
    for args, ks, sel in _blocks(ms, pairs, len(coef), kpm_prec):
        vals, m = _contracted(*args, len(coef), coef)
        out[ks, sel] = np.asarray(vals)
        mumax.append(np.max(np.asarray(m)))
    return out, float(np.max(mumax))


def trace_moments(ms, nm, kpm_prec="double"):
    """For every k-point (ms, the matrices H(k)/scale), the first nm
    Chebyshev moments averaged over the orbitals, (1/N) Tr T_n(H), as an
    (nk,nm) array, with the largest modulus of any moment"""
    N = ms[0].shape[0]
    pairs = np.stack([np.arange(N), np.arange(N)], axis=1)
    out = np.zeros((len(ms), nm), dtype=np.complex128)
    mumax = [] # see pair_values
    for args, ks, sel in _blocks(ms, pairs, nm, kpm_prec):
        sums, m = _summed(*args, nm)
        out[ks] += np.asarray(sums)
        mumax.append(np.max(np.asarray(m)))
    return out/N, float(np.max(mumax))
