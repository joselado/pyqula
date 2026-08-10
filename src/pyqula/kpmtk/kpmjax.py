def is_gpu_available():
    import jax
    try:
        jax.devices("gpu")
        return True
    except Exception: return False


if is_gpu_available(): # GPU available
    print("GPU available for KPM")
    pass
else: # use the CPU
    print("GPU is NOT available for KPM, using the CPU")
    import os
    os.environ['JAX_PLATFORMS'] = 'cpu'

import numpy as np
import jax
jax.config.update("jax_enable_x64",True) # allow float64/complex128 (JAX
# defaults to 32 bit precision, which would silently truncate "double"
# precision requests down to "single")
import jax.numpy as jnp
from jax import jit
from jax.experimental import sparse


def _kpm_moments_sparse(v,m,n):
    """Chebyshev moments of v under the sparse BCOO matrix m, using the
    Chebyshev recursion relations. Works for real or complex, single or
    double precision input -- the dtype is inherited from v (and must
    match the dtype of m)."""
    mus = jnp.zeros(2*n,dtype=v.dtype) # empty array for the moments
    am = v # zero vector
    a = m@v # vector number 1
    bk = jnp.sum(jnp.conjugate(v)*v)
    bk1 = jnp.sum(jnp.conjugate(a)*v)
    mus = mus.at[0].set(bk) # mu0
    mus = mus.at[1].set(bk1) # mu1

    def body(i,val):
        mus,am,a = val
        ap = 2*(m@a) - am # recursion relation
        bk = jnp.sum(jnp.conjugate(a)*a)
        bk1 = jnp.sum(jnp.conjugate(ap)*a)
        mus = mus.at[2*i].set(2.*bk)
        mus = mus.at[2*i+1].set(2.*bk1)
        return mus,a,ap

    mus,_,_ = jax.lax.fori_loop(1,n,body,(mus,am,a))

    mu0 = mus[0] # first
    mu1 = mus[1] # second
    mus = mus.at[2::2].add(-mu0)
    mus = mus.at[3::2].add(-mu1)
    return mus


_kpm_moments_sparse_jit = jit(_kpm_moments_sparse,static_argnums=(2,))


# A plain jax.vmap over the whole batch materializes nvec copies of the
# per-vector recursion state (am/a/ap/mus) on-device at once. That's fine
# for small batches (e.g. a handful of random tries) but a full-space trace
# (kpm.full_trace/full_trace_A -- one basis vector per site) passes
# nvec=nsites, which for a densedimension-sized system (limits.py, 10000)
# would try to allocate on the order of nvec*nsites elements and likely
# exceed real GPU memory. jax.lax.map(...,batch_size=k) runs the same
# vmapped computation in fixed-size chunks of k vectors at a time instead,
# bounding device memory independent of nvec.
_GPU_BATCH_CHUNK = 256


def _kpm_moments_sparse_batch(vs,m,n,batch_size):
    """vs has a leading batch axis (nvec,nsites); m (the same sparse matrix
    for every vector) and n are shared/unbatched, so each chunk broadcasts
    them across its sub-batch instead of mapping over them."""
    return jax.lax.map(lambda v: _kpm_moments_sparse(v,m,n),vs,
            batch_size=batch_size)


_kpm_moments_sparse_batch_jit = jit(_kpm_moments_sparse_batch,static_argnums=(2,3))


def kpm_moments_gpu(v0,data0,row0,col0,n=100):
    """Chebyshev moments computed with JAX, on the GPU if one is available
    and otherwise transparently on the CPU. v0 and data0 fix both the
    precision (float32/float64 for real input, complex64/complex128 for
    complex input) and whether the real or complex code path is used --
    both dtype families are supported in single and double precision."""
    dtype = v0.dtype
    nd = v0.shape[0]
    v = jnp.array(v0,dtype=dtype)
    data = jnp.array(data0,dtype=dtype)
    indices = jnp.stack([jnp.asarray(row0,dtype=jnp.int32),
                          jnp.asarray(col0,dtype=jnp.int32)],axis=1)
    m = sparse.BCOO((data,indices),shape=(nd,nd))
    mus = _kpm_moments_sparse_jit(v,m,n)
    return np.array(mus)


def kpm_moments_batch_gpu(vs0,data0,row0,col0,n=100,gpu_batch_size=_GPU_BATCH_CHUNK,
        **kwargs):
    """Batched counterpart of kpm_moments_gpu: Chebyshev moments for a
    batch of starting vectors sharing the same sparse matrix, computed in
    fixed-size chunks of gpu_batch_size vectors (each chunk vmapped in one
    dispatch) so the whole batch is never materialized on-device at once --
    see the note above _GPU_BATCH_CHUNK. This still dispatches far fewer,
    far larger calls than looping over the single-vector kpm_moments_gpu
    once per vector, and mirrors the numba prange-batched CPU path in
    kpmnumba.py. vs0 has shape (nvec,nsites); returns an (nvec,2n) array of
    moments. **kwargs absorbs and ignores any other backend-selection
    kwargs (e.g. kpm_prec, already consumed by the caller) that
    kpm_moments_batch splats through here alongside gpu_batch_size -- the
    CPU/numba branch silently ignores unknown kwargs the same way, so a
    stray kwarg must not behave differently across backends."""
    dtype = vs0.dtype
    nd = vs0.shape[1]
    vs = jnp.array(vs0,dtype=dtype)
    data = jnp.array(data0,dtype=dtype)
    indices = jnp.stack([jnp.asarray(row0,dtype=jnp.int32),
                          jnp.asarray(col0,dtype=jnp.int32)],axis=1)
    m = sparse.BCOO((data,indices),shape=(nd,nd))
    mus = _kpm_moments_sparse_batch_jit(vs,m,n,min(gpu_batch_size,vs.shape[0]))
    return np.array(mus)


def _kpm_momentsA_sparse(v,Av,m,n):
    """Operator-weighted Chebyshev moments mus[i] = <T_i(m) v|A|v>, with Av
    (=A@v) precomputed once outside the recursion (see
    kpmnumba.kpm_moments_A_batch for why). Mirrors
    python_kpm_momentsA_batch_complex/_real's recursion."""
    mus = jnp.zeros(n,dtype=v.dtype)
    am = v
    a = m@v
    bk = jnp.sum(jnp.conjugate(v)*Av)
    bk1 = jnp.sum(jnp.conjugate(a)*Av)
    mus = mus.at[0].set(bk)
    mus = mus.at[1].set(bk1)

    def body(i,val):
        mus,am,a = val
        ap = 2*(m@a) - am # recursion relation
        bk = jnp.sum(jnp.conjugate(ap)*Av)
        mus = mus.at[i].set(bk)
        return mus,a,ap

    mus,_,_ = jax.lax.fori_loop(2,n,body,(mus,am,a))
    return mus


def _kpm_momentsA_sparse_batch(vs,Avs,m,n,batch_size):
    return jax.lax.map(lambda vAv: _kpm_momentsA_sparse(vAv[0],vAv[1],m,n),
            (vs,Avs),batch_size=batch_size)


_kpm_momentsA_sparse_batch_jit = jit(_kpm_momentsA_sparse_batch,static_argnums=(3,4))


def kpm_momentsA_batch_gpu(vs0,Avs0,data0,row0,col0,n=100,
        gpu_batch_size=_GPU_BATCH_CHUNK,**kwargs):
    """Batched operator-weighted moments (see kpm_moments_A_batch in
    kpmnumba.py), computed in fixed-size chunks of gpu_batch_size vectors
    (see the note above _GPU_BATCH_CHUNK / kpm_moments_batch_gpu), analogous
    to kpm_moments_batch_gpu above. vs0 and Avs0 both have shape
    (nvec,nsites); returns an (nvec,n) array. **kwargs is ignored -- see
    kpm_moments_batch_gpu's docstring for why it must accept and drop
    stray kwargs rather than raising."""
    dtype = vs0.dtype
    nd = vs0.shape[1]
    vs = jnp.array(vs0,dtype=dtype)
    Avs = jnp.array(Avs0,dtype=dtype)
    data = jnp.array(data0,dtype=dtype)
    indices = jnp.stack([jnp.asarray(row0,dtype=jnp.int32),
                          jnp.asarray(col0,dtype=jnp.int32)],axis=1)
    m = sparse.BCOO((data,indices),shape=(nd,nd))
    mus = _kpm_momentsA_sparse_batch_jit(vs,Avs,m,n,min(gpu_batch_size,vs.shape[0]))
    return np.array(mus)
