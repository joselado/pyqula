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
