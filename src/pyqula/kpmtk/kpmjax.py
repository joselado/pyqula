

def is_gpu_available():
    import jax
    try:
        devices = jax.devices()
        return True
    except: return False


if is_gpu_available(): # GPU available
    print("GPU available for KPM")
    pass
else: # use the CPU
    print("GPU is NOT available for KPM, using the CPU")
    import os
    os.environ['JAX_PLATFORMS'] = 'cpu'

import jax
import jax.numpy as jnp
from jax import jit





def Mtimesv_jax(data, row, col, v):
    """Matrix times vector"""
    return jax.ops.segment_sum(data * v[col], row, num_segments=v.shape[0])





def python_kpm_moments_real_jax(v, data, row, col, n=100):
    """Python routine to calculate moments"""
    mus = jnp.zeros(2 * n, dtype=v.dtype)  # empty array for the moments
    am = v.copy()  # zero vector
    a = Mtimesv_jax(data, row, col, v)  # m@v  # vector number 1
    bk = jnp.sum(v * v)
    bk1 = jnp.sum(a * v)  # algebra.braket_ww(a,v)

    mus = mus.at[0].set(bk)  # mu0
    mus = mus.at[1].set(bk1)  # mu1

    def body_fn(i, carry):
        am, a, mus = carry
        ap = 2 * Mtimesv_jax(data, row, col, a) - am  # recursion relation
        bk = jnp.sum(a * a)  # algebra.braket_ww(a,a)
        bk1 = jnp.sum(ap * a)  # algebra.braket_ww(ap,a)
        mus = mus.at[2 * i].set(2. * bk)
        mus = mus.at[2 * i + 1].set(2. * bk1)
        return a, ap, mus

    _, _, mus = jax.lax.fori_loop(1, n, body_fn, (am, a, mus))

    mu0 = mus[0]  # first
    mu1 = mus[1]  # second

    def update_mus(i, mus):
        mus = mus.at[2 * i].add(-mu0)
        mus = mus.at[2 * i + 1].add(-mu1)
        return mus

    mus = jax.lax.fori_loop(1, n, update_mus, mus)
    
    return mus




# JIT compile the functions
python_kpm_moments_real = jit(python_kpm_moments_real_jax, static_argnums=(4,))
Mtimesv = jit(Mtimesv_jax)



import numpy as np


def kpm_moments_real_gpu(v0,data0,row0,col0,n=100):
    v = jnp.array(v0,dtype=jnp.float32)
    data = jnp.array(data0,dtype=jnp.float32)
    row = jnp.array(row0,dtype=jnp.int32)
    col = jnp.array(col0,dtype=jnp.int32)
    mus = python_kpm_moments_real(v,data,row,col,n=n)
    return np.array(mus) # return



def kpm_moments_real_gpu_sparse(v0,m,n=100):
    from scipy.sparse import csr_matrix
    from jax.experimental import sparse
    m = csr_matrix(m.real,dtype=np.float32)
    data = m.data
    indices = m.indices
    indptr = m.indptr
    shape = m.shape
    jm = sparse.BCOO.from_scipy_sparse(m) # sparse JAX matrix
    v = jnp.array(v0,dtype=jnp.float32) # as float
    return python_kpm_moments_real_sparse(v,jm,n=n)




#def python_kpm_moments_real_sparse_jax(v,m,n=100):
#    """Python routine to calculate moments"""
#    mus = jnp.zeros(2*n,dtype=v.dtype) # empty array for the moments
#    am = v.copy() # zero vector
#    a = m@v  # vector number 1
#    bk = jnp.sum(v*v)
#    bk1 = jnp.sum(a*v) #algebra.braket_ww(a,v)
#    mus[0] = bk  # mu0
#    mus[1] = bk1 # mu1
#    for i in range(1,n):
#        ap = 2*m@a - am # recursion relation
#        bk = jnp.sum(a*a) # algebra.braket_ww(a,a)
#        bk1 = jnp.sum(ap*a) # algebra.braket_ww(ap,a)
#        mus[2*i] = 2.*bk
#        mus[2*i+1] = 2.*bk1
#        am,a = a,ap # reassign
#    mu0 = mus[0] # first
#    mu1 = mus[1] # second
#    for i in range(1,n):
#      mus[2*i] +=  - mu0
#      mus[2*i+1] += -mu1
#    return mus
#



def python_kpm_moments_real_sparse_jax(v, m, n=100):
    """Python routine to calculate moments"""
    mus = jnp.zeros(2*n, dtype=v.dtype)  # empty array for the moments
    am = v  # zero vector
    a = m @ v  # vector number 1
    bk = jnp.sum(v * v)
    bk1 = jnp.sum(a * v)
    mus = mus.at[0].set(bk)  # mu0
    mus = mus.at[1].set(bk1)  # mu1

    def body(i, val):
        mus, am, a = val
        ap = 2 * (m @ a) - am  # recursion relation
        bk = jnp.sum(a * a)
        bk1 = jnp.sum(ap * a)
        mus = mus.at[2 * i].set(2. * bk)
        mus = mus.at[2 * i + 1].set(2. * bk1)
        return mus, a, ap

    mus, _, _ = jax.lax.fori_loop(1, n, body, (mus, am, a))

    mu0 = mus[0]  # first
    mu1 = mus[1]  # second
    mus = mus.at[2::2].add(-mu0)
    mus = mus.at[3::2].add(-mu1)

    return mus





python_kpm_moments_real_sparse = jit(python_kpm_moments_real_sparse_jax, static_argnums=(2,))





