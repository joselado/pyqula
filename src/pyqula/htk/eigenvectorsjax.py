"""Batched dense diagonalization on the GPU, the device counterpart of
htk/eigenvectors.py's numba kernels.

Both entry points take a stack of Hermitian matrices, shape (nh,n,n), and
return exactly what the numba path returns: float64 eigenvalues, and for
peigh_gpu complex128 eigenvectors, whatever precision the solve ran in.

Two things are not obvious and are measured, on a GTX 1060, in
documentation/gpu_porting_plan.md:

- The crossover is in the matrix size n, not in the batch size. In double
  precision the device loses to 8-thread numba below n~32 however many
  matrices are stacked, because each solve is too small to fill the card
  and the transfers are not amortized. eigenvectors.py applies that as a
  size threshold before dispatching here.
- cuSOLVER's batched Jacobi solver asks for roughly 500 bytes of workspace
  per matrix entry, so a 4096 x 64 x 64 stack requested 7.75 GiB in one
  dispatch and failed on a 6 GB card. The batch is therefore chunked by
  the number of entries, not by the number of matrices.
"""

import numpy as np

from .. import gpu
gpu.apply() # the package-wide CPU/GPU switch, see pyqula/gpu.py

import jax
jax.config.update("jax_enable_x64",True) # keep "double" actually double
import jax.numpy as jnp

# matrix entries per device dispatch, which bounds the solver workspace
CHUNK_ELEMENTS = 2**21

_DTYPES = {"single": np.complex64, "double": np.complex128}


@jax.jit
def _eigh(m): return jnp.linalg.eigh(m)


@jax.jit
def _eigvalsh(m): return jnp.linalg.eigvalsh(m)


def _chunks(hks,prec):
    """Yield the stack in device-sized pieces, cast to the solve precision.
    Every chunk but the last has the same shape, so the jit compiles once"""
    if prec not in _DTYPES:
        raise ValueError("the precision must be 'single' or 'double', got "
                +repr(prec))
    nh,n = hks.shape[0],hks.shape[1]
    size = max(1,CHUNK_ELEMENTS//(n*n)) # matrices per dispatch
    for i in range(0,nh,size):
        yield jnp.asarray(np.asarray(hks[i:i+size],dtype=_DTYPES[prec]))


def peigh_gpu(hks,prec="double"):
    """Eigenvalues and eigenvectors of a stack of Hermitian matrices"""
    es,ws = [],[]
    for m in _chunks(hks,prec):
        e,w = _eigh(m)
        es.append(np.array(e,dtype=np.float64))
        ws.append(np.array(w,dtype=np.complex128))
    return np.concatenate(es),np.concatenate(ws)


def peigvalsh_gpu(hks,prec="double"):
    """Eigenvalues of a stack of Hermitian matrices"""
    es = [np.array(_eigvalsh(m),dtype=np.float64) for m in _chunks(hks,prec)]
    return np.concatenate(es)


# The Bloch sum fused into the solve
#
# The two entry points above take a stack that the host has already built,
# one k-point at a time, with htk/eigenvectors.py's hk_matrix_batch -- a
# Python loop over the Bloch generator, followed by a transfer of the whole
# (nk,n,n) stack to the device. Once the solve itself is on the GPU that
# build is no longer a rounding error: measured on a GTX 1060 it is 15% of
# the pair's cost in double precision and 34% in single (n=64-144,
# nk=900-1600, see documentation/gpu_porting_plan.md).
#
# The functions below take the Bloch ingredients instead -- the hopping
# matrices ms, their lattice vectors ds, and the k-mesh -- and do the sum
# on the device, so what crosses the bus is (nhop,n,n) once rather than
# (nk,n,n) per call, and the host loop disappears. Everything else matches
# the two entry points above, chunking included.

_REAL_DTYPES = {"single": np.float32, "double": np.float64}


@jax.jit
def _bloch(ms,ds,ks):
    """The Bloch sum over a whole k-mesh at once, shape (nk,n,n)"""
    phases = jnp.exp(1j*2*jnp.pi*(ks@ds.T)) # (nk,nhop)
    return jnp.einsum("kh,hij->kij",phases,ms)


@jax.jit
def _bloch_eigh(ms,ds,ks): return jnp.linalg.eigh(_bloch(ms,ds,ks))


@jax.jit
def _bloch_eigvalsh(ms,ds,ks): return jnp.linalg.eigvalsh(_bloch(ms,ds,ks))


def _bloch_chunks(ms,ds,ks,prec):
    """Yield the k-mesh in device-sized pieces, with the (small, shared)
    hopping arrays cast once. The chunk bound is on the matrices the sum
    produces, n*n per k-point, since those are what the solver's workspace
    is sized against"""
    if prec not in _DTYPES:
        raise ValueError("the precision must be 'single' or 'double', got "
                +repr(prec))
    ms = jnp.asarray(np.asarray(ms,dtype=_DTYPES[prec]))
    ds = jnp.asarray(np.asarray(ds,dtype=_REAL_DTYPES[prec]))
    ks = np.asarray(ks,dtype=_REAL_DTYPES[prec])
    ks = ks[:,0:ds.shape[1]] # the Bloch phases only use the periodic directions
    n = ms.shape[1]
    size = max(1,CHUNK_ELEMENTS//(n*n)) # k-points per dispatch
    for i in range(0,ks.shape[0],size):
        yield ms,ds,jnp.asarray(ks[i:i+size])


def peigh_bloch_gpu(ms,ds,ks,prec="double"):
    """Eigenvalues and eigenvectors of the Bloch Hamiltonian at every k in
    ks, built on the device from the hopping matrices ms and their lattice
    vectors ds"""
    es,ws = [],[]
    for chunk in _bloch_chunks(ms,ds,ks,prec):
        e,w = _bloch_eigh(*chunk)
        es.append(np.array(e,dtype=np.float64))
        ws.append(np.array(w,dtype=np.complex128))
    return np.concatenate(es),np.concatenate(ws)


def peigvalsh_bloch_gpu(ms,ds,ks,prec="double"):
    """Eigenvalues of the Bloch Hamiltonian at every k in ks, built on the
    device -- the eigenvector-free counterpart of peigh_bloch_gpu"""
    es = [np.array(_bloch_eigvalsh(*chunk),dtype=np.float64)
            for chunk in _bloch_chunks(ms,ds,ks,prec)]
    return np.concatenate(es)
