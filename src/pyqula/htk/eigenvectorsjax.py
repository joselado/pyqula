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
