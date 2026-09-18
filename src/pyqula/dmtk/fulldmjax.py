"""The full density matrix on the GPU, the device counterpart of
dmtk/fulldm.py's numba kernels.

The CPU route diagonalizes a batch of k-points, brings the whole
(nk,n,n) stack of eigenvectors back to the host, and only then contracts
it down to one (n,n) density matrix per hopping direction. That round trip
is the expensive part once the solve itself is on the device: the
eigenvectors are the largest array in the calculation and they exist only
to be summed away. Worse, the SCF loop pays it once per iteration.

Everything here therefore stays on the device from the hopping matrices to
the finished density matrix, and what comes back is (nd,n,n) -- independent
of how dense the k-mesh is. The Bloch sum is fused in as well, for the same
reason it is in htk/eigenvectorsjax.py.

The contraction is the one dmtk/fulldm.py's full_dm_batch_d_vectorized
performs,

    dm_d[i,j] = sum_k exp(2*pi*i*k.d) sum_a conj(w[k,i,a]) occ[k,a] w[k,j,a]

with occ the Fermi function of the eigenvalue measured from the Fermi
energy. The undirected density matrix (fulldm's full_dm_batch_vectorized)
is the same expression at d=(0,0,0), where the phase is 1, so both take
this one path.
"""

import numpy as np

from .. import gpu
gpu.apply() # the package-wide CPU/GPU switch, see pyqula/gpu.py

import jax
jax.config.update("jax_enable_x64",True) # keep "double" actually double
import jax.numpy as jnp

from ..htk import eigenvectorsjax as evjax

_DTYPES = {"single": np.complex64, "double": np.complex128}
_REAL_DTYPES = {"single": np.float32, "double": np.float64}


@jax.jit
def _dm_chunk(ms,dirs,kbloch,kfull,ds,fermi,delta):
    """The density matrix of one chunk of the k-mesh, one slot per
    direction, summed over the chunk's k-points.

    kbloch is the k-mesh cropped to the periodic directions, which is what
    the Bloch sum uses; kfull keeps all three components, which is what the
    hopping-direction phases use -- exactly the split the numba kernels
    make between the generator and full_dm_batch_d_vectorized."""
    phases = jnp.exp(1j*2*jnp.pi*(kbloch@dirs.T)) # (nk,nhop)
    hk = jnp.einsum("kh,hij->kij",phases,ms) # the Bloch Hamiltonian
    es,ws = jnp.linalg.eigh(hk)
    # the Fermi function, written as a sigmoid so that a level far from
    # the Fermi energy saturates instead of overflowing: delta is ~1e-6 by
    # default, so es/delta reaches the exponential's limit routinely
    occ = jax.nn.sigmoid(-(es-fermi)/delta) # (nk,n)
    # the k-resolved density matrix, before the direction phases
    dmk = jnp.einsum("kia,ka,kja->kij",jnp.conjugate(ws),occ.astype(ws.dtype),ws)
    kd = jnp.exp(1j*2*jnp.pi*(kfull@ds.T)) # (nk,nd) hopping-direction phases
    return jnp.einsum("kd,kij->dij",kd,dmk)


def full_dm_gpu(ms,dirs,ks,ds,fermi=0.0,delta=1e-7,prec="double"):
    """The density matrix of the Bloch Hamiltonian built from ms/dirs, on
    the k-mesh ks, for every hopping direction in ds.

    Returns (nd,n,n), NOT normalized by the number of k-points -- the
    caller does that, matching the numba route. ks carries three
    components per k-point and dirs is already cropped to the periodic
    directions, as htk/bloch.py stores it."""
    if prec not in _DTYPES:
        raise ValueError("the precision must be 'single' or 'double', got "
                +repr(prec))
    ct,rt = _DTYPES[prec],_REAL_DTYPES[prec]
    ms = jnp.asarray(np.asarray(ms,dtype=ct))
    dirs = jnp.asarray(np.asarray(dirs,dtype=rt))
    ds = jnp.asarray(np.asarray(ds,dtype=rt))
    ks = np.asarray(ks,dtype=rt)
    n = ms.shape[1]
    # the same chunking the batched solve uses, and for the same reason:
    # the solver's workspace is sized against the matrices the Bloch sum
    # produces, n*n per k-point
    size = max(1,evjax.CHUNK_ELEMENTS//(n*n))
    out = np.zeros((ds.shape[0],n,n),dtype=np.complex128)
    for i0 in range(0,ks.shape[0],size):
        kc = ks[i0:i0+size]
        chunk = _dm_chunk(ms,dirs,jnp.asarray(kc[:,0:dirs.shape[1]]),
                jnp.asarray(kc),ds,rt(fermi),rt(delta))
        out += np.array(chunk,dtype=np.complex128) # pool on the host, (nd,n,n)
    return out
