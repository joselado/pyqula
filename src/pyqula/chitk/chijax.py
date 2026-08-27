"""jax/GPU implementation of the Lindhard kernel behind the RPA response.

This is the device counterpart of chitk/chiAB.py's numba `chiAB_matrix`,
i.e. of `chiAB`'s mode="matrix" branch, which is what every RPA spin/charge
entry point bottoms out in (get_spinchi_full, get_spinchi_ladder,
get_magnon_bands(method="rpa"), get_rpa_kernel_poles, get_iets_ldos,
chitk/densitychi.py). mode="trace"/"diagonal" use a different kernel and
are deliberately NOT ported here -- see
future_development/gpu_rpa_spin_response.md.

The contraction

    out[i,j,w] = sum_ab MA[i,a,b] MB[j,b,a] (f_a-f_b)/(e_a-e'_b-w+i*delta)

is a GEMM in disguise: flattening p=(a,b) and writing TA[i,p]=MA[i,a,b],
TB[j,p]=MB[j,b,a], D[p,w]=(f_a-f_b)/(e_a-e'_b-w+i*delta),

    out[w] = (TA * D[:,w]) @ TB.T

one (ni x P) x (P x nj) GEMM per frequency. That form is what a GPU wants
and it is bit-for-bit the same arithmetic as the numba loop (verified to
~1e-16 relative).

Two implementation points that are not cosmetic:

- The numba kernel skips pairs with |f_a-f_b| < delta/100, which at low
  temperature is every occupied-occupied and empty-empty pair. Reproducing
  that as a mask would multiply zeros; here the surviving pairs are
  *gathered* into a compact index list. The number of survivors varies with
  k, q and filling, and a varying shape makes jax retrace the whole kernel,
  so the list is padded to a quantized fixed length with the padding
  entries carrying f_a-f_b = 0 (so they contribute exactly zero). Shape
  stability is what makes the jit compile once for a whole k-mesh/q-path
  instead of once per k-point.
- The frequency axis is mapped over with jax.lax.map (a scan), not vmap: it
  is a free output axis, so splitting it costs nothing, while vmap would
  materialize one (ni x P) scaled copy of TA per frequency at once. Same
  reasoning as kpmtk/kpmjax.py's chunked batch dispatch.
"""

def is_gpu_available():
    import jax
    try:
        jax.devices("gpu")
        return True
    except Exception: return False


if is_gpu_available(): # GPU available
    print("GPU available for chi")
    pass
else: # use the CPU
    import os
    os.environ['JAX_PLATFORMS'] = 'cpu'

import numpy as np
import jax
jax.config.update("jax_enable_x64",True) # allow float64/complex128 (JAX
# defaults to 32 bit precision, which would silently truncate "double"
# precision requests down to "single")
import jax.numpy as jnp
from functools import partial


PAIR_PAD_QUANTUM = 2048 # pad the gathered pair list up to a multiple of
# this, so that neighbouring k-points (whose survivor counts differ by a
# few) share one compiled kernel instead of triggering a retrace each


def _occupations(es,temp):
    """Fermi occupations, mathematically identical to chiAB.chiAB_matrix's
    1/(1+exp(beta*e)) but written with tanh, which does not overflow at low
    temperature (chiAB_full_matrix_jit already uses the tanh form)."""
    beta = 1./temp
    return 0.5*(1. - np.tanh(0.5*beta*np.asarray(es)))


def pair_plan(es1,es2,temp,delta,pair_pad=None):
    """Return (idx_ab, idx_ba, facs, dE), the gathered+padded list of
    (a,b) pairs that survive the occupation cutoff of chiAB_matrix.

    idx_ab indexes a flattened (a,b) array, idx_ba a flattened (b,a) one
    (TB is built from MB[j,b,a], hence the second index list). Padding
    entries carry facs=0, so they contribute exactly zero to the result
    while keeping every device-side shape constant."""
    n = len(es1)
    cutoff = delta/100. # same cutoff as the numba kernel
    o1 = _occupations(es1,temp)
    o2 = _occupations(es2,temp)
    fac = o1[:,None] - o2[None,:] # (n,n) occupation factor
    sel = np.abs(fac) >= cutoff # the pairs the numba loop does not skip
    idx = np.flatnonzero(sel.ravel()) # flat (a,b) indices
    count = len(idx)
    if pair_pad is None: # quantize, so nearby k-points share a compilation
        pair_pad = int(np.ceil(max(count,1)/PAIR_PAD_QUANTUM)*PAIR_PAD_QUANTUM)
        pair_pad = min(pair_pad,n*n) # never pad beyond the full pair set
    if count>pair_pad:
        raise ValueError(f"pair_pad={pair_pad} is smaller than the number "
                         f"of contributing pairs ({count})")
    pad = pair_pad - count
    idx_pad = np.concatenate([idx,np.zeros(pad,dtype=idx.dtype)])
    a = idx_pad//n # first state index
    b = idx_pad%n # second state index
    facs = np.concatenate([fac.ravel()[idx],np.zeros(pad)])
    dE = np.asarray(es1)[a] - np.asarray(es2)[b]
    return idx_pad,(b*n+a),facs.astype(np.complex128),dE


@partial(jax.jit,static_argnums=())
def _gathered_operator_tensors(ws1,ws2,Ais,Bjs,idx_ab,idx_ba):
    """Build TA[i,p]=MA[i,a,b] and TB[j,p]=MB[j,b,a] for the gathered
    pairs. MA[i] = conj(ws1)@(Ais[i]@ws2.T) and MB[j] = conj(ws2)@(Bjs[j]
    @ws1.T), exactly as chiAB_matrix builds them."""
    ni = Ais.shape[0]
    nj = Bjs.shape[0]
    n = ws1.shape[0]
    MA = jnp.conjugate(ws1)@(Ais@ws2.T) # (ni,n,n)
    MB = jnp.conjugate(ws2)@(Bjs@ws1.T) # (nj,n,n)
    TA = MA.reshape(ni,n*n)[:,idx_ab] # (ni,P)
    TB = MB.reshape(nj,n*n)[:,idx_ba] # (nj,P), note the (b,a) index list
    return TA,TB


def _chi_from_tensors(TA,TB,facs,dE,energies,delta):
    """out[w] = (TA*D[:,w]) @ TB.T, mapped over the frequency axis."""
    def one_frequency(w):
        D = facs/(dE - w + 1j*delta) # (P,)
        return (TA*D)@TB.T # (ni,nj)
    return jax.lax.map(one_frequency,energies) # (nw,ni,nj)


_chi_from_tensors_jit = jax.jit(_chi_from_tensors)


def chiAB_matrix_gpu(ws1,es1,ws2,es2,energies,Ais,Bjs,temp,delta,
                     pair_pad=None):
    """Drop-in replacement for chiAB.chiAB_matrix: same arguments, same
    (nw,ni,nj) return value. Provided so the two can be diffed on
    identical input; production calls should use chi_matrix_kmesh_gpu,
    which keeps the whole k-mesh on the device."""
    idx_ab,idx_ba,facs,dE = pair_plan(es1,es2,temp,delta,pair_pad=pair_pad)
    TA,TB = _gathered_operator_tensors(jnp.asarray(ws1),jnp.asarray(ws2),
                                       jnp.asarray(Ais),jnp.asarray(Bjs),
                                       jnp.asarray(idx_ab),jnp.asarray(idx_ba))
    out = _chi_from_tensors_jit(TA,TB,jnp.asarray(facs),jnp.asarray(dE),
                                jnp.asarray(energies),delta)
    return np.array(out)


def chi_matrix_kmesh_gpu(hks1,hks2,energies,Ais,Bjs,temp,delta,
                         pair_pad=None):
    """Full k-mesh response, averaged over k, without returning to the
    host in between.

    hks1[ik], hks2[ik] are H(k) and H(k+q) for every k of the mesh. The
    eigendecompositions are done once as a single batched eigh; the pair
    plan is built for every k first so that one padded length covers the
    whole mesh, which is what keeps the kernel compiled exactly once."""
    hks1 = jnp.asarray(hks1)
    hks2 = jnp.asarray(hks2)
    es1k,wsk1 = jnp.linalg.eigh(hks1) # batched over the mesh
    es2k,wsk2 = jnp.linalg.eigh(hks2)
    es1k_h = np.array(es1k) # occupations/pair selection are host-side
    es2k_h = np.array(es2k)
    nk = hks1.shape[0]
    if pair_pad is None: # one padded length for the whole mesh
        n = hks1.shape[1]
        cutoff = delta/100.
        counts = []
        for ik in range(nk):
            fac = _occupations(es1k_h[ik],temp)[:,None] \
                    - _occupations(es2k_h[ik],temp)[None,:]
            counts.append(int(np.sum(np.abs(fac)>=cutoff)))
        pair_pad = int(np.ceil(max(max(counts),1)/PAIR_PAD_QUANTUM)
                       *PAIR_PAD_QUANTUM)
        pair_pad = min(pair_pad,n*n)
    Ais = jnp.asarray(Ais)
    Bjs = jnp.asarray(Bjs)
    energies = jnp.asarray(energies)
    out = None
    for ik in range(nk): # loop over k, everything stays on the device
        idx_ab,idx_ba,facs,dE = pair_plan(es1k_h[ik],es2k_h[ik],temp,delta,
                                          pair_pad=pair_pad)
        # eigenvectors come back as columns; the kernel wants states as rows
        ws1 = wsk1[ik].T
        ws2 = wsk2[ik].T
        TA,TB = _gathered_operator_tensors(ws1,ws2,Ais,Bjs,
                                           jnp.asarray(idx_ab),
                                           jnp.asarray(idx_ba))
        chik = _chi_from_tensors_jit(TA,TB,jnp.asarray(facs),jnp.asarray(dE),
                                     energies,delta)
        out = chik if out is None else out + chik
    return np.array(out/nk) # mean over the mesh, same as the CPU path
