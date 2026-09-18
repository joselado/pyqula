"""jax/GPU implementation of the pair-basis Lindhard kernel.

This is the device counterpart of chitk/pairchi.py's numba `_accumulate`,
the four-fold loop behind every pair-basis entry point (pair_chi0,
pair_rpa_kernel, pair_chi_rpa, pair_rpa_poles, magnon_bands_pair, and the
public spin response of any Hamiltonian whose interaction couples
different sites, which chitk/spinchi.py reroutes here).

It is the same contraction chitk/chijax.py already ports, with the site
index of the site-basis route replaced by the interaction's pair index:

    chi0[P,P',w] += sum_nm (f_n-f_m) M_P conj(M_P')/(e1_n-e2_m-w+i*delta)

with M_P = conj(u1[n,iP]) u2[m,jP] phase_P. Flattening the band pair
g=(n,m) and writing T[P,g] = M_P, D[g,w] = (f_n-f_m)/(e1_n-e2_m-w+i*delta),

    chi0[w] = (T * D[:,w]) @ T^dagger

one (npair x G) x (G x npair) GEMM per frequency -- so the whole file is
chijax.py's structure with one operator tensor instead of two, and with
TB = conj(TA) because both sides of the pair response carry the same pair
operators. Its two non-cosmetic implementation points carry over verbatim
and are documented there: the surviving band pairs are *gathered* rather
than masked, and padded to a quantized fixed length so that jax compiles
the kernel once for a whole k-mesh; and the frequency axis is mapped over
with jax.lax.map rather than vmap, which would materialize one scaled copy
of T per frequency at once.

One thing differs from chijax and matters for correctness: the numba
kernel here skips a band pair on `df == 0.` exactly, not on
`|df| < delta/100`. That is a different set of survivors -- at finite
temperature almost every pair contributes, where the site-basis kernel
drops the ones below its cutoff -- so the gather below reproduces this
module's own rule and the two must not be unified.

Memory, since the pair basis is the wide one: the operator tensor is
(npair, G) with G up to nb^2, and the result is (npair, npair, nw). Both
grow with the interaction's support rather than with N^2, which is the
whole point of the pair basis, but a long-ranged interaction on a large
cell will reach the device's memory in the tensor before it reaches it in
the response.
"""

import numpy as np

from .. import gpu
# the package-wide CPU/GPU switch decides where this kernel runs, see
# pyqula/gpu.py and the note in kpmtk/kpmjax.py
gpu.apply()

import jax
jax.config.update("jax_enable_x64",True) # allow float64/complex128, which
# jax does not use by default
import jax.numpy as jnp

# the precision table and the padding quantum are the site-basis kernel's,
# deliberately shared: chi_prec means the same thing on both routes, and a
# second copy of either would be free to drift
from .chijax import _chi_dtypes, PAIR_PAD_QUANTUM


def band_pair_plan(e1,e2,f1,f2,pair_pad=None):
    """Return (ns, ms, facs, dE), the gathered and padded list of band
    pairs (n,m) that contribute at one k-point.

    The selection is `_accumulate`'s own: a pair enters unless its
    occupation difference is exactly zero. Padding entries carry facs=0,
    so they contribute exactly zero while keeping every device-side shape
    constant -- see the module docstring."""
    nb = len(e1)
    df = np.asarray(f1)[:,None] - np.asarray(f2)[None,:] # (nb,nb)
    idx = np.flatnonzero((df!=0.).ravel()) # the pairs the loop does not skip
    count = len(idx)
    if pair_pad is None: # quantize, so nearby k-points share a compilation
        pair_pad = int(np.ceil(max(count,1)/PAIR_PAD_QUANTUM)*PAIR_PAD_QUANTUM)
        pair_pad = min(pair_pad,nb*nb) # never pad beyond the full pair set
    if count>pair_pad:
        raise ValueError(f"pair_pad={pair_pad} is smaller than the number "
                         f"of contributing band pairs ({count})")
    idx_pad = np.concatenate([idx,np.zeros(pair_pad-count,dtype=idx.dtype)])
    ns = idx_pad//nb # index into the first set of states
    ms = idx_pad%nb # index into the second set
    facs = np.concatenate([df.ravel()[idx],np.zeros(pair_pad-count)])
    dE = np.asarray(e1)[ns] - np.asarray(e2)[ms]
    return ns,ms,facs,dE


@jax.jit
def _pair_operator_tensor(u1,u2,iP,jP,phase,ns,ms):
    """Build T[P,g] = conj(u1[n,iP[P]]) u2[m,jP[P]] phase[P] for the
    gathered band pairs g=(n,m), exactly the M_P of `_accumulate`.

    u1[n], u2[m] are states as rows (pairchi's u1 = w1.T)."""
    A = jnp.conjugate(u1)[ns][:,iP] # (G,npair), the creation index
    B = u2[ms][:,jP] # (G,npair), the annihilation index
    return (A*B).T*phase[:,None] # (npair,G)


def _chi_from_tensor(T,facs,dE,energies,delta,cdt):
    """chi0[w] = (T*D[:,w]) @ T^dagger, mapped over the frequency axis.

    D is formed in double precision and only then rounded to the precision
    of the GEMM, the same care chitk/chiAB.py's numba kernel takes: it is
    O(G) against the GEMM's O(npair^2 G), and e1-e2-w is a near
    cancellation right where the response is large."""
    Td = T.conj().T
    def one_frequency(w):
        D = (facs/(dE - w + 1j*delta)).astype(cdt) # (G,)
        return (T*D)@Td # (npair,npair)
    return jax.lax.map(one_frequency,energies) # (nw,npair,npair)


_chi_from_tensor_jit = jax.jit(_chi_from_tensor,static_argnums=(5,))


def pair_chi0_kmesh_gpu(hks1,hks2,phases,iP,jP,energies,T,delta,
                        pair_pad=None,chi_prec="single"):
    """Drop-in device replacement for chitk/pairchi.py's per-k loop over
    `_accumulate`: same (npair,npair,nomega) complex128 return value,
    already averaged over the mesh.

    hks1[ik], hks2[ik] are H(k) and H(k+q) over the mesh and phases[ik] the
    Bloch phase of every pair at that k, all built by the caller, which is
    where the geometry lives. The eigendecompositions are done as one
    batched eigh and everything stays on the device across the k-loop.

    chi_prec sets the precision of the operator tensor and of the GEMM,
    which is where the time goes. The eigendecompositions, the occupations,
    the pair plan and the energy denominators stay in double precision
    either way: they are cheap, and doing them in double keeps single
    precision's error at the rounding of the contraction rather than
    compounding it with eigenvector error."""
    from .pairchi import _occupations
    _,cdt = _chi_dtypes(chi_prec) # only the tensor and the GEMM are rounded
    es1k,wsk1 = jnp.linalg.eigh(jnp.asarray(hks1)) # batched over the mesh
    es2k,wsk2 = jnp.linalg.eigh(jnp.asarray(hks2))
    es1k_h = np.array(es1k) # occupations and pair selection are host-side
    es2k_h = np.array(es2k)
    nk = np.shape(hks1)[0]
    nb = np.shape(hks1)[1]
    f1 = [_occupations(es1k_h[ik],T) for ik in range(nk)]
    f2 = [_occupations(es2k_h[ik],T) for ik in range(nk)]
    if pair_pad is None: # one padded length for the whole mesh, so that
        # the kernel compiles once rather than once per k-point
        counts = [int(np.sum(f1[ik][:,None]-f2[ik][None,:]!=0.))
                  for ik in range(nk)]
        pair_pad = int(np.ceil(max(max(counts),1)/PAIR_PAD_QUANTUM)
                       *PAIR_PAD_QUANTUM)
        pair_pad = min(pair_pad,nb*nb)
    iP = jnp.asarray(iP)
    jP = jnp.asarray(jP)
    # the frequencies stay double: they enter only the denominator, which
    # is formed in double and rounded after the division, see
    # _chi_from_tensor
    energies = jnp.asarray(energies,dtype=np.float64)
    out = None
    for ik in range(nk): # loop over k, everything stays on the device
        ns,ms,facs,dE = band_pair_plan(es1k_h[ik],es2k_h[ik],f1[ik],f2[ik],
                                       pair_pad=pair_pad)
        # eigenvectors come back as columns; the kernel wants states as rows
        Tm = _pair_operator_tensor(wsk1[ik].T.astype(cdt),
                                   wsk2[ik].T.astype(cdt),iP,jP,
                                   jnp.asarray(phases[ik],dtype=cdt),
                                   jnp.asarray(ns),jnp.asarray(ms))
        chik = _chi_from_tensor_jit(Tm,jnp.asarray(facs),
                                    jnp.asarray(dE),energies,delta,cdt)
        # the k-sum accumulates in double even when the GEMM ran in single:
        # a single precision accumulator drifts with the size of the mesh,
        # and the cast costs one output-sized array per k-point against the
        # GEMM's factor G more work
        chik = chik.astype(np.complex128)
        out = chik if out is None else out + chik
    # (nw,npair,npair) on the device, (npair,npair,nw) on the way out, the
    # layout pairchi.pair_chi0 returns
    return np.array(jnp.transpose(out,(1,2,0))/nk,dtype=np.complex128)
