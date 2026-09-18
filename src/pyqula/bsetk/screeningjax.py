"""jax/GPU implementation of the static polarizability behind the screened
interaction.

This is the device counterpart of bsetk/screening.py's numba
`polarizability_jit`, the kernel every `screening="rpa"`/`"crpa"` BSE pays
for (screened_interaction -> static_polarizability, and so
h.get_bse(screening=...), h.get_screened_interaction and
h.get_static_polarizability).

The contraction is the same one chitk/chijax.py and chitk/pairchijax.py
port, with the operator index replaced by the orbital index of the density
form factor:

    chi0_ab(q) = (1/nk) sum_{k,n,m} w_knm rho_a conj(rho_b),
    rho_a = conj(ck[k,n,a]) ck[k+q,m,a],
    w_knm = (f_kn - f_{k+q,m})/(e_kn - e_{k+q,m})

so with the transition index g = (k,n,m) flattened and rho[g,a] the form
factor,

    chi0(q) = (rho^T * w) @ conj(rho)

one (norb x G) x (G x norb) GEMM per q-point, G = nk*nb^2.

**This kernel masks where the other two gather, and the difference is
measured.** chijax and pairchijax pull the contributing transitions out
into a compact list, because their contraction is per k-point and the
frequency axis makes the tensor expensive. Here the whole Brillouin zone
is one contraction and the output is only norb x norb, so the GEMM is
small next to the cost of *preparing* it: a first version of this module
gathered on the host, one numpy pass per q-point, and that gather --
not the GEMM -- was the whole device time (on a GTX 1060, norb=64: 1.38 s,
against 2.97 s for numba, where the GEMMs themselves account for
milliseconds). Building rho and the weights on the device instead, and
zeroing the excluded transitions rather than dropping them, does ~2x more
arithmetic in the GEMM and none of the host work. It is also why nothing
here needs a padding quantum: every shape is fixed by (nk,nb,norb) alone,
so the kernel compiles once no matter what the occupations do.

Two selections are folded into that mask, not one: a transition enters
only if the occupations differ AND `allowed[n,m]`, which is what makes
cRPA cRPA.

Memory is the one thing to watch, since rho is materialized: it is
(nk, nb, nb, norb) for one q-point, complex, i.e. 75 MB at nk=36, nb=64,
norb=64 in single precision and twice that in double. lax.map over q keeps
exactly one of those alive at a time; a much larger mesh or cell wants a
chunk over k as well, which is not implemented.
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
from functools import partial

# the precision table is the site-basis kernel's, deliberately shared:
# chi_prec means the same thing on every route
from ..chitk.chijax import _chi_dtypes


def _weights(ek,occ,jk,allowed):
    """The weight (f_kn - f_{k+q,m})/(e_kn - e_{k+q,m}) of every
    transition at one q-point, zero where it does not contribute.

    jk[ik] is the mesh index of k_ik + q. The two zeros are
    polarizability_jit's two `continue`s: an occupation difference of
    exactly zero, and a transition `allowed` excludes. The denominator is
    guarded before the division rather than after, since 0*inf is NaN
    where 0*1 is not -- a same-occupancy pair at the same energy is
    exactly the case that would otherwise poison the whole sum."""
    df = occ[:,:,None] - occ[jk][:,None,:] # (nk,nb,nb), n at k, m at k+q
    de = ek[:,:,None] - ek[jk][:,None,:]
    keep = (df!=0.)&allowed[None,:,:]
    return jnp.where(keep,df/jnp.where(keep,de,1.),0.)


def _chi0_one_q(ck,ek,occ,jk,allowed,cdt):
    """chi0(q) for one q-point, as one GEMM over every transition"""
    norb = ck.shape[2]
    w = _weights(ek,occ,jk,allowed) # (nk,nb,nb), double precision
    # rho[k,n,m,a] = conj(ck[k,n,a]) ck[k+q,m,a], the density form factor
    rho = (jnp.conjugate(ck)[:,:,None,:]*ck[jk][:,None,:,:]).reshape(-1,norb)
    return (rho.T*w.reshape(-1).astype(cdt))@jnp.conjugate(rho)


@partial(jax.jit,static_argnums=(5,))
def _chi0_all_q(ck,ek,occ,ikq,allowed,cdt):
    """chi0 at every q-point of the mesh, mapped over q.

    jax.lax.map rather than vmap: q is a free output axis, so splitting it
    costs nothing, while vmap would hold one rho per q-point on the device
    at once -- and rho is the large array here. Same reasoning as
    kpmtk/kpmjax.py's chunked batch dispatch."""
    return jax.lax.map(lambda jk: _chi0_one_q(ck,ek,occ,jk,allowed,cdt),
                       ikq.T) # (nq,nk) -> one row of k+q indices per q


def polarizability_gpu(ck,ek,occ,ikq,allowed,chi_prec="single"):
    """Drop-in device replacement for screening.polarizability_jit: same
    arguments, same (nq,norb,norb) complex128 return value, already
    divided by the number of k-points.

    chi_prec sets the precision of the form factor and of the GEMM, which
    is where the time goes. The eigenstates, the occupations and the
    weights stay in double precision either way: they are cheap, and doing
    them in double keeps single precision's error at the rounding of the
    contraction rather than compounding it with eigenvector error."""
    _,cdt = _chi_dtypes(chi_prec)
    out = _chi0_all_q(jnp.asarray(ck,dtype=cdt),
                      jnp.asarray(ek,dtype=np.float64),
                      jnp.asarray(occ,dtype=np.float64),
                      jnp.asarray(ikq),
                      jnp.asarray(np.asarray(allowed,dtype=bool)),cdt)
    # 1/N, N the number of unit cells, as the CPU kernel does
    return np.array(out,dtype=np.complex128)/np.shape(ek)[0]
