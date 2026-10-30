from .. import gpu
# the package-wide CPU/GPU switch decides where this module's kernels run
# (pyqula/gpu.py); applying it here means importing this module does not
# leave the placement to jax's own default
gpu.apply()

# The Chebyshev resolvent of the Tamm-Dancoff BSE block on the device, the
# jax counterpart of bsetk/chebyshev.py (resolvent_apply and
# resolvent_braket) with the convolution matvec of bsetk/convolution.py
# written out in jax: the gather of the pair vectors into orbital matrices
# and the scatter back are batched contractions over the mesh, and the
# direct term is a device FFT. The recursion runs in blocks of nb steps
# inside one compiled scan, and each block is folded into the result with
# one matrix product, so the host only sees the diagonal moments of the
# scale guard.
#
# Every contraction asks for Precision.HIGHEST. Without it jax runs
# float32 products on an Ampere or later card in TF32, with a ten-bit
# mantissa, which put the single precision resolvent 8e-4 away from the
# double one on an RTX A2000 against 4e-6 on jax's CPU backend.

import numpy as np
from functools import partial
import jax
import jax.numpy as jnp

from . import chebyshev

_DTYPES = {"single":(jnp.complex64,jnp.float32),
        "double":(jnp.complex128,jnp.float64)}
_HI = jax.lax.Precision.HIGHEST
nb = 16 # recursion steps per compiled block


assert set(_DTYPES)==set(chebyshev._precisions) # one name per precision


def get_precision_names():
    """Every precision the kernels of this module accept"""
    return chebyshev.get_precision_names()


def _check_precision(kpm_prec): chebyshev.check_precision(kpm_prec)


def _block(f,kpm_prec):
    """The block A of a convolution.ConvolutionKernel f on the device, as
    (static,arrays): the shapes, a hashable tuple, and the device arrays,
    a dict. The jitted kernels below take both as arguments rather than
    closing over the arrays, so that jax keeps one compiled kernel per
    shape and precision instead of tracing and compiling the block again
    for every new f, which was 0.45 s for the Lanczos run and 0.7 s for the
    recursion of every call on the TMD tetralayer at nk=48 in single
    precision"""
    ct,rt = _DTYPES[kpm_prec]
    nk,nv,nc = f.shape
    norb = f.Cv.shape[2]
    static = (nk,nv,nc,norb,tuple(f.nks),f.ndirect,f.nterm==0,
            f.direct is not None,f.exchange is not None)
    arrays = dict(Cc=jnp.asarray(np.swapaxes(f.CcT,1,2),dtype=ct), # (nk,nc,norb)
            Cvc=jnp.asarray(np.conj(f.Cv),dtype=ct), # (nk,nv,norb)
            Ccc=jnp.asarray(f.Ccc,dtype=ct),Cv=jnp.asarray(f.Cv,dtype=ct),
            diag=jnp.asarray(f.diag,dtype=rt),
            chans=jnp.asarray(f.ia*norb + f.ib)) # the flat channel (a,b)
    if f.direct is not None: arrays["Wf"] = jnp.asarray(f.direct,dtype=ct)
    if f.exchange is not None: arrays["WQ"] = jnp.asarray(f.exchange,dtype=ct)
    return static,arrays


def _mv(static,A,X):
    """The block applied to a (npair,ncol) array. Unlike the numba kernels
    of convolution.py, the orbital matrices are gathered and scattered
    whole, as two dense contractions each, and the channels of the
    interaction are only picked for the FFT: a channel-by-channel scatter
    lowers to atomic adds on the device, and was three times slower in
    single precision on a consumer card for a TMD tetralayer"""
    nk,nv,nc,norb,nks,nd,diagonal,direct,exchange = static
    if diagonal: return A["diag"][:,None]*X
    axes = tuple(range(len(nks)))
    chans = A["chans"]
    idiag = jnp.arange(norb)*(norb+1) # the flat channels (a,a)
    ncol = X.shape[1]
    Xk = X.reshape(nk,nv,nc,ncol)
    # Z[k,n,a,b] = sum_vc Cc[k,c,a] X[k,v,c,n] conj(Cv[k,v,b])
    T = jnp.einsum("kvcn,kca->knva",Xk,A["Cc"],precision=_HI)
    Z = jnp.einsum("knva,kvb->knab",T,A["Cvc"],precision=_HI)
    Z = Z.reshape(nk,ncol,norb*norb)
    M = jnp.zeros_like(Z)
    if direct: # the direct term, a convolution over the mesh
        Zr = jnp.take(Z,chans[:nd],axis=2).reshape(nks+(ncol,nd))
        Y = jnp.fft.ifftn(jnp.fft.fftn(Zr,axes=axes)*A["Wf"],axes=axes)
        M = M.at[:,:,chans[:nd]].set(Y.reshape(nk,ncol,nd))
    if exchange: # the exchange term, on the orbital diagonal
        rho = jnp.sum(Z[:,:,idiag],axis=0)
        M = M.at[:,:,idiag].add(jnp.matmul(rho,A["WQ"].T,precision=_HI)[None])
    # back onto the pairs, <C_c(k)|M(k)|C_v(k)>
    M = M.reshape(nk,ncol,norb,norb)
    S = jnp.einsum("knab,kvb->knva",M,A["Cv"],precision=_HI)
    Y = jnp.einsum("kca,knva->kvcn",A["Ccc"],S,precision=_HI)
    return A["diag"][:,None]*X + Y.reshape(X.shape)


@partial(jax.jit,static_argnums=0)
def _lanczos_block(static,A,v,vm,b):
    """nb Lanczos steps, without reorthogonalization"""
    def step(carry,_):
        v,vm,b = carry
        w = _mv(static,A,v[:,None])[:,0] - b*vm
        a = jnp.real(jnp.vdot(v,w))
        w = w - a*v
        bn = jnp.linalg.norm(w)
        vn = jnp.where(bn>0.,w/jnp.where(bn>0.,bn,1.),0.)
        return (vn,v,bn),(a,bn)
    return jax.lax.scan(step,(v,vm,b),None,length=nb)


@partial(jax.jit,static_argnums=0)
def _first(static,A,D,c,s):
    return (_mv(static,A,D) - c*D)/s


def _steps(static,A,vm,v,c,s):
    def step(carry,_):
        vm,v = carry
        vp = 2.*(_mv(static,A,v) - c*v)/s - vm
        return (v,vp),vp
    return jax.lax.scan(step,(vm,v),None,length=nb)


@partial(jax.jit,static_argnums=0)
def _block_apply(static,A,vm,v,out,g,D,c,s):
    """nb more vectors, folded into out with their coefficients g"""
    (vm,v),B = _steps(static,A,vm,v,c,s)
    out = out + jnp.matmul(B.reshape(nb,-1).T,g,precision=_HI)
    mus = jnp.einsum("pc,npc->nc",jnp.conj(D),B,precision=_HI)
    return vm,v,out,mus


@partial(jax.jit,static_argnums=0)
def _block_moments(static,A,vm,v,c,s):
    """nb more vectors, reduced to <v_n|v_n> and <v_n|v_n-1>"""
    (vmn,vn),B = _steps(static,A,vm,v,c,s)
    Bm = jnp.concatenate([v[None],B[:-1]]) # the vector before each
    sq = jnp.einsum("npa,npb->nab",jnp.conj(B),B,precision=_HI)
    cross = jnp.einsum("npa,npb->nab",jnp.conj(B),Bm,precision=_HI)
    return vmn,vn,sq,cross


def _kernels(f,kpm_prec):
    """The first step, recursion blocks and Lanczos block bound to the
    device copy of the block, kept on f so that the bounds and the
    frequency batches of one calculation share one copy"""
    if not hasattr(f,"_jax_kernels"): f._jax_kernels = dict()
    if kpm_prec in f._jax_kernels: return f._jax_kernels[kpm_prec]
    static,A = _block(f,kpm_prec)
    f._jax_kernels[kpm_prec] = (partial(_first,static,A),
            partial(_block_apply,static,A),partial(_block_moments,static,A),
            partial(_lanczos_block,static,A))
    return f._jax_kernels[kpm_prec]


def lanczos(f,kpm_prec="double"):
    """The Lanczos recursion of the block on the device, in the form
    chebyshev.lanczos_edges takes: the next nb tridiagonal elements per
    call, whatever number of steps is asked for, so that only those reach
    the host. The ARPACK runs this replaced
    applied the device block to host vectors and spent most of their time
    in host work, 1.9 s against 0.055 s for this on the TMD tetralayer at
    nk=48 in single precision"""
    _check_precision(kpm_prec)
    ct,rt = _DTYPES[kpm_prec]
    lanczos_block = _kernels(f,kpm_prec)[3]
    v = jnp.asarray(chebyshev.start_vector(f.npair),dtype=ct)
    state = [v,jnp.zeros_like(v),jnp.asarray(0.,dtype=rt)]
    def run(m): # always the nb steps of one compiled block
        (v,vm,b),(a,bs) = lanczos_block(*state)
        state[:] = v,vm,b
        return np.asarray(a,dtype=np.float64),np.asarray(bs,dtype=np.float64)
    return run


def _coefficients(zs,c,s,n):
    """The coefficients of T_0 ... T_n-1 for every energy, with the 1/s of
    the resolvent folded in"""
    r,q = chebyshev._coefficients(zs,c,s)
    return chebyshev._series(r,q,np.arange(n))/s


def resolvent_apply(f,D,zs,c,s,kpm_prec="double"):
    """(z - A)^-1 D for every z of zs, D of shape (npair,ncol), on the
    device. Returns (npair,ncol,nz) on the host"""
    _check_precision(kpm_prec)
    ct,rt = _DTYPES[kpm_prec]
    first,block_apply,block_moments = _kernels(f,kpm_prec)[0:3]
    N = chebyshev.number_of_moments(zs,c,s)
    nblock = int(np.ceil(max(N-2,1)/nb)) # whole blocks, a few more terms
    g = jnp.asarray(_coefficients(zs,c,s,2+nblock*nb),dtype=ct)
    D = np.asarray(D,dtype=np.complex128)
    npair,ncol = D.shape
    norm2 = np.sum(np.abs(D)**2,axis=0)
    cj,sj = rt(c),rt(s) # a float64 constant would promote the vectors
    Dj = jnp.asarray(D,dtype=ct)
    v = first(Dj,cj,sj)
    out = jnp.outer(Dj.reshape(-1),g[0]) + jnp.outer(v.reshape(-1),g[1])
    vm = Dj
    for b in range(nblock):
        vm,v,out,mus = block_apply(vm,v,out,g[2+b*nb:2+(b+1)*nb],Dj,cj,sj)
        chebyshev.check_moments(np.asarray(mus),norm2,c,s,kpm_prec=kpm_prec)
    return np.asarray(out,dtype=np.complex128).reshape(npair,ncol,len(zs))


def resolvent_braket(f,D,zs,c,s,kpm_prec="double"):
    """D_a^+ (z - A)^-1 D_b for every z of zs, on the device, from vectors
    of half the order as in chebyshev.resolvent_braket. Returns
    (nz,ncol,ncol) on the host"""
    _check_precision(kpm_prec)
    ct,rt = _DTYPES[kpm_prec]
    first,block_apply,block_moments = _kernels(f,kpm_prec)[0:3]
    N = chebyshev.number_of_moments(zs,c,s)
    nblock = int(np.ceil(max(N//2+1-2,1)/nb))
    M = 2+nblock*nb # vectors v_0 ... v_M-1, moments up to 2M-2
    D = np.asarray(D,dtype=np.complex128)
    ncol = D.shape[1]
    norm2 = np.sum(np.abs(D)**2,axis=0)
    cj,sj = rt(c),rt(s)
    Dj = jnp.asarray(D,dtype=ct)
    v = first(Dj,cj,sj)
    mu0 = np.asarray(jnp.conj(Dj).T@Dj,dtype=np.complex128)
    mu1 = np.asarray(jnp.conj(v).T@Dj,dtype=np.complex128) # <v_1|v_0>
    mus = np.zeros((2*M-1,ncol,ncol),dtype=np.complex128)
    mus[0],mus[1] = mu0,mu1
    sq1 = np.asarray(jnp.conj(v).T@v,dtype=np.complex128)
    mus[2] = 2.*sq1 - mu0
    vm = Dj
    for b in range(nblock):
        vm,v,sq,cross = block_moments(vm,v,cj,sj)
        sq = np.asarray(sq,dtype=np.complex128)
        cross = np.asarray(cross,dtype=np.complex128)
        # the diagonal moments of the scale guard, <d|T_2n|d> from <v_n|v_n>
        diag = 2.*np.einsum("naa->na",sq) - np.diag(mu0)[None,:]
        chebyshev.check_moments(diag,norm2,c,s,kpm_prec=kpm_prec)
        ns = np.arange(2+b*nb,2+(b+1)*nb)
        mus[2*ns] = 2.*sq - mu0
        mus[2*ns-1] = 2.*cross - mu1
    g = _coefficients(zs,c,s,2*M-1)
    return np.einsum("nz,nab->zab",g,mus)
