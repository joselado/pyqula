# Resolvent of the Tamm-Dancoff BSE block, (z - A)^-1, applied to vectors
# through a Chebyshev expansion of A at complex energy, so that a response
# summed over every exciton is obtained from matrix-vector products alone,
# without diagonalizing A.
#
# With the spectrum of A inside [c - s, c + s] and A' = (A - c)/s,
#
#   (z - A)^-1 = (1/s) sum_n (2 - delta_n0) q^n/r T_n(A') ,
#   z' = (z - c)/s ,  r = sqrt(z'^2 - 1) ,  q = z' - r ,  |q| < 1
#
# (Weisse et al., arXiv:cond-mat/0504627, continued to complex energy as in
# Ferreira and Mucciolo, arXiv:1507.00488). The imaginary part of z damps
# the series by itself, the coefficients fall as |q|^n, so no Jackson
# kernel is needed and the converged result is the Lorentzian-broadened
# resolvent of the dense route, exactly. The number of moments is roughly
# ln(1/tol) s/delta and does not depend on the size of the matrix. A
# polynomial of A never selects an eigenvector, so the degenerate exciton
# multiplets that a Krylov eigensolver cannot resolve play no role here.
import numpy as np
from numba import njit, prange

from ..kpmtk import scaleguard

tol = 1e-9 # truncation of the series, |q|^N, relative to its first term
max_moments = 200000 # the series is refused beyond this


def spectral_bounds(f,margin=0.02,lanczos=None):
    """Center c and half width s of an interval holding the spectrum of
    the Hermitian block f (a convolution.ConvolutionKernel or a
    factorize.KernelFactorization). The extreme eigenvalues come from one
    Lanczos run, which gives both edges at once and is reliable for a
    value even when the edge is degenerate, and a Ritz value always lies
    inside the true interval, hence the margin. lanczos replaces the host
    recursion, for a device copy of the block (see lanczos_edges)"""
    n = f.npair
    if n<=64:
        es = np.linalg.eigvalsh(f.to_dense())
        emin,emax = es[0],es[-1]
    elif f.nterm==0: # the diagonal alone
        emin,emax = np.min(f.diag),np.max(f.diag)
    else:
        if lanczos is None: lanczos = host_lanczos(f.matvec,n)
        emin,emax = lanczos_edges(lanczos)
    c = 0.5*(emax+emin)
    s = 0.5*(emax-emin)*(1.+margin) + margin*(abs(c)+1e-3)
    return c,s


def start_vector(n):
    """The deterministic start of the Lanczos run. A fixed-seed random
    vector rather than a constant one, which a symmetry of the block can
    make orthogonal to the eigenvector of an edge"""
    rng = np.random.default_rng(0)
    v = rng.normal(size=n) + 1j*rng.normal(size=n)
    return v/np.linalg.norm(v)


def host_lanczos(matvec,n):
    """A Lanczos recursion on the host without reorthogonalization, as a
    function of the number of steps m that returns the next m diagonal
    and off-diagonal elements of the tridiagonal matrix. The loss of
    orthogonality only duplicates converged Ritz values, which leaves the
    extreme ones untouched"""
    v = start_vector(n)
    state = [v,np.zeros_like(v),0.]
    def run(m):
        v,vm,b = state
        als,bes = np.zeros(m),np.zeros(m)
        for i in range(m):
            w = matvec(v) - b*vm
            als[i] = np.real(np.vdot(v,w))
            w -= als[i]*v
            bes[i] = b = np.linalg.norm(w)
            if b==0.: break # an invariant subspace, nothing more to find
            vm,v = v,w/b
        state[:] = v,vm,b
        return als,bes
    return run


def lanczos_edges(run,nb=16,maxsteps=400,rtol=1e-5):
    """The lowest and highest Ritz values of a Lanczos run, extended by
    about nb steps at a time (run may return a different number) until
    both move by less than rtol of the width
    between two extensions. On the TMD tetralayer at nk=48 that took 50
    steps and put the edges within 2e-5 of ARPACK's, far inside the margin
    of spectral_bounds, at 0.055 s on an RTX A2000 in single precision
    against 1.9 s for the two ARPACK runs it replaced, whose host work was
    most of the cost. The scale guard of the recursion (check_moments) is
    the backstop if an edge is missed"""
    from scipy.linalg import eigvalsh_tridiagonal
    als,bes,old = [],[],None
    while len(als)<maxsteps:
        a,b = run(nb)
        als,bes = np.concatenate([als,a]),np.concatenate([bes,b])
        if np.any(b==0.): # invariant subspace, the edges are exact
            m = len(als) - len(b) + int(np.argmax(b==0.)) + 1
            als,bes = als[:m],bes[:m]
        es = eigvalsh_tridiagonal(als,bes[:-1]) if len(als)>1 else als
        e0,e1 = es[0],es[-1]
        if np.any(b==0.): break
        if old is not None and max(abs(e0-old[0]),abs(e1-old[1])) \
                <= rtol*max(e1-e0,1e-12): break
        old = (e0,e1)
    return e0,e1


def _coefficients(zs,c,s):
    """r and q of every complex energy, on the branch |q|<1"""
    zp = (np.asarray(zs,dtype=np.complex128) - c)/s
    r = np.sqrt(zp*zp-1.+0j)
    q = zp - r
    bad = np.abs(q)>1.
    r[bad] = -r[bad] ; q[bad] = zp[bad] - r[bad]
    return r,q


def number_of_moments(zs,c,s):
    """Moments needed for the series to fall below tol at every energy"""
    r,q = _coefficients(zs,c,s)
    qmax = np.max(np.abs(q))
    if not qmax<1.: raise ValueError("the Chebyshev resolvent needs every "
            "energy off the real axis, the broadening delta must be "
            "positive")
    n = int(np.ceil(np.log(tol)/np.log(qmax))) + 1
    if n>max_moments: raise ValueError("the Chebyshev resolvent would need "
            "%d moments at this broadening, more than max_moments=%d; the "
            "number grows as the spectral width over delta, so use a larger "
            "delta or the dense route"%(n,max_moments))
    return max(n,2)


def check_moments(mus,norm2,c,s,kpm_prec="double"):
    """Refuse if a diagonal moment <d|T_n(A')|d>, mus of shape (n,ncol),
    exceeds its bound |d|^2 = norm2, which only happens if the spectrum is
    not inside [c - s, c + s]"""
    ok = norm2>0.
    if scaleguard.moments_within_bound(mus[:,ok]/norm2[ok][None,:],
            kpm_prec=kpm_prec): return
    raise ValueError("the Chebyshev expansion of the BSE matrix "
        "diverges: its spectrum is not inside the interval [%g,%g] "
        "found by Lanczos, and the moments grow past their bound. Use "
        "method='dense' for this case"%(c-s,c+s))


def _step(matvec,v,vm,c,s):
    """The next Chebyshev vector, 2 (A v - c v)/s - vm, written in place
    on the output of the matvec, so that no other vector-sized array is
    made: a fresh array of this size costs its page faults on every step"""
    vp = np.ascontiguousarray(matvec(v))
    _combine(vp.reshape(-1),v.reshape(-1),vm.reshape(-1),c,s)
    return vp


@njit(parallel=True,cache=True)
def _combine(vp,v,vm,c,s):
    """vp <- 2 (vp - c v)/s - vm, elementwise"""
    for i in prange(len(vp)):
        vp[i] = 2.*(vp[i] - c*v[i])/s - vm[i]


def _recursion(matvec,D,c,s,N):
    """Yield blocks of Chebyshev vectors T_n(A') D, as (n0,V) with V of
    shape (nb,npair,ncol), checking the diagonal moments <d|T_n|d> against
    their bound |d|^2, which only fails if the spectrum is not inside
    [c - s, c + s]"""
    norm2 = np.sum(np.abs(D)**2,axis=0)
    def check(V): check_moments(np.einsum("pc,npc->nc",np.conj(D),V),
            norm2,c,s)
    nb = 32 # vectors per block, so the sums over n are matrix products
    # one buffer, refilled for every block, so the consumer has to be done
    # with a block before asking for the next one
    V = np.empty((min(nb,N),)+D.shape,dtype=np.complex128)
    vm = D
    v = (matvec(D) - c*D)/s
    V[0],V[1] = vm,v
    i,n0 = 2,0
    for n in range(2,N):
        if i==len(V):
            check(V) ; yield n0,V
            n0 += i ; i = 0
        vp = _step(matvec,v,vm,c,s)
        vm,v = v,vp
        V[i] = vp ; i += 1
    check(V[:i]) ; yield n0,V[:i]


def resolvent_apply(matvec,D,zs,c,s):
    """(z - A)^-1 D for every z of zs, D of shape (npair,ncol). Returns
    (npair,ncol,nz)"""
    D = np.ascontiguousarray(D,dtype=np.complex128) # and so every v_n
    r,q = _coefficients(zs,c,s)
    N = number_of_moments(zs,c,s)
    npair,ncol = D.shape
    out = np.zeros((npair*ncol,len(r)),dtype=np.complex128)
    for (n0,V) in _recursion(matvec,D,c,s,N):
        ns = np.arange(n0,n0+len(V))
        g = 2.*q[None,:]**ns[:,None]/r[None,:] # (nb,nz)
        if n0==0: g[0] = g[0]/2.
        out += V.reshape(len(V),-1).T@g
    return out.reshape(npair,ncol,len(r))/s


def _inner(a,b):
    """a^+ b for two C-ordered (npair,ncol) arrays, by BLAS with the
    conjugation inside it rather than on a copy of a"""
    from scipy.linalg.blas import zgemm
    return zgemm(1.,b.T,a.T,trans_b=2).T


def resolvent_braket(matvec,D,zs,c,s):
    """D_a^+ (z - A)^-1 D_b for every z of zs, from the moments
    mu_n = <D_a|T_n(A')|D_b> of one recursion. Returns (nz,ncol,ncol).

    A' is Hermitian, so T_m T_n = (T_{m+n} + T_{|m-n|})/2 gives every
    moment from vectors of half the order,

      mu_2n = 2 <v_n|v_n> - mu_0 ,  mu_2n+1 = 2 <v_n+1|v_n> - mu_1 ,

    with v_n = T_n(A') D, which halves the number of matvecs. Only the
    last two vectors are kept, each new one giving two ncol x ncol
    products"""
    D = np.ascontiguousarray(D,dtype=np.complex128) # and so every v_n
    r,q = _coefficients(zs,c,s)
    N = number_of_moments(zs,c,s)
    M = N//2 + 1 # vectors v_0 ... v_M-1 give the moments up to 2M-1 >= N
    ncol = D.shape[1]
    mus = np.zeros((2*M,ncol,ncol),dtype=np.complex128)
    norm2 = np.sum(np.abs(D)**2,axis=0)
    vm = D
    v = (matvec(D) - c*D)/s
    mu0,mu1 = np.conj(D).T@D,np.conj(v).T@D
    mus[0],mus[1],mus[2] = mu0,mu1,2.*(np.conj(v).T@v) - mu0
    n0 = 2 # first moment index of T_2n not yet checked
    for n in range(2,M):
        vp = _step(matvec,v,vm,c,s)
        mus[2*n] = 2.*_inner(vp,vp) - mu0
        mus[2*n-1] = 2.*_inner(vp,v) - mu1
        vm,v = v,vp
        if n%32==0 or n==M-1: # the diagonal moments <d|T_2n|d>
            check_moments(np.einsum("naa->na",mus[2*n0:2*n+1:2]),norm2,c,s)
            n0 = n+1
    ns = np.arange(2*M)
    g = 2.*q[None,:]**ns[:,None]/r[None,:] # (2M,nz)
    g[0] = g[0]/2.
    return np.einsum("nz,nab->zab",g,mus)/s
