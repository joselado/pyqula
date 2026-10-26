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

from ..kpmtk import scaleguard

tol = 1e-9 # truncation of the series, |q|^N, relative to its first term
max_moments = 200000 # the series is refused beyond this


def spectral_bounds(f,margin=0.02):
    """Center c and half width s of an interval holding the spectrum of
    the Hermitian block f (a factorize.KernelFactorization). The extreme
    eigenvalues come from Lanczos, which is reliable for a value even when
    the edge is degenerate, and a Ritz value always lies inside the true
    interval, hence the margin"""
    n = f.npair
    if n<=64:
        es = np.linalg.eigvalsh(f.to_dense())
        emin,emax = es[0],es[-1]
    elif f.nterm==0: # the diagonal alone
        emin,emax = np.min(f.diag),np.max(f.diag)
    else:
        import scipy.sparse.linalg as sla
        from .iterative import as_linear_operator
        op = as_linear_operator(f)
        v0 = np.ones(n,dtype=np.complex128) # deterministic start
        emax = sla.eigsh(op,k=1,which="LA",v0=v0,tol=1e-8,
                return_eigenvectors=False)[0]
        emin = sla.eigsh(op,k=1,which="SA",v0=v0,tol=1e-8,
                return_eigenvectors=False)[0]
    c = 0.5*(emax+emin)
    s = 0.5*(emax-emin)*(1.+margin) + margin*(abs(c)+1e-3)
    return c,s


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


def _recursion(matvec,D,c,s,N):
    """Yield blocks of Chebyshev vectors T_n(A') D, as (n0,V) with V of
    shape (nb,npair,ncol), checking the diagonal moments <d|T_n|d> against
    their bound |d|^2, which only fails if the spectrum is not inside
    [c - s, c + s]"""
    norm2 = np.sum(np.abs(D)**2,axis=0)
    ok = norm2>0.
    def check(V):
        if ok.all(): mus = np.einsum("pc,npc->nc",np.conj(D),V)
        else: mus = np.einsum("pc,npc->nc",np.conj(D[:,ok]),V[:,:,ok])
        if scaleguard.moments_within_bound(mus/norm2[ok][None,:]): return
        raise ValueError("the Chebyshev expansion of the BSE matrix "
            "diverges: its spectrum is not inside the interval [%g,%g] "
            "found by Lanczos, and the moments grow past their bound. Use "
            "method='dense' for this case"%(c-s,c+s))
    nb = 32 # vectors per block, so the sums over n are matrix products
    vm = D
    v = (matvec(D) - c*D)/s
    block = [vm,v]
    n0 = 0
    for n in range(2,N):
        vp = 2.*(matvec(v) - c*v)/s - vm
        vm,v = v,vp
        block.append(vp)
        if len(block)==nb:
            V = np.array(block) ; check(V)
            yield n0,V
            n0 += nb ; block = []
    if len(block)>0:
        V = np.array(block) ; check(V)
        yield n0,V


def resolvent_apply(matvec,D,zs,c,s):
    """(z - A)^-1 D for every z of zs, D of shape (npair,ncol). Returns
    (npair,ncol,nz)"""
    D = np.asarray(D,dtype=np.complex128)
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


def resolvent_braket(matvec,D,zs,c,s):
    """D_a^+ (z - A)^-1 D_b for every z of zs, from the moments
    <D_a|T_n(A')|D_b> of one recursion. Returns (nz,ncol,ncol)"""
    D = np.asarray(D,dtype=np.complex128)
    r,q = _coefficients(zs,c,s)
    N = number_of_moments(zs,c,s)
    ncol = D.shape[1]
    out = np.zeros((len(r),ncol,ncol),dtype=np.complex128)
    for (n0,V) in _recursion(matvec,D,c,s,N):
        ns = np.arange(n0,n0+len(V))
        g = 2.*q[None,:]**ns[:,None]/r[None,:] # (nb,nz)
        if n0==0: g[0] = g[0]/2.
        mus = np.einsum("pa,npb->nab",np.conj(D),V) # (nb,ncol,ncol)
        out += np.einsum("nz,nab->zab",g,mus)
    return out/s
