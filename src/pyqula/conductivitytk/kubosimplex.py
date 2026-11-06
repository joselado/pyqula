# Simplex (linear triangle) integration of the Kubo-Greenwood optical
# conductivity, and its quantics tensor cross interpolation (QTCI) variant.
# The formula, the units and the sign conventions are those of kubo.py and
# conductivity.py; this module only replaces the k-sum.
#
# The grid method of kubo.py samples 1/(omega + i delta - dE_nm(k)) on the
# mesh, so it needs a mesh whose energy spacing is below delta. Here the
# transition energy dE_nm(k) and the weight W^ab_nm(k) = ratio_nm v^a_nm v^b_mn
# are interpolated linearly inside each simplex of the mesh (segments in 1D,
# triangles in 2D), and the resolvent is integrated over the simplex
# exactly. The mesh then only has to resolve the smooth functions dE and W,
# not the broadening, and every frequency comes from the same mesh.
#
# The simplex average of a function phi(e(k)) with e linear in k is given by
# the Hermite-Genocchi formula, d! times the d-th divided difference of the
# d-th antiderivative of phi over the vertex energies. For phi(E) = 1/(z-E),
# z = omega + i delta, the antiderivatives are, with x = E - z,
#   psi_1 = -ln x,  psi_2 = -(x ln x - x).
# Im x = -delta < 0 for every real E, so the principal logarithm is
# continuous along the real axis and there is no branch to track. This is
# the 2D analogue of the analytic tetrahedron method for complex energies
# of Ph. Lambin and J.-P. Vigneron, PRB 29, 3430 (1984).
#
# The weights are averaged over the vertices of each simplex, and over
# degenerate multiplets at each vertex: a single W_nm inside a degenerate
# multiplet depends on the basis chosen there (numba's eigh picks its own),
# while the multiplet average is basis independent and continuous in k.
import numpy as np
from numba import jit, prange

from . import kubo
from .. import parallel
# numba's cache does not see edits in another file: after a change to
# simplexdos._dd, clear __pycache__ or touch this file
from ..dostk.simplexdos import _dd, _sort_small
from ..dostk.simplexdos import _harm as _dos_harm, _fact as _dos_fact
from ..dostk.simplexdos import _nterms as _dos_nterms


def check_broadening(delta):
    """The simplex integral needs a finite broadening: at delta = 0 the
    logarithm has its branch point on the real axis."""
    if not delta>0.:
        raise ValueError("the triangle and qtci integrations of the optical "
                "conductivity need a positive broadening delta, got delta="
                +str(delta))


# ------------------------------------------------------------ pair fields

def _active_axes(vs):
    """Cartesian axes with a nonzero velocity anywhere on the mesh (the
    velocity along a non-periodic direction is identically zero)"""
    return [a for a in range(3) if np.max(np.abs(vs[:,a]))>0.]


def _components(axes):
    """Cartesian pairs (a,b) kept, and for each the index of (b,a)"""
    comps = [(a,b) for a in axes for b in axes]
    trans = np.array([comps.index((b,a)) for (a,b) in comps],dtype=np.int64)
    return comps,trans


@jit(nopython=True,parallel=True,cache=True)
def _pair_fields_jit(es,ratio,vs,pairs,ca,cb,tol):
    """Transition energies dE[k,p] = E_m - E_n and weights
    W[k,p,c] = ratio_nm v^a_nm v^b_mn for the stored pairs p = (n,m), n <= m,
    each averaged over the degenerate multiplets of n and of m at that k"""
    nk = es.shape[0]
    nb = es.shape[1]
    npair = pairs.shape[0]
    nc = ca.shape[0]
    dE = np.zeros((nk,npair),dtype=np.float64)
    W = np.zeros((nk,npair,nc),dtype=np.complex128)
    for ik in prange(nk): # loop over k-points
        # first band of the multiplet that each band belongs to
        start = np.zeros(nb,dtype=np.int64)
        for i in range(1,nb):
            if es[ik,i]-es[ik,i-1]<tol: start[i] = start[i-1]
            else: start[i] = i
        stop = np.zeros(nb,dtype=np.int64) # one past the last band
        stop[nb-1] = nb
        for i in range(nb-2,-1,-1):
            if start[i+1]==start[i]: stop[i] = stop[i+1]
            else: stop[i] = i+1
        for p in range(npair): # loop over stored pairs
            n = pairs[p,0]
            m = pairs[p,1]
            dE[ik,p] = es[ik,m] - es[ik,n]
            norm = 1./((stop[n]-start[n])*(stop[m]-start[m]))
            for c in range(nc): # loop over Cartesian pairs
                acc = 0j
                for i in range(start[n],stop[n]):
                    for j in range(start[m],stop[m]):
                        acc += ratio[ik,i,j]*vs[ik,ca[c],i,j]*vs[ik,cb[c],j,i]
                W[ik,p,c] = acc*norm
    return dE,W


def pair_fields(es,vs,T,tol,intraband,interband,axes=None,pairs=None):
    """Smooth fields on a set of k-points: the stored band pairs, their
    transition energies dE[k,p] and multiplet-averaged weights W[k,p,c],
    and the Cartesian components. The pair (m,n) is not stored: its energy
    is -dE_nm and its weight for (a,b) is W_nm for (b,a)."""
    ratio,dEfull = kubo._response_weights(es,T,tol,
            intraband=intraband,interband=interband)
    nb = es.shape[1]
    if axes is None: axes = _active_axes(vs)
    comps,trans = _components(axes)
    if pairs is None:
        pairs = np.array([(n,m) for n in range(nb) for m in range(n,nb)],
                dtype=np.int64)
    ca = np.array([c[0] for c in comps],dtype=np.int64)
    cb = np.array([c[1] for c in comps],dtype=np.int64)
    parallel.set_num_threads() # set the number of threads for numba
    dE,W = _pair_fields_jit(es,np.ascontiguousarray(ratio),vs,pairs,ca,cb,tol)
    return pairs,dE,W,comps,trans


def _prune(pairs,dE,W,rtol=1e-12):
    """Drop the pairs whose weight vanishes everywhere, e.g. two occupied
    bands of an insulator, whose occupation factor is exponentially small"""
    wmax = np.max(np.abs(W),axis=(0,2)) if W.size else np.zeros(0)
    keep = wmax>rtol*max(np.max(wmax),1e-300)
    return pairs[keep],dE[:,keep],W[:,keep]


# ------------------------------------------------------------ simplex kernel
#
# Both kernels evaluate the divided differences of psi_d exactly, or by a
# Taylor series around the mean of the nodes when they are close. What
# "close" means sets the precision. A plain divided difference over nodes
# of spread s, at a distance |x| = |c - z| from z, cancels terms and loses
# digits like eps (|x|/s)^d, so for a nearly flat simplex far from
# resonance it fails even when s is much larger than delta. A spread
# compared with delta alone (the rule used here before) gave a relative
# error of 0.1 for a triangle of spread 3e-8 at |x| = 0.5 and delta = 1e-6.
# The second order series used below that spread also took the variance
# of the uncentered energies, which loses eps c^2/delta^2 near resonance.
#
# The direct kernel works frequency by frequency, so it uses the rule of
# dostk/simplexdos.py directly: every cluster of nodes whose spread is
# below 0.05 |c - z| goes to its Taylor series, up to 14 terms.

@jit(nopython=True,parallel=True,cache=True)
def _sigma_simplex_jit(simplices,dE,W,trans,diag,omegas,delta,harm,fact):
    """Sum over simplices and stored pairs of the vertex-averaged weight
    times the simplex average of the resolvent, for every frequency and
    every Cartesian pair, with the divided differences of
    simplexdos._dd. As in kubo._sigma_jit the parallel loop runs over
    frequencies, so each thread owns one output row. Not normalized: the
    caller multiplies by i, the simplex volume fraction and 1/V_cell."""
    ns = simplices.shape[0]
    nv = simplices.shape[1] # vertices per simplex, 2 or 3
    npair = dE.shape[1]
    nc = W.shape[2]
    nw = omegas.shape[0]
    out = np.zeros((nw,nc),dtype=np.complex128)
    for iw in prange(nw): # parallel loop over frequencies
        w = omegas[iw]
        t = np.zeros(nv+1)
        hbuf = np.zeros(_dos_nterms+1)
        d = np.zeros(nv+1,dtype=np.complex128)
        acc = np.zeros(nc,dtype=np.complex128)
        wbar = np.zeros(nc,dtype=np.complex128)
        for it in range(ns): # loop over simplices
            for p in range(npair): # loop over stored pairs
                for c in range(nc): wbar[c] = 0.
                for v in range(nv):
                    for c in range(nc): wbar[c] += W[simplices[it,v],p,c]
                for v in range(nv): t[v] = dE[simplices[it,v],p]-w
                _sort_small(t,nv)
                sf = fact[nv-1]*_dd(nv-1,t,nv,-delta,harm,fact,hbuf,d)
                for c in range(nc): acc[c] += wbar[c]*sf
                if not diag[p]: # the pair (m,n), at energy -dE
                    for v in range(nv): t[v] = -dE[simplices[it,v],p]-w
                    _sort_small(t,nv)
                    sr = fact[nv-1]*_dd(nv-1,t,nv,-delta,harm,fact,hbuf,d)
                    for c in range(nc): acc[c] += wbar[trans[c]]*sr
        out[iw] = acc/nv
    return out


def _sigma_simplices(simplices,dE,W,trans,diag,omegas,delta):
    """The direct sum over simplices at every frequency"""
    return _sigma_simplex_jit(simplices,dE,W,trans,diag,omegas,delta,
            _dos_harm,_dos_fact)


def simplices(dim,nk):
    """Simplices of the periodic nk (1D) or nk x nk (2D) mesh, as vertex
    indices into the row-major list of klist.kmesh, and the volume fraction
    of the Brillouin zone each one covers"""
    if dim==1:
        i = np.arange(nk)
        return np.stack([i,(i+1)%nk],axis=1).astype(np.int64),1./nk
    i,j = np.meshgrid(np.arange(nk),np.arange(nk),indexing="ij")
    i = i.ravel(); j = j.ravel()
    ip = (i+1)%nk; jp = (j+1)%nk
    k00 = i*nk+j; k10 = ip*nk+j; k01 = i*nk+jp; k11 = ip*nk+jp
    t1 = np.stack([k00,k10,k01],axis=1)
    t2 = np.stack([k10,k11,k01],axis=1)
    return np.concatenate([t1,t2]).astype(np.int64),1./(2*nk*nk)


# ------------------------------------------------------------ source kernel
#
# The default kernel decides the branch of each simplex once, for all
# frequencies, so the whole sum over simplices and pairs is a sum of fixed
# coefficients times a few functions of x = e - z:
#   psi_2(x) (psi_1(x) in 1D) at the vertex energies (exact branches),
#   psi_1(x) plus a series in (h/x)^2 at the middle of a close edge,
#   the Taylor series of a close simplex around its mean.
# In other words the weighted joint density of states, a sum of linear
# B-splines, is stored by its knots. The coefficients are built once, the
# vertex ones indexed by k-point, so the simplices sharing a vertex share
# one source, and the sum is then evaluated per frequency over the sources.
#
# The branch cannot depend on omega here, so the rule has to hold at every
# frequency at once. A simplex (or an edge) of spread s is expanded when
# s < _alpha delta. Its Taylor term of order n is bounded by
# (2 s/(3 |c - z|))^n, and |c - z| >= delta for every real omega, so the
# series converges at a rate of at least 2 _alpha/3 for any omega; it runs
# until two terms in a row are below 1e-17 of the sum, which takes up to
# about 80 terms on resonance and a few far from it. The series keeps its
# first order term, which vanishes for the exact mean, because the stored
# center is the rounded mean. What is left is the exact branch for a
# simplex of spread just above _alpha delta, evaluated at a frequency far
# from its mean, which loses eps (|x|/s)^2 of that simplex. The rule used
# before, s < 1e-2 delta with a second order series, lost 1e4 times more
# there. Only a rule per frequency removes this last loss, and that is
# what the direct kernel above does.

_alpha = 1. # see above
_nmax = 100 # largest order of the Taylor series of a point source


@jit(nopython=True,cache=True)
def _sort3i(a,b,c,ia,ib,ic):
    """sort three energies, carrying along the vertex indices"""
    if a>b:
        a,b = b,a
        ia,ib = ib,ia
    if b>c:
        b,c = c,b
        ib,ic = ic,ib
    if a>b:
        a,b = b,a
        ia,ib = ib,ia
    return a,b,c,ia,ib,ic


@jit(nopython=True,cache=True)
def _add_point(write,n,ep,tp,yp,ca,e,t,y0,y1,y2,wbar,f):
    """append a point source of type t at energy e, with shape (y0,y1,y2)
    and coefficients f*wbar; the counting pass only counts"""
    if write:
        ep[n] = e
        tp[n] = t
        yp[n,0] = y0
        yp[n,1] = y1
        yp[n,2] = y2
        for c in range(wbar.shape[0]): ca[n,c] = f*wbar[c]
    return n+1


@jit(nopython=True,cache=True)
def _simplex_sources(write,e0,e1,e2,k0,k1,k2,nv,wbar,tolc,Av,p,n,
        ep,tp,yp,ca):
    """sources of one simplex and one pair. Av[k,p,c] collects the vertex
    coefficients, the point sources go to (ep,tp,yp,ca) from index n on.
    Type 1: an edge of half-length yp[0], psi_2[m-yp[0],m+yp[0]]. Type 2:
    a simplex with the vertex energies m + yp, its Taylor series. Returns
    the new n."""
    if nv==2: # segment
        h = abs(e1-e0)
        cm = 0.5*(e0+e1)
        if h<tolc: # Taylor
            n = _add_point(write,n,ep,tp,yp,ca,cm,2,e0-cm,e1-cm,0.,wbar,1.)
        elif write: # (psi_1(e1)-psi_1(e0))/(e1-e0)
            f = 1./(e1-e0)
            for c in range(wbar.shape[0]):
                Av[k1,p,c] += f*wbar[c]
                Av[k0,p,c] -= f*wbar[c]
        return n
    e0,e1,e2,k0,k1,k2 = _sort3i(e0,e1,e2,k0,k1,k2)
    s = e2-e0
    cm = (e0+e1+e2)/3.
    if s<tolc: # Taylor
        return _add_point(write,n,ep,tp,yp,ca,cm,2,e0-cm,e1-cm,e2-cm,
                wbar,1.)
    # 2 (psi_2[e1,e2] - psi_2[e0,e1])/s
    for side in range(2):
        if side==0:
            a = e1; b = e2; ka = k1; kb = k2; sgn = 2./s
        else:
            a = e0; b = e1; ka = k0; kb = k1; sgn = -2./s
        h = b-a
        m = 0.5*(a+b)
        if h<tolc: # close edge
            n = _add_point(write,n,ep,tp,yp,ca,m,1,0.5*h,0.,0.,wbar,sgn)
        elif write: # (psi_2(b)-psi_2(a))/h
            f = sgn/h
            for c in range(wbar.shape[0]):
                Av[kb,p,c] += f*wbar[c]
                Av[ka,p,c] -= f*wbar[c]
    return n


@jit(nopython=True,cache=True)
def _sources_jit(write,simplices,dE,W,trans,diag,tolc,Af,Ar,ep,tp,yp,ca):
    """Build (write=True) or count (write=False) the sources of the whole
    simplex sum, with the clusters of spread below tolc Taylor expanded.
    Af and Ar take the vertex coefficients of the forward (energy dE) and
    reverse (energy -dE) terms. Returns the number of point sources."""
    ns = simplices.shape[0]
    nv = simplices.shape[1]
    npair = dE.shape[1]
    nc = W.shape[2]
    wbar = np.zeros(nc,dtype=np.complex128)
    wrev = np.zeros(nc,dtype=np.complex128)
    n = 0
    for it in range(ns): # loop over simplices
        k0 = simplices[it,0]
        k1 = simplices[it,1]
        k2 = simplices[it,nv-1]
        for p in range(npair): # loop over stored pairs
            for c in range(nc): wbar[c] = 0.
            for v in range(nv):
                for c in range(nc): wbar[c] += W[simplices[it,v],p,c]
            for c in range(nc): wbar[c] = wbar[c]/nv
            e0 = dE[k0,p]
            e1 = dE[k1,p]
            e2 = dE[k2,p]
            n = _simplex_sources(write,e0,e1,e2,k0,k1,k2,nv,wbar,tolc,
                    Af,p,n,ep,tp,yp,ca)
            if not diag[p]:
                for c in range(nc): wrev[c] = wbar[trans[c]]
                n = _simplex_sources(write,-e0,-e1,-e2,k0,k1,k2,nv,wrev,
                        tolc,Ar,p,n,ep,tp,yp,ca)
    return n


@jit(nopython=True,cache=True)
def _point_value(t,x,y0,y1,y2,nv):
    """value of a point source at x = m - z, see _simplex_sources"""
    small = 0
    if t==1: # psi_1(m) + sum over even n >= 2 of (y0/x)^n/(n (n+1))
        q = y0/x
        q2 = q*q
        aq = abs(q)
        pw = 1.+0j
        acc = 0j
        for n in range(2,_nmax+1,2):
            pw *= q2
            term = pw/(n*(n+1))
            acc += term
            if abs(term)<=1e-17*aq: break
        return -np.log(x)+acc
    # sum_n (nv-1)! n!/(nv-1+n)! h_n(y) u^(n+1), u = 1/(z-m), recurring on
    # w = y u so that no power of u appears alone
    u = -1./x
    w0 = y0*u; w1 = y1*u; w2 = y2*u
    H0 = 1.+0j; H1 = 1.+0j; H2 = 1.+0j # h_n(w_0..w_i), i = 0..2
    acc = 1.+0j
    for n in range(1,_nmax+1):
        H0 = w0*H0
        H1 = H0+w1*H1
        if nv==3:
            H2 = H1+w2*H2
            term = H2*(2./((n+1)*(n+2)))
        else: term = H1/(n+1)
        acc += term
        if abs(term)<=1e-17*abs(acc): # stop after two negligible terms
            small += 1
            if small==2: break
        else: small = 0
    return u*acc


@jit(nopython=True,parallel=True,cache=True)
def _evaluate_sources_jit(ev,cv,vkind,ep,tp,yp,ca,omegas,delta,nchunk):
    """Sum of all sources at every frequency. The parallel loop runs over
    (frequency, chunk of sources), each task writing its own partial sum,
    so the result does not depend on the number of threads."""
    nw = omegas.shape[0]
    nc = cv.shape[1]
    nvs = ev.shape[0]
    nps = ep.shape[0]
    nv = 3 if vkind==2 else 2 # vertices per simplex
    part = np.zeros((nw,nchunk,nc),dtype=np.complex128)
    for task in prange(nw*nchunk): # parallel loop over frequency and chunk
        iw = task//nchunk
        ic = task%nchunk
        z = omegas[iw] + 1j*delta
        acc = np.zeros(nc,dtype=np.complex128)
        i0 = (ic*nvs)//nchunk
        i1 = ((ic+1)*nvs)//nchunk
        for i in range(i0,i1): # vertex sources
            x = ev[i]-z
            if vkind==2: f = -(x*np.log(x)-x) # psi_2
            else: f = -np.log(x) # psi_1
            for c in range(nc): acc[c] += cv[i,c]*f
        i0 = (ic*nps)//nchunk
        i1 = ((ic+1)*nps)//nchunk
        for i in range(i0,i1): # point sources
            f = _point_value(tp[i],ep[i]-z,yp[i,0],yp[i,1],yp[i,2],nv)
            for c in range(nc): acc[c] += ca[i,c]*f
        for c in range(nc): part[iw,ic,c] = acc[c]
    out = np.zeros((nw,nc),dtype=np.complex128)
    for iw in range(nw):
        for ic in range(nchunk):
            for c in range(nc): out[iw,c] += part[iw,ic,c]
    return out


def _merge(keys,*cs):
    """Merge the sources whose keys (one row of exactly equal numbers per
    source) coincide, adding their coefficients, and drop those whose
    coefficients all vanish. Returns the merged keys and coefficients.
    A 1D array of keys is one key per source, and comes back 1D."""
    if keys.ndim==1: # one key per source, as shgsimplex passes them
        out = _merge(keys[:,None],*cs)
        return (out[0][:,0],)+out[1:]
    if keys.shape[0]==0: return (keys,)+cs
    if keys.shape[1]==1: order = np.argsort(keys[:,0],kind="stable")
    else: order = np.lexsort(keys.T[::-1])
    keys = keys[order]
    cs = [c[order] for c in cs]
    new = np.concatenate([[True],np.any(keys[1:]!=keys[:-1],axis=1)])
    idx = np.nonzero(new)[0]
    keys = keys[idx]
    cs = [np.add.reduceat(c,idx,axis=0) for c in cs]
    nz = np.zeros(len(keys),dtype=np.bool_)
    for c in cs: nz |= np.any(c!=0.,axis=1)
    return (keys[nz],)+tuple(c[nz] for c in cs)


def simplex_sources(simplices,dE,W,trans,diag,delta,alpha=_alpha):
    """All the sources of the simplex sum, see _sources_jit: vertex sources
    (energies ev, coefficients cv) and point sources (energies ep, types
    tp, shapes yp, coefficients ca)"""
    npts,npair = dE.shape
    nc = W.shape[2]
    A = np.zeros((2,npts,npair,nc),dtype=np.complex128) # forward, reverse
    e0 = np.zeros(0); t0 = np.zeros(0,dtype=np.int64)
    y0 = np.zeros((0,3)); c0 = np.zeros((0,nc),dtype=np.complex128)
    args = (simplices,dE,W,trans,diag,float(alpha*delta),A[0],A[1])
    npoint = _sources_jit(False,*args,e0,t0,y0,c0) # count the point sources
    ep = np.zeros(npoint); tp = np.zeros(npoint,dtype=np.int64)
    yp = np.zeros((npoint,3)); ca = np.zeros((npoint,nc),dtype=np.complex128)
    _sources_jit(True,*args,ep,tp,yp,ca)
    A = A.reshape(-1,nc)
    nz = np.any(A!=0.,axis=1) # e.g. the reverse term of a diagonal pair
    ev = np.concatenate([dE.ravel(),-dE.ravel()])[nz]
    ev,cv = _merge(ev[:,None],A[nz])
    del A
    # the point sources merge when their type, energy and shape coincide,
    # as in the copies of a spinful model
    keys,ca = _merge(np.concatenate([tp[:,None].astype(np.float64),ep[:,None],
            yp],axis=1),ca)
    tp = keys[:,0].astype(np.int64)
    return (ev[:,0].copy(),cv,keys[:,1].copy(),tp,
            np.ascontiguousarray(keys[:,2:]),ca)


def _sigma_sources(simplices,dE,W,trans,diag,omegas,delta):
    """The output of _sigma_simplices, from the sources"""
    import numba
    ev,cv,ep,tp,yp,ca = simplex_sources(simplices,dE,W,trans,diag,delta)
    vkind = 2 if simplices.shape[1]==3 else 1
    nw = len(omegas)
    nsrc = len(ev)+len(ep)
    # enough tasks to keep every thread busy when there are few frequencies
    nchunk = max(1,min(nsrc//1024,-(-4*numba.get_num_threads()//max(nw,1))))
    return _evaluate_sources_jit(ev,np.ascontiguousarray(cv),vkind,ep,tp,yp,
            np.ascontiguousarray(ca),omegas,delta,nchunk)


_kernels = {
    "sources": _sigma_sources, # sources built once, then summed per frequency
    "simplices": _sigma_simplices, # every simplex at every frequency
}


def _sigma_from_fields(dim,nk,pairs,dE,W,comps,trans,energies,delta,cellvol,
        kernel="sources"):
    """Assemble the 3x3 tensor from the stored pair fields on the mesh.
    kernel="simplices" is the direct sum over simplices, kept as the
    reference that the tests compare the default against."""
    if kernel not in _kernels:
        raise ValueError("unknown simplex kernel "+repr(kernel)+
                ", accepted values are "+", ".join(_kernels))
    sim,frac = simplices(dim,nk)
    diag = np.array([p[0]==p[1] for p in pairs],dtype=np.bool_)
    parallel.set_num_threads() # set the number of threads for numba
    s = _kernels[kernel](sim,np.ascontiguousarray(dE),
            np.ascontiguousarray(W),trans,diag,
            np.asarray(energies,dtype=np.float64),delta)
    sigma = np.zeros((len(energies),3,3),dtype=np.complex128)
    for c,(a,b) in enumerate(comps): sigma[:,a,b] = s[:,c]
    return 1j*sigma*frac/cellvol


# ------------------------------------------------------------ public paths

def sigma_triangle(h,energies,nk,T,delta,intraband,interband,degeneracy_tol,
        window=None):
    """Optical conductivity tensor with the simplex integration on the
    uniform nk mesh, see conductivity.optical_conductivity"""
    check_broadening(delta)
    ks = kubo._kmesh(h,nk)
    es,vs,cellvol,scale = kubo.mesh_bands(h,ks,window,degeneracy_tol)
    pairs,dE,W,comps,trans = pair_fields(es,vs,T,degeneracy_tol*scale,
            intraband,interband)
    pairs,dE,W = _prune(pairs,dE,W)
    return _sigma_from_fields(h.dimensionality,nk,pairs,dE,W,comps,trans,
            energies,delta,cellvol)


def sigma_qtci(h,energies,nk,T,delta,intraband,interband,degeneracy_tol,
        tolerance=1e-6,window=None):
    """Optical conductivity tensor with the simplex integration on a
    2^R x 2^R mesh (nk rounded up to a power of two), where the smooth
    fields dE and W are not computed on every point but reconstructed from
    a quantics tensor cross interpolation, see
    conductivity.optical_conductivity"""
    check_broadening(delta)
    if h.dimensionality!=2:
        raise NotImplementedError("integration='qtci' of the optical "
                "conductivity is only implemented in 2D, got dimensionality "
                +str(h.dimensionality)+"; use integration='triangle', in 1D "
                "a fine mesh is already cheap")
    from ..qtcitk.meshfields import qtci_fields
    R = max(1,int(np.ceil(np.log2(nk)))) # bits per direction
    n = 2**R
    nc0 = 2**min(R,5)
    ks0 = kubo._kmesh(h,nc0)
    if window is None:
        setup = kubo._setup(h)
        setup = (setup,kubo._hopping_arrays(setup[0]))
        cellvol,scale = setup[0][5],setup[0][6]
        def bands(ks):
            return kubo._bands_and_velocities(h,ks,setup=setup)[:2]
        es0,vs0 = bands(ks0)
    else: # the band range is fixed on the coarse scan, then only checked
        from . import kubowindow
        ws = kubowindow.WindowSetup(h)
        es0,vs0,cellvol,scale,rng = kubowindow.bands_and_velocities(h,ks0,
                window,tol_rel=degeneracy_tol,ws=ws)
        def bands(ks):
            return kubowindow.bands_and_velocities(h,ks,window,
                    tol_rel=degeneracy_tol,ws=ws,fixed=rng)[:2]
    tol = degeneracy_tol*scale
    # coarse scan: fixes the pairs and components that matter (qtci_fields
    # repeats it on the same points to seed the pivots)
    axes = _active_axes(vs0)
    pairs,dE0,W0,comps,trans = pair_fields(es0,vs0,T,tol,intraband,interband,
            axes=axes)
    pairs,dE0,W0 = _prune(pairs,dE0,W0)
    # the independent real fields: dE of the off-diagonal pairs, and the
    # real and imaginary parts of W for a <= b (W for (b,a) is its conjugate)
    fields = []
    for p in range(len(pairs)):
        if pairs[p,0]!=pairs[p,1]: fields.append(("dE",p,None))
        for c,(a,b) in enumerate(comps):
            if a>b: continue
            fields.append(("re",p,c))
            if a!=b: fields.append(("im",p,c))
    def evaluate(ks): # every field at the k-points ks
        es,vs = bands(ks)
        _,dEk,Wk,_,_ = pair_fields(es,vs,T,tol,intraband,interband,
                axes=axes,pairs=pairs)
        out = np.zeros((len(ks),len(fields)))
        for f,(kind,p,c) in enumerate(fields):
            if kind=="dE": out[:,f] = dEk[:,p]
            elif kind=="re": out[:,f] = Wk[:,p,c].real
            else: out[:,f] = Wk[:,p,c].imag
        return out
    vals,ncalls,ranks = qtci_fields(evaluate,R,len(fields),
            tolerance=tolerance,ncoarse=nc0)
    npts = n*n
    dE = np.zeros((npts,len(pairs)),dtype=np.float64)
    W = np.zeros((npts,len(pairs),len(comps)),dtype=np.complex128)
    for f,(kind,p,c) in enumerate(fields):
        if kind=="dE": dE[:,p] = vals[:,f]
        elif kind=="re": W[:,p,c] += vals[:,f]
        else: W[:,p,c] += 1j*vals[:,f]
    for c,(a,b) in enumerate(comps): # the (b,a) components, by conjugation
        if a>b: W[:,:,c] = np.conjugate(W[:,:,trans[c]])
    return _sigma_from_fields(2,n,pairs,dE,W,comps,trans,energies,delta,cellvol)
