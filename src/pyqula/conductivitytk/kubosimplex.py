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

@jit(nopython=True,cache=True)
def _sort3(a,b,c):
    if a>b: a,b = b,a
    if b>c: b,c = c,b
    if a>b: a,b = b,a
    return a,b,c


@jit(nopython=True,cache=True)
def _psi1(e,z): return -np.log(e-z)


@jit(nopython=True,cache=True)
def _psi2(e,z):
    x = e-z
    return -(x*np.log(x)-x)


@jit(nopython=True,cache=True)
def _segment_average(e0,e1,z,tolc):
    """average of 1/(z-E) over a segment with linear E from e0 to e1"""
    h = abs(e1-e0)
    if h<tolc: # confluent: Taylor series around the midpoint
        u = 1./(z-0.5*(e0+e1))
        return u + u*u*u*h*h/12.
    return (_psi1(e1,z)-_psi1(e0,z))/(e1-e0)


@jit(nopython=True,cache=True)
def _dd1_psi2(a,b,z,tolc):
    """first divided difference of psi_2, with the confluent limit"""
    h = b-a
    if h<tolc:
        mid = 0.5*(a+b)
        u = 1./(z-mid)
        return _psi1(mid,z) + u*u*h*h/24.
    return (_psi2(b,z)-_psi2(a,z))/h


@jit(nopython=True,cache=True)
def _triangle_average(e0,e1,e2,z,tolc):
    """average of 1/(z-E) over a triangle with linear E taking the values
    e0, e1, e2 at the vertices: 2 psi_2[e0,e1,e2]"""
    e0,e1,e2 = _sort3(e0,e1,e2)
    s = e2-e0
    if s<tolc: # confluent: second order Taylor series around the mean
        mean = (e0+e1+e2)/3.
        var = (e0*e0+e1*e1+e2*e2+e0*e1+e0*e2+e1*e2)/6. - mean*mean
        u = 1./(z-mean)
        return u + u*u*u*var
    return 2.*(_dd1_psi2(e1,e2,z,tolc)-_dd1_psi2(e0,e1,z,tolc))/s


@jit(nopython=True,parallel=True,cache=True)
def _sigma_simplex_jit(simplices,dE,W,trans,diag,omegas,delta):
    """Sum over simplices and stored pairs of the vertex-averaged weight
    times the exact simplex average of the resolvent, for every frequency
    and every Cartesian pair. As in kubo._sigma_jit the parallel loop runs
    over frequencies, so each thread owns one output row. Not normalized:
    the caller multiplies by i, the simplex volume fraction and 1/V_cell."""
    ns = simplices.shape[0]
    nv = simplices.shape[1] # vertices per simplex, 2 or 3
    npair = dE.shape[1]
    nc = W.shape[2]
    nw = omegas.shape[0]
    tolc = 1e-2*delta # below this energy spread the Taylor series is exact
                      # to (spread/delta)^3, above it the divided difference
                      # loses at most eps*delta/tolc^2 to cancellation
    out = np.zeros((nw,nc),dtype=np.complex128)
    for iw in prange(nw): # parallel loop over frequencies
        z = omegas[iw] + 1j*delta
        acc = np.zeros(nc,dtype=np.complex128)
        wbar = np.zeros(nc,dtype=np.complex128)
        for it in range(ns): # loop over simplices
            for p in range(npair): # loop over stored pairs
                for c in range(nc): wbar[c] = 0.
                for v in range(nv):
                    for c in range(nc): wbar[c] += W[simplices[it,v],p,c]
                e0 = dE[simplices[it,0],p]
                e1 = dE[simplices[it,1],p]
                if nv==2:
                    sf = _segment_average(e0,e1,z,tolc)
                    if not diag[p]: sr = _segment_average(-e0,-e1,z,tolc)
                else:
                    e2 = dE[simplices[it,2],p]
                    sf = _triangle_average(e0,e1,e2,z,tolc)
                    if not diag[p]: sr = _triangle_average(-e0,-e1,-e2,z,tolc)
                for c in range(nc):
                    acc[c] += wbar[c]*sf
                    if not diag[p]: acc[c] += wbar[trans[c]]*sr
        out[iw] = acc/nv
    return out


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
# The branches of _sigma_simplex_jit depend on the vertex energies and on
# delta, never on omega. So the whole sum over simplices and pairs is a sum
# of fixed coefficients times a few functions of x = e - z:
#   psi_2(x) (psi_1(x) in 1D) at the vertex energies (exact branches),
#   psi_1(x) and 1/x^2 at the middle of a confluent edge,
#   1/x and 1/x^3 at the mean of a confluent simplex.
# In other words the weighted joint density of states, a sum of linear
# B-splines, is stored by its knots. The coefficients are built once, the
# vertex ones indexed by k-point, so the simplices sharing a vertex share
# one source, and the sum is then evaluated per frequency over the sources.
# This is the same sum as _sigma_simplex_jit, term by term, with the
# additions in another order.

@jit(nopython=True,cache=True)
def _sort3i(a,b,c,ia,ib,ic):
    """_sort3, carrying along the vertex indices"""
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
def _add_point(write,n,ep,tp,ca,cb,e,t,wbar,fa,fb):
    """append a point source of type t at energy e, with coefficients
    fa*wbar and fb*wbar of its two kernels; the counting pass only counts"""
    if write:
        ep[n] = e
        tp[n] = t
        for c in range(wbar.shape[0]):
            ca[n,c] = fa*wbar[c]
            cb[n,c] = fb*wbar[c]
    return n+1


@jit(nopython=True,cache=True)
def _simplex_sources(write,e0,e1,e2,k0,k1,k2,nv,wbar,tolc,
        Av,p,n,ep,tp,ca,cb):
    """sources of one simplex and one pair, with the branches of
    _segment_average and _triangle_average. Av[k,p,c] collects the vertex
    coefficients, the point sources go to (ep,tp,ca,cb) from index n on.
    Type 1: psi_1 and 1/x^2, type 2: 1/x and 1/x^3. Returns the new n."""
    if nv==2: # segment
        h = abs(e1-e0)
        if h<tolc: # confluent, u + u^3 h^2/12 with u = -1/x
            n = _add_point(write,n,ep,tp,ca,cb,0.5*(e0+e1),2,wbar,
                    -1.,-h*h/12.)
        elif write: # (psi_1(e1)-psi_1(e0))/(e1-e0)
            f = 1./(e1-e0)
            for c in range(wbar.shape[0]):
                Av[k1,p,c] += f*wbar[c]
                Av[k0,p,c] -= f*wbar[c]
        return n
    e0,e1,e2,k0,k1,k2 = _sort3i(e0,e1,e2,k0,k1,k2)
    s = e2-e0
    if s<tolc: # confluent, u + u^3 var
        mean = (e0+e1+e2)/3.
        var = (e0*e0+e1*e1+e2*e2+e0*e1+e0*e2+e1*e2)/6. - mean*mean
        return _add_point(write,n,ep,tp,ca,cb,mean,2,wbar,-1.,-var)
    # 2 (dd1(e1,e2) - dd1(e0,e1))/s
    for side in range(2):
        if side==0:
            a = e1; b = e2; ka = k1; kb = k2; sgn = 2./s
        else:
            a = e0; b = e1; ka = k0; kb = k1; sgn = -2./s
        h = b-a
        if h<tolc: # confluent edge, psi_1(mid) + u^2 h^2/24
            n = _add_point(write,n,ep,tp,ca,cb,0.5*(a+b),1,wbar,
                    sgn,sgn*h*h/24.)
        elif write: # (psi_2(b)-psi_2(a))/h
            f = sgn/h
            for c in range(wbar.shape[0]):
                Av[kb,p,c] += f*wbar[c]
                Av[ka,p,c] -= f*wbar[c]
    return n


@jit(nopython=True,cache=True)
def _sources_jit(write,simplices,dE,W,trans,diag,delta,
        Af,Ar,ep,tp,ca,cb):
    """Build (write=True) or count (write=False) the sources of the whole
    simplex sum. Af and Ar take the vertex coefficients of the forward
    (energy dE) and reverse (energy -dE) terms. Returns the number of
    point sources."""
    ns = simplices.shape[0]
    nv = simplices.shape[1]
    npair = dE.shape[1]
    nc = W.shape[2]
    tolc = 1e-2*delta # as in _sigma_simplex_jit
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
                    Af,p,n,ep,tp,ca,cb)
            if not diag[p]:
                for c in range(nc): wrev[c] = wbar[trans[c]]
                n = _simplex_sources(write,-e0,-e1,-e2,k0,k1,k2,nv,wrev,
                        tolc,Ar,p,n,ep,tp,ca,cb)
    return n


@jit(nopython=True,parallel=True,cache=True)
def _evaluate_sources_jit(ev,cv,vkind,ep,tp,ca,cb,omegas,delta,nchunk):
    """Sum of all sources at every frequency. The parallel loop runs over
    (frequency, chunk of sources), each task writing its own partial sum,
    so the result does not depend on the number of threads."""
    nw = omegas.shape[0]
    nc = cv.shape[1]
    nvs = ev.shape[0]
    nps = ep.shape[0]
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
            x = ep[i]-z
            if tp[i]==1:
                fa = -np.log(x)
                fb = 1./(x*x)
            else:
                fa = 1./x
                fb = fa*fa*fa
            for c in range(nc): acc[c] += ca[i,c]*fa + cb[i,c]*fb
        for c in range(nc): part[iw,ic,c] = acc[c]
    out = np.zeros((nw,nc),dtype=np.complex128)
    for iw in range(nw):
        for ic in range(nchunk):
            for c in range(nc): out[iw,c] += part[iw,ic,c]
    return out


def _merge(e,*cs):
    """Merge the sources with exactly the same energy, adding their
    coefficients, and drop those whose coefficients all vanish"""
    if len(e)==0: return (e,)+cs
    order = np.argsort(e,kind="stable")
    e = e[order]
    cs = [c[order] for c in cs]
    idx = np.nonzero(np.concatenate([[True],e[1:]!=e[:-1]]))[0]
    e = e[idx]
    cs = [np.add.reduceat(c,idx,axis=0) for c in cs]
    nz = np.zeros(len(e),dtype=np.bool_)
    for c in cs: nz |= np.any(c!=0.,axis=1)
    return (e[nz],)+tuple(c[nz] for c in cs)


def simplex_sources(simplices,dE,W,trans,diag,delta):
    """All the sources of the simplex sum, see _sources_jit: vertex sources
    (energies ev, coefficients cv) and point sources (energies ep, types tp,
    coefficients ca and cb of the two kernels of each type)"""
    npts,npair = dE.shape
    nc = W.shape[2]
    A = np.zeros((2,npts,npair,nc),dtype=np.complex128) # forward, reverse
    e0 = np.zeros(0); t0 = np.zeros(0,dtype=np.int64)
    c0 = np.zeros((0,nc),dtype=np.complex128)
    npoint = _sources_jit(False,simplices,dE,W,trans,diag,delta,
            A[0],A[1],e0,t0,c0,c0) # first pass: count the point sources
    ep = np.zeros(npoint); tp = np.zeros(npoint,dtype=np.int64)
    ca = np.zeros((npoint,nc),dtype=np.complex128)
    cb = np.zeros((npoint,nc),dtype=np.complex128)
    _sources_jit(True,simplices,dE,W,trans,diag,delta,A[0],A[1],ep,tp,ca,cb)
    A = A.reshape(-1,nc)
    nz = np.any(A!=0.,axis=1) # e.g. the reverse term of a diagonal pair
    ev = np.concatenate([dE.ravel(),-dE.ravel()])[nz]
    ev,cv = _merge(ev,A[nz])
    del A
    pts = [] # merge each type of point source separately
    for t in (1,2):
        m = tp==t
        e,a,b = _merge(ep[m],ca[m],cb[m])
        pts.append((e,np.full(len(e),t,dtype=np.int64),a,b))
    ep,tp,ca,cb = [np.concatenate([q[i] for q in pts]) for i in range(4)]
    return ev,cv,ep,tp,ca,cb


def _sigma_sources(simplices,dE,W,trans,diag,omegas,delta):
    """Same output as _sigma_simplex_jit, from the sources"""
    import numba
    ev,cv,ep,tp,ca,cb = simplex_sources(simplices,dE,W,trans,diag,delta)
    vkind = 2 if simplices.shape[1]==3 else 1
    nw = len(omegas)
    nsrc = len(ev)+len(ep)
    # enough tasks to keep every thread busy when there are few frequencies
    nchunk = max(1,min(nsrc//1024,-(-4*numba.get_num_threads()//max(nw,1))))
    return _evaluate_sources_jit(ev,np.ascontiguousarray(cv),vkind,ep,tp,
            np.ascontiguousarray(ca),np.ascontiguousarray(cb),omegas,
            delta,nchunk)


_kernels = {
    "sources": _sigma_sources, # sources built once, then summed per frequency
    "simplices": _sigma_simplex_jit, # every simplex at every frequency
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

def sigma_triangle(h,energies,nk,T,delta,intraband,interband,degeneracy_tol):
    """Optical conductivity tensor with the simplex integration on the
    uniform nk mesh, see conductivity.optical_conductivity"""
    check_broadening(delta)
    ks = kubo._kmesh(h,nk)
    es,vs,cellvol,scale = kubo._bands_and_velocities(h,ks)
    pairs,dE,W,comps,trans = pair_fields(es,vs,T,degeneracy_tol*scale,
            intraband,interband)
    pairs,dE,W = _prune(pairs,dE,W)
    return _sigma_from_fields(h.dimensionality,nk,pairs,dE,W,comps,trans,
            energies,delta,cellvol)


def sigma_qtci(h,energies,nk,T,delta,intraband,interband,degeneracy_tol,
        tolerance=1e-6):
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
    from ..qutecipytk import crossinterpolate2
    from ..qutecipytk.tensortrain.core import tensortrain
    R = max(1,int(np.ceil(np.log2(nk)))) # bits per direction
    n = 2**R
    setup = kubo._setup(h)
    setup = (setup,kubo._hopping_arrays(setup[0]))
    cellvol,scale = setup[0][5],setup[0][6]
    tol = degeneracy_tol*scale
    # coarse scan: fixes the pairs and components that matter, and seeds
    # the pivots of every field at its largest value
    nc0 = 2**min(R,5)
    ks0 = kubo._kmesh(h,nc0)
    es0,vs0,_,_ = kubo._bands_and_velocities(h,ks0,setup=setup)
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
    def value(kind,p,c,dE,W):
        if kind=="dE": return dE[...,p]
        if kind=="re": return W[...,p,c].real
        return W[...,p,c].imag
    cache = {} # grid index (i,j) -> (dE[p], W[p,c]) of the stored pairs
    def ensure(ij):
        new = [q for q in dict.fromkeys(ij) if q not in cache]
        if not new: return
        ks = np.zeros((len(new),3))
        ks[:,:2] = np.array(new,dtype=np.float64)/n
        es,vs,_,_ = kubo._bands_and_velocities(h,ks,setup=setup)
        _,dE,W,_,_ = pair_fields(es,vs,T,tol,intraband,interband,
                axes=axes,pairs=pairs)
        for q,a,b in zip(new,dE,W): cache[q] = (a,b)
    weights = 2**np.arange(R-1,-1,-1)
    def decode(keys): # interleaved bits (x,y) per scale, most significant first
        q = np.array(keys,dtype=np.int64).reshape(-1,R,2)
        return list(zip((q[:,:,0]@weights).tolist(),(q[:,:,1]@weights).tolist()))
    def encode(i,j):
        out = []
        for s in range(R):
            out += [(i>>(R-1-s))&1,(j>>(R-1-s))&1]
        return out
    localdims = [2]*(2*R)
    from ..qutecipytk.tensortrain.batcheval import BatchEvaluator
    import itertools
    class Field(BatchEvaluator):
        def __init__(self,kind,p,c): self.key = (kind,p,c)
        def evalmany(self,keys):
            ij = decode(keys)
            ensure(ij)
            kind,p,c = self.key
            return np.array([value(kind,p,c,*cache[q]) for q in ij])
        def __call__(self,x): return self.evalmany([tuple(x)])[0]
        def batchevaluate(self,L,Rr,ncent):
            nl = len(L[0]) if len(L) else 0
            cd = localdims[nl:nl+ncent]
            combos = list(itertools.product(*[range(d) for d in cd]))
            keys = [tuple(l)+c+tuple(r) for l in L for c in combos for r in Rr]
            return self.evalmany(keys).reshape(len(L),*cd,len(Rr))
    npts = n*n
    dE = np.zeros((npts,len(pairs)),dtype=np.float64)
    W = np.zeros((npts,len(pairs),len(comps)),dtype=np.complex128)
    xaxes = list(range(0,2*R,2)); yaxes = list(range(1,2*R,2))
    stride = n//nc0 # coarse point (i,j) is the fine point (i,j)*stride
    for (kind,p,c) in fields:
        f0 = value(kind,p,c,dE0,W0)
        if np.max(np.abs(f0))==0.: continue # identically zero, e.g. Im W_xx
        i0 = int(np.argmax(np.abs(f0)))
        piv = [encode((i0//nc0)*stride,(i0%nc0)*stride)]
        tci,ranks,errors = crossinterpolate2(np.float64,Field(kind,p,c),
                localdims,initialpivots=piv,tolerance=tolerance,
                maxbonddim=400,pivotsearch="full")
        full = tensortrain(tci).fulltensor()
        full = np.transpose(full,xaxes+yaxes).reshape(npts) # row-major (i,j)
        if kind=="dE": dE[:,p] = full
        elif kind=="re": W[:,p,c] += full
        else: W[:,p,c] += 1j*full
    for c,(a,b) in enumerate(comps): # the (b,a) components, by conjugation
        if a>b: W[:,:,c] = np.conjugate(W[:,:,trans[c]])
    return _sigma_from_fields(2,n,pairs,dE,W,comps,trans,energies,delta,cellvol)
