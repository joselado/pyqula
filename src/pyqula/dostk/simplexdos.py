# Linear-simplex (segment, triangle, tetrahedron) integration of the density
# of states and of the bulk Green's function, and its quantics tensor cross
# interpolation (QTCI) variant in 2D.
#
# The band energies E_n(k) and the weights w_n(k) (one, an operator
# expectation value, or the projector u_n u_n^dagger) are interpolated
# linearly inside each simplex of a uniform mesh, and the resolvent
# 1/(z - E) is integrated over the simplex exactly. The mesh only has to
# resolve the smooth fields E and w, not the broadening, so the same mesh
# gives every energy at any delta, delta = 0 included for the DOS. This is
# the analytic tetrahedron method for complex energies of Ph. Lambin and
# J.-P. Vigneron, PRB 29, 3430 (1984), the same construction as
# conductivitytk/kubosimplex.py for the optical conductivity.
#
# The simplex average of phi(E(k)) with E linear in k is, by the
# Hermite-Genocchi formula, d! times the d-th divided difference over the
# vertex energies of the d-th antiderivative psi_d of phi. For
# phi(E) = 1/(z - E) and x = E - z,
#   psi_0 = -1/x,  psi_k = -x^(k-1)/(k-1)! (ln x - H_(k-1)),
# with H_m the harmonic number. Im x = -delta <= 0, so the principal
# logarithm is continuous along the real axis; at delta = 0 the imaginary
# part of x is a negative zero and ln x takes the lower side of the cut,
# which is the retarded limit, and -Im/pi gives the exact density of a
# linear function on a simplex (the tent in 2D).
#
# The divided differences are evaluated with a Newton table in which every
# cluster of nodes whose spread is small compared to its distance to z is
# replaced by its Taylor series, f[x_0..x_m] = sum_n f^(m+n)(c) h_n(x-c) /
# (m+n)!, with h_n the complete homogeneous symmetric polynomial. The
# criterion is relative to |c - z| and not to delta: the cancellation of a
# plain divided difference grows like (|x|/spread)^m, which matters for a
# nearly flat simplex far from resonance even when the spread is much
# larger than delta. kubosimplex.py switches on the spread over delta,
# which is enough at the delta of an optical spectrum but not below 1e-4.
#
# The weights are averaged over the vertices of each simplex and over
# degenerate multiplets at each vertex, as in kubosimplex.py: a single
# w_n inside a degenerate multiplet depends on the basis eigh picks, the
# multiplet average does not.
import numpy as np
from numba import jit, prange

from .. import parallel


_rtol = 0.05 # a cluster is Taylor expanded when spread < _rtol*|c - z|
_nterms = 14 # largest order of that Taylor series
_fact = np.array([float(np.prod(np.arange(1,n+1))) for n in range(40)])
_harm = np.array([float(np.sum(1./np.arange(1,n+1))) for n in range(8)])


# ------------------------------------------------------------ the kernel

@jit(nopython=True,cache=True)
def _psi(k,x,harm,fact):
    """k-th antiderivative of 1/(z-E) in E, at x = E - z"""
    if x.real==0. and x.imag==0.: # only at delta = 0, on a vertex
        x = complex(0.,-1e-300) # the retarded side, psi_k(0) = 0 for k>1
    if k==0: return -1./x
    p = 1.+0j
    for i in range(k-1): p *= x
    return -p*(np.log(x)-harm[k-1])/fact[k-1]


@jit(nopython=True,cache=True)
def _deriv(K,j,x,harm,fact):
    """j-th derivative of psi_K at x, divided by j!"""
    if j<=K: return _psi(K-j,x,harm,fact)/fact[j]
    n = j-K # psi_0^(n) = n!/(z-E)^(n+1) = n! (-1/x)^(n+1)
    u = -1./x
    p = u
    for i in range(n): p *= u
    return p*fact[n]/fact[j]


@jit(nopython=True,cache=True)
def _taylor_dd(K,t,lo,hi,s,harm,fact,h):
    """divided difference of psi_K over the nodes t[lo..hi] + i s, as the
    Taylor series around their mean. h is work space of length >= the
    number of nodes, which holds h_n(y_1..y_i) for the current n"""
    m = hi-lo
    c = 0.
    for i in range(lo,hi+1): c += t[i]
    c /= (m+1)
    xc = complex(c,s)
    if t[hi]==t[lo]: # confluent nodes: the series stops at its first term
        return _deriv(K,m,xc,harm,fact)
    p = m+1
    for i in range(p): h[i] = 1. # h_0 = 1
    out = 0j
    u = -1./xc
    pw = 1.+0j # u^ip, raised as needed to u^(j-K+1) for j = m+n > K
    ip = 0
    small = 0
    hn = 1.
    for n in range(_nterms+1):
        if n>0: # h_n(y_1..y_i) = h_n(y_1..y_(i-1)) + y_i h_(n-1)(y_1..y_i)
            prev = 0.
            for i in range(p):
                prev = prev + (t[lo+i]-c)*h[i]
                h[i] = prev
            hn = h[p-1]
            if n==1: continue # h_1 vanishes for centered nodes
        j = m+n
        if j<=K: term = _deriv(K,j,xc,harm,fact)*hn
        else:
            while ip<j-K+1:
                pw *= u
                ip += 1
            term = pw*(fact[j-K]/fact[j]*hn)
        out += term
        if n>2: # stop after two negligible terms in a row
            if abs(term.real)+abs(term.imag)<1e-17*(abs(out.real)
                    +abs(out.imag)): small += 1
            else: small = 0
            if small==2: break
    return out


@jit(nopython=True,cache=True)
def _taylor_simplex(t,p,c,s,fact):
    """psi_(p-1)[t_0..t_(p-1)] as the series around the mean c, the common
    case of a simplex far from resonance, written out: with u = 1/(z-c),
    (p-1)! psi_(p-1)[...] = sum_n (p-1)! n!/(p-1+n)! h_n(t-c) u^(n+1)"""
    if t[p-1]==t[0]: return 1./(complex(-c,-s))/fact[p-1]
    u = 1./complex(-c,-s)
    y0 = t[0]-c; y1 = t[1]-c
    y2 = t[2]-c if p>2 else 0.
    y3 = t[3]-c if p>3 else 0.
    h0 = 1.; h1 = 1.; h2 = 1.; h3 = 1. # h_n(y_0..y_i), i = 0..3
    pw = u
    out = u/fact[p-1]
    small = 0
    for n in range(1,_nterms+1):
        h0 = y0*h0
        h1 = h0 + y1*h1
        h2 = h1 + y2*h2
        h3 = h2 + y3*h3
        pw = pw*u
        if n==1: continue # h_1 vanishes for centered nodes
        hn = h1 if p==2 else (h2 if p==3 else h3)
        term = pw*(fact[n]/fact[p-1+n]*hn)
        out += term
        if n>2: # stop after two negligible terms in a row
            if abs(term.real)+abs(term.imag)<1e-17*(abs(out.real)
                    +abs(out.imag)): small += 1
            else: small = 0
            if small==2: break
    return out


@jit(nopython=True,cache=True)
def _close(sp,c,s):
    """whether a cluster of spread sp around c is Taylor expanded"""
    return sp*sp<=_rtol*_rtol*(c*c+s*s)


@jit(nopython=True,cache=True)
def _dd(K,t,p,s,harm,fact,h,d):
    """divided difference psi_K[t_0+is,...,t_(p-1)+is] for sorted real
    parts t, robust against confluent and nearly confluent nodes"""
    if p==1: return _psi(K,complex(t[0],s),harm,fact)
    c = 0.
    for i in range(p): c += t[i]
    if _close(t[p-1]-t[0],c/p,s): # the whole simplex at once
        if K==p-1: return _taylor_simplex(t,p,c/p,s,fact)
        return _taylor_dd(K,t,0,p-1,s,harm,fact,h)
    for i in range(p): d[i] = _psi(K,complex(t[i],s),harm,fact)
    for m in range(1,p): # Newton table, in place
        for i in range(p-m):
            sp = t[i+m]-t[i]
            cc = 0.
            for j in range(i,i+m+1): cc += t[j]
            if _close(sp,cc/(m+1),s):
                d[i] = _taylor_dd(K,t,i,i+m,s,harm,fact,h)
            else: d[i] = (d[i+1]-d[i])/sp
    return d[0]


@jit(nopython=True,cache=True)
def _sort_small(t,p):
    """insertion sort of the first p entries"""
    for i in range(1,p):
        v = t[i]
        j = i-1
        while j>=0 and t[j]>v:
            t[j+1] = t[j]
            j -= 1
        t[j+1] = v


@jit(nopython=True,cache=True)
def _simplex_average(es,sim,it,n,E,s,t,harm,fact,h,d):
    """average of 1/(E+i delta - E_n(k)) over simplex it, s = -delta"""
    p = sim.shape[1]
    for v in range(p): t[v] = es[sim[it,v],n]-E
    _sort_small(t,p)
    return fact[p-1]*_dd(p-1,t,p,s,harm,fact,h,d)


@jit(nopython=True,parallel=True,cache=True)
def _simplex_sum(sim,es,W,energies,delta,harm,fact):
    """sum over simplices and bands of the vertex-averaged weight times the
    exact simplex average of the resolvent, at every energy. Not
    normalized: the caller multiplies by the volume fraction of a simplex"""
    ns = sim.shape[0]
    p = sim.shape[1]
    nb = es.shape[1]
    nc = W.shape[2]
    ne = energies.shape[0]
    s = -delta
    out = np.zeros((ne,nc),dtype=np.complex128)
    for ie in prange(ne): # parallel loop over energies
        E = energies[ie]
        t = np.zeros(p+1)
        h = np.zeros(_nterms+1)
        d = np.zeros(p+1,dtype=np.complex128)
        acc = np.zeros(nc,dtype=np.complex128)
        for it in range(ns): # loop over simplices
            for n in range(nb): # loop over bands
                a = _simplex_average(es,sim,it,n,E,s,t,harm,fact,h,d)
                for c in range(nc):
                    wbar = 0j
                    for v in range(p): wbar += W[sim[it,v],n,c]
                    acc[c] += wbar*a/p
        out[ie] = acc
    return out


@jit(nopython=True,parallel=True,cache=True)
def _simplex_dos0(sim,es,W,energies,nchunk,harm,fact):
    """the delta = 0 limit of -Im/pi of _simplex_sum, for sorted energies.
    A simplex only contributes at the energies between its lowest and its
    highest vertex, which are found by bisection, so the cost is set by the
    number of simplices and not by their product with the energies"""
    ns = sim.shape[0]
    p = sim.shape[1]
    nb = es.shape[1]
    nc = W.shape[2]
    ne = energies.shape[0]
    out = np.zeros((nchunk,ne,nc),dtype=np.float64)
    size = (ns+nchunk-1)//nchunk
    for ic in prange(nchunk): # parallel loop over chunks of simplices
        t = np.zeros(p+1)
        h = np.zeros(_nterms+1)
        d = np.zeros(p+1,dtype=np.complex128)
        for it in range(ic*size,min(ns,(ic+1)*size)):
            for n in range(nb):
                emin = es[sim[it,0],n]
                emax = emin
                for v in range(1,p):
                    e = es[sim[it,v],n]
                    if e<emin: emin = e
                    if e>emax: emax = e
                i0 = np.searchsorted(energies,emin)
                i1 = np.searchsorted(energies,emax,side="right")
                # a flat simplex (to rounding) is a delta function in
                # energy, which no grid point resolves
                if i1<=i0 or emax-emin<=1e-12*(1.+abs(emax)): continue
                wbar = np.zeros(nc,dtype=np.float64)
                for c in range(nc):
                    for v in range(p): wbar[c] += W[sim[it,v],n,c].real
                for ie in range(i0,i1):
                    a = _simplex_average(es,sim,it,n,energies[ie],-0.,t,
                            harm,fact,h,d)
                    for c in range(nc): out[ic,ie,c] += -a.imag*wbar[c]/p
    return out.sum(axis=0)/np.pi


# ------------------------------------------------------------ meshes

def simplices(dim,nk):
    """Simplices of the periodic uniform mesh of klist.kmesh (row-major
    vertex indices) and the volume fraction of the Brillouin zone that each
    one covers: segments in 1D, two triangles per square in 2D, and in 3D
    the six tetrahedra of each cube that share its main diagonal"""
    if dim in (1,2):
        from ..conductivitytk.kubosimplex import simplices as sim12
        return sim12(dim,nk)
    if dim!=3:
        raise ValueError("the simplex integration needs a dimensionality of "
                "1, 2 or 3, got "+str(dim))
    i,j,l = np.meshgrid(np.arange(nk),np.arange(nk),np.arange(nk),
            indexing="ij")
    i = i.ravel(); j = j.ravel(); l = l.ravel()
    def idx(a,b,c): return ((i+a)%nk)*nk*nk + ((j+b)%nk)*nk + (l+c)%nk
    out = []
    import itertools
    for perm in itertools.permutations(range(3)): # path 000 -> 111
        v = [0,0,0]
        verts = [idx(*v)]
        for ax in perm:
            v[ax] = 1
            verts.append(idx(*v))
        out.append(np.stack(verts,axis=1))
    return np.concatenate(out).astype(np.int64),1./(6*nk**3)


def uniform_mesh(dim,nk):
    """the uniform mesh of klist.kmesh(dim,nk=nk), in the same row-major
    order, built with numpy instead of a Python loop over the points"""
    k1 = np.arange(nk)/nk
    grids = np.meshgrid(*([k1]*dim),indexing="ij")
    ks = np.zeros((nk**dim,3))
    for i in range(dim): ks[:,i] = grids[i].ravel()
    return ks


# ------------------------------------------------------------ fields

def _hk_batch(hkgen,ks):
    """Bloch Hamiltonians at every k of ks, as one contraction over the
    Bloch phases when the generator carries its hoppings"""
    data = getattr(hkgen,"bloch_data",None)
    if data is None:
        from ..htk.eigenvectors import hk_matrix_batch
        return hk_matrix_batch(hkgen,ks)
    ms,ds = data
    dim = ds.shape[1]
    ph = np.exp(2j*np.pi*(np.asarray(ks,dtype=np.float64)[:,:dim]@ds.T))
    return np.einsum("kh,hij->kij",ph,ms,optimize=True)


@jit(nopython=True,parallel=True,cache=True)
def _multiplet_average(es,W,tol):
    """replace W[k,n,:] by its average over the degenerate multiplet of n"""
    nk = es.shape[0]
    nb = es.shape[1]
    nc = W.shape[2]
    for ik in prange(nk):
        i = 0
        while i<nb:
            j = i+1
            while j<nb and es[ik,j]-es[ik,j-1]<tol: j += 1
            if j-i>1:
                for c in range(nc):
                    acc = 0j
                    for m in range(i,j): acc += W[ik,m,c]
                    for m in range(i,j): W[ik,m,c] = acc/(j-i)
            i = j


def _degeneracy_tol(es):
    return 1e-8*max(1.,float(np.max(np.abs(es))))


class _Fields():
    """Diagonalizes H(k) on batches of k-points and returns the band
    energies es[k,n] and the weights W[k,n,c] the simplex sum needs:
    kind="dos" gives W = 1, kind="operator" the multiplet-averaged
    <u_n|O|u_n>, kind="green" the multiplet-averaged projector
    u_n,i u_n,j^* flattened over (i,j). ndiag counts the diagonalizations"""
    def __init__(self,h,kind="dos",operator=None,batch=2**15):
        if h.is_sparse: h = h.get_dense()
        self.hkgen = h.get_hk_gen()
        self.kind = kind
        self.batch = batch
        self.ndiag = 0
        self.op = None
        if kind=="operator":
            from .. import algebra
            self.op = np.asarray(algebra.todense(
                h.get_operator(operator).get_matrix()),dtype=np.complex128)
            if self.op.shape!=h.intra.shape:
                raise ValueError("the operator of the simplex DOS must be a "
                        "matrix of the size of the Hamiltonian, "
                        +str(h.intra.shape)+", got "+str(self.op.shape))
    def __call__(self,ks):
        from ..htk.eigenvectors import parallel_diagonalization, peigvalsh
        ks = np.asarray(ks,dtype=np.float64)
        parallel.set_num_threads()
        ess = []; Ws = []
        for i0 in range(0,len(ks),self.batch):
            hks = _hk_batch(self.hkgen,ks[i0:i0+self.batch])
            if self.kind=="dos":
                es = peigvalsh(hks)
                W = np.ones((es.shape[0],es.shape[1],1),dtype=np.complex128)
            else:
                es,U = parallel_diagonalization(hks)
                if self.kind=="operator":
                    w = np.einsum("kin,ij,kjn->kn",U.conj(),self.op,U,
                            optimize=True).real
                    W = w[:,:,None].astype(np.complex128)
                else: # projector P_n[i,j] = U[i,n] U[j,n]^*
                    n = U.shape[1]
                    W = np.einsum("kin,kjn->knij",U,U.conj(),optimize=True)
                    W = np.ascontiguousarray(W.reshape(len(es),n,n*n))
                _multiplet_average(es,W,_degeneracy_tol(es))
            ess.append(es); Ws.append(W)
        self.ndiag += len(ks)
        return np.concatenate(ess),np.concatenate(Ws)


# ------------------------------------------------------------ integration

def simplex_sum(dim,nk,es,W,energies,delta):
    """sum_n int dk wbar_n(k)/(E + i delta - E_n(k)) over the Brillouin
    zone (normalized to one), from the fields on the uniform nk^dim mesh,
    at every energy: a (len(energies),nc) complex array"""
    sim,frac = simplices(dim,nk)
    parallel.set_num_threads()
    out = _simplex_sum(sim,np.ascontiguousarray(es),np.ascontiguousarray(W),
            np.asarray(energies,dtype=np.float64),float(delta),_harm,_fact)
    return out*frac


def simplex_dos_from_fields(dim,nk,es,W,energies,delta):
    """the (projected) density of states from the fields on the mesh"""
    energies = np.asarray(energies,dtype=np.float64)
    if delta>0.:
        return -simplex_sum(dim,nk,es,W,energies,delta)[:,0].imag/np.pi
    sim,frac = simplices(dim,nk)
    order = np.argsort(energies)
    parallel.set_num_threads()
    import numba
    nchunk = 8*numba.get_num_threads()
    out = _simplex_dos0(sim,np.ascontiguousarray(es),np.ascontiguousarray(W),
            energies[order],nchunk,_harm,_fact)[:,0]*frac
    res = np.zeros(len(energies))
    res[order] = out
    return res


def check_broadening(delta):
    if delta is None or not delta>=0.:
        raise ValueError("the simplex DOS needs a broadening delta >= 0 "
                "(delta=0 is the exact limit), got delta="+str(delta))


def _mesh_kind(operator):
    return "dos" if operator is None else "operator"


def dos_triangle(h,energies,nk=100,delta=0.,operator=None):
    """Density of states with the linear simplex integration on the uniform
    nk^d mesh (segments, triangles or tetrahedra), returned as
    (energies, dos); dos_triangle_counted also returns the number of
    diagonalizations.

    The result is the exact DOS of the linearly interpolated bands,
    broadened by a Lorentzian of width delta (delta=0 is the exact
    limit). Three limits of that interpolation are worth knowing:
    - a flat simplex (all vertices at one energy) is a delta function in
      energy. At delta=0 it is dropped, since no grid point resolves it;
      at small finite delta it is a spike of weight 1/(2 nk^2) per
      triangle. They appear when a straight constant-energy line runs
      along the mesh, as the van Hove lines of graphene do for an even
      nk, and around a K point at the centroid of a triangle (3 does not
      divide nk). An odd nk divisible by 3 avoids both in graphene.
    - the bands are ordered by energy at each k, so where two bands cross
      the interpolation has a kink inside the triangles, which gives an
      error of order one in a window of width of order v/nk around the
      crossing energy; it shrinks with the window as nk grows.
    - in 1D the pointwise error is first order in 1/nk, in 2D and 3D it
      is close to second order away from van Hove singularities."""
    return dos_triangle_counted(h,energies,nk=nk,delta=delta,
            operator=operator)[0:2]


def dos_triangle_counted(h,energies,nk=100,delta=0.,operator=None):
    check_broadening(delta)
    dim = h.dimensionality
    if dim not in (1,2,3):
        raise ValueError("the simplex DOS needs a periodic Hamiltonian of "
                "dimensionality 1, 2 or 3, got "+str(dim))
    fields = _Fields(h,kind=_mesh_kind(operator),operator=operator)
    es,W = fields(uniform_mesh(dim,nk))
    energies = np.asarray(energies,dtype=np.float64)
    ds = simplex_dos_from_fields(dim,nk,es,W,energies,delta)
    return energies,ds,fields.ndiag


# ------------------------------------------------------------ qtci

def qtci_fields(evaluate,R,nfield,tolerance=1e-6,maxbonddim=400,ncoarse=32):
    """Reconstruct the real fields evaluate(ks) -> (len(ks),nfield) on the
    2^R x 2^R mesh (row-major, as klist.kmesh) from one quantics tensor
    cross interpolation per field. Every field shares one cache of
    evaluations, so a k-point is diagonalized once whatever the number of
    fields that ask for it. Returns the fields (n*n,nfield), the number of
    distinct k-points evaluated and the largest bond dimension per field.
    The structure is the one of kubosimplex.sigma_qtci."""
    from ..qutecipytk import crossinterpolate2
    from ..qutecipytk.tensortrain.core import tensortrain
    from ..qutecipytk.tensortrain.batcheval import BatchEvaluator
    import itertools
    n = 2**R
    cache = {}
    def ensure(ij):
        new = [q for q in dict.fromkeys(ij) if q not in cache]
        if not new: return
        ks = np.zeros((len(new),3))
        ks[:,:2] = np.array(new,dtype=np.float64)/n
        vals = evaluate(ks)
        for q,v in zip(new,vals): cache[q] = v
    weights = 2**np.arange(R-1,-1,-1)
    def decode(keys): # interleaved bits (x,y), most significant first
        q = np.array(keys,dtype=np.int64).reshape(-1,R,2)
        return list(zip((q[:,:,0]@weights).tolist(),
            (q[:,:,1]@weights).tolist()))
    def encode(i,j):
        out = []
        for s in range(R): out += [(i>>(R-1-s))&1,(j>>(R-1-s))&1]
        return out
    localdims = [2]*(2*R)
    class Field(BatchEvaluator):
        def __init__(self,f): self.f = f
        def evalmany(self,keys):
            ij = decode(keys)
            ensure(ij)
            return np.array([cache[q][self.f] for q in ij])
        def __call__(self,x): return self.evalmany([tuple(x)])[0]
        def batchevaluate(self,L,Rr,ncent):
            nl = len(L[0]) if len(L) else 0
            cd = localdims[nl:nl+ncent]
            combos = list(itertools.product(*[range(d) for d in cd]))
            keys = [tuple(l)+c+tuple(r) for l in L for c in combos for r in Rr]
            return self.evalmany(keys).reshape(len(L),*cd,len(Rr))
    # coarse scan on points of the fine grid, which seeds the pivots
    nc0 = min(n,ncoarse)
    stride = n//nc0
    ij0 = [(i*stride,j*stride) for i in range(nc0) for j in range(nc0)]
    ensure(ij0)
    f0 = np.array([cache[q] for q in ij0])
    out = np.zeros((n*n,nfield))
    xaxes = list(range(0,2*R,2)); yaxes = list(range(1,2*R,2))
    ranks = []
    for f in range(nfield):
        if np.max(np.abs(f0[:,f]))==0.:
            ranks.append(0); continue
        i0 = int(np.argmax(np.abs(f0[:,f])))
        piv = [encode(*ij0[i0])]
        tci,rk,errors = crossinterpolate2(np.float64,Field(f),localdims,
                initialpivots=piv,tolerance=tolerance,maxbonddim=maxbonddim,
                pivotsearch="full")
        ranks.append(int(max(rk)) if len(rk) else 0)
        full = tensortrain(tci).fulltensor()
        out[:,f] = np.transpose(full,xaxes+yaxes).reshape(n*n)
    return out,len(cache),ranks


def dos_qtci_counted(h,energies,nk=128,delta=0.,operator=None,
        tolerance=1e-6):
    """Density of states with the triangle integration on a 2^R x 2^R mesh
    (nk rounded up to a power of two), with the band energies (and the
    weights, with an operator) reconstructed from a QTCI of each field
    instead of computed at every point"""
    check_broadening(delta)
    if h.dimensionality!=2:
        raise NotImplementedError("the qtci integration of the DOS is only "
                "implemented in 2D, got dimensionality "
                +str(h.dimensionality)+"; use integration='triangle'")
    R = max(1,int(np.ceil(np.log2(nk))))
    fields = _Fields(h,kind=_mesh_kind(operator),operator=operator)
    nb = h.intra.shape[0]
    def evaluate(ks):
        es,W = fields(ks)
        if operator is None: return es
        return np.concatenate([es,W[:,:,0].real],axis=1)
    nf = nb if operator is None else 2*nb
    vals,ncalls,ranks = qtci_fields(evaluate,R,nf,tolerance=tolerance)
    es = np.ascontiguousarray(vals[:,:nb])
    if operator is None: W = np.ones((len(es),nb,1),dtype=np.complex128)
    else: W = vals[:,nb:,None].astype(np.complex128)
    energies = np.asarray(energies,dtype=np.float64)
    ds = simplex_dos_from_fields(2,2**R,es,W,energies,delta)
    return energies,ds,fields.ndiag,ranks


def dos_qtci(h,energies,nk=128,delta=0.,operator=None,tolerance=1e-6):
    return dos_qtci_counted(h,energies,nk=nk,delta=delta,operator=operator,
            tolerance=tolerance)[0:2]


# ------------------------------------------------------------ Green's function

def green_triangle(h,energies,nk=100,delta=1e-2):
    """Bulk Green's function of the unit cell, (1/N) sum_k (z - H_k)^-1 with
    z = E + i delta, from the simplex integration of the multiplet-averaged
    band projectors on the uniform nk^d mesh, at every energy: a
    (len(energies),n,n) array. One diagonalization of the mesh serves every
    energy"""
    if not delta>0.:
        raise ValueError("the simplex Green's function needs a positive "
                "broadening delta, got delta="+str(delta)+"; at delta=0 "
                "its real part has logarithmic divergences")
    dim = h.dimensionality
    if dim not in (1,2,3):
        raise ValueError("the simplex Green's function needs a periodic "
                "Hamiltonian of dimensionality 1, 2 or 3, got "+str(dim))
    fields = _Fields(h,kind="green")
    es,W = fields(uniform_mesh(dim,nk))
    n = h.intra.shape[0]
    g = simplex_sum(dim,nk,es,W,np.atleast_1d(energies),delta)
    return g.reshape(len(g),n,n)
