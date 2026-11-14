# Linear-simplex (segment, triangle, tetrahedron) integration of the density
# of states and of the bulk Green's function, and its quantics tensor cross
# interpolation (QTCI) variant, in 2D and 3D for the DOS and in 2D for the
# LDOS.
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
# larger than delta. The direct kernel of kubosimplex.py calls _dd as it
# is; its source kernel needs a rule independent of the energy and uses
# spread < delta with a longer series instead, and calls _dd per frequency
# for the simplices whose vertex form would cancel too much.
#
# The weights are averaged over the vertices of each simplex and over
# degenerate multiplets at each vertex, as in kubosimplex.py: a single
# w_n inside a degenerate multiplet depends on the basis eigh picks, the
# multiplet average does not. This holds because every weight is linear
# in the projector u_n u_n^dagger, so its sum over a multiplet is a trace:
# 1, <u_n|O|u_n>, the projector itself, and the site weight
# Re<u_n|P_i O|u_n> (|u_n|^2 on the site without an operator).
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
def _vertex_resolvent(sim,esT,energies,delta,nrep,harm,fact):
    """A[n,ie,r,k], the sum over the simplices that have the mesh point k
    as a vertex of the exact simplex average of 1/(E_ie + i delta - E_n),
    from the band energies esT[n,k] (one row per band). The simplices are
    cut in nrep slices and r is the slice, so the sum over r is the sum
    over all the simplices.

    The sum over simplices of the vertex-averaged weight times that average
    is sum_(k,n) A[n,ie,k] W[k,n,c] / p, so the weights enter through one
    matrix product, once per mesh point and not once per vertex of every
    simplex at every energy. Each parallel task owns one band, one energy
    and one slice, and adds into its own row of mesh points, so no two
    tasks write the same entry"""
    ns = sim.shape[0]
    p = sim.shape[1]
    nb = esT.shape[0]
    nkp = esT.shape[1]
    ne = energies.shape[0]
    s = -delta
    A = np.zeros((nb,ne,nrep,nkp),dtype=np.complex128)
    for task in prange(nb*ne*nrep): # bands, energies and slices
        n = task//(ne*nrep)
        ie = (task//nrep)%ne
        r = task%nrep
        E = energies[ie]
        e = esT[n]
        acc = A[n,ie,r]
        t = np.zeros(p+1)
        h = np.zeros(_nterms+1)
        d = np.zeros(p+1,dtype=np.complex128)
        for it in range((r*ns)//nrep,((r+1)*ns)//nrep): # simplices
            for v in range(p): t[v] = e[sim[it,v]]-E
            _sort_small(t,p)
            a = fact[p-1]*_dd(p-1,t,p,s,harm,fact,h,d) # simplex average
            for v in range(p): acc[sim[it,v]] += a
    return A


_block = 2**22 # complex numbers in one block of A, 64 MB


def _vertex_resolvent_blocks(sim,esT,energies,delta):
    """the vertex resolvent A[n,ie,k] of _vertex_resolvent, one block of
    energies at a time: yields (slice of the energies, A of that slice).
    A block holds near _block numbers, and at least one energy. When a
    block has fewer bands times energies than there are threads, the
    simplices are cut in slices so that every thread has a task; the
    kernel then holds one row of mesh points per thread"""
    import numba
    nb,nkp = esT.shape
    ne = len(energies)
    size = max(1,min(ne,_block//(nkp*nb))) # energies per block
    for i in range(0,ne,size):
        b = slice(i,min(ne,i+size))
        nrep = max(1,numba.get_num_threads()//(nb*(b.stop-b.start)))
        A = _vertex_resolvent(sim,esT,energies[b],float(delta),nrep,
                _harm,_fact)
        yield b,(A.sum(axis=2) if nrep>1 else A[:,:,0])


@jit(nopython=True,parallel=True,cache=True)
def _simplex_dos0(sim,es,W,energies,nchunk,harm,fact):
    """the delta = 0 limit of -Im/pi of simplex_sum, for sorted energies.
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
    return _tetrahedra(nk),1./(6*nk**3)


def _tetrahedra(nk):
    """the vertex indices of the six tetrahedra of every cube, ordered by
    permutation first: tetrahedron perm*nk^3 + cube goes from the corner
    000 of the cube to 111 along the axes in the order perm. The array is
    filled in place, so the peak memory is the array and a few index
    vectors, not three copies of it"""
    import itertools
    nc = nk**3
    cube = np.arange(nc,dtype=np.int64) # row-major index of the corner 000
    strides = (nk*nk,nk,1)
    # part of the row-major index along each axis, at the corner (0) and
    # one step further (1), periodic
    part = []
    for ax,x in enumerate((cube//(nk*nk),(cube//nk)%nk,cube%nk)):
        part.append((x*strides[ax],((x+1)%nk)*strides[ax]))
    out = np.empty((6*nc,4),dtype=np.int64)
    for ip,perm in enumerate(itertools.permutations(range(3))):
        rows = out[ip*nc:(ip+1)*nc]
        rows[:,0] = cube
        v = [0,0,0]
        for iv,ax in enumerate(perm): # path 000 -> 111
            v[ax] = 1
            rows[:,iv+1] = part[0][v[0]] + part[1][v[1]] + part[2][v[2]]
    return out


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
    <u_n|O|u_n>, and kind="site" the multiplet-averaged density of u_n on
    each site, its spin and Nambu components summed, or with an operator
    the local matrix element Re<u_n|P_i O|u_n>, which sums over the sites
    to <u_n|O|u_n>. kind="green" returns the eigenvectors U[k,i,n] in
    place of W, n times smaller than the band projectors, and
    green_triangle averages over the multiplets on the resolvent instead.
    ndiag counts the diagonalizations"""
    def __init__(self,h,kind="dos",operator=None,batch=2**15):
        if h.is_sparse: h = h.get_dense()
        self.hkgen = h.get_hk_gen()
        self.kind = kind
        self.ndiag = 0
        self.op = None
        n = h.intra.shape[0]
        # keep the Bloch matrices and the eigenvectors of one batch, n^2
        # per k-point each, near 2^22 complex numbers (64 MB)
        self.batch = max(1,min(batch,2**22//(n*n)))
        if kind=="site": # orbital -> site, the resummation of the LDOS maps
            from ..increase_hilbert import full2profile
            iden = np.identity(n)
            self.sites = np.array([full2profile(h,iden[i]) for i in range(n)],
                    dtype=np.float64) # (orbitals, sites)
            batch = max(1,min(batch,2**22//(n*self.sites.shape[1]+1)))
            self.batch = batch # keep the batch of densities small
        if operator is not None and kind in ("operator","site"):
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
        ess = Ws = None # filled batch by batch, never concatenated
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
                elif self.kind=="site": # density on each site
                    if self.op is None: # |u_n(a)|^2, (k,orbital,band)
                        rho = (U.real**2 + U.imag**2)
                    else: # the local element Re[conj(u_n(a)) (O u_n)(a)]
                        rho = (U.conj()*np.matmul(self.op,U)).real
                    W = np.einsum("kin,ic->knc",rho,self.sites,
                            optimize=True).astype(np.complex128)
                else: W = U # the eigenvectors, (k,orbital,band)
                if self.kind!="green":
                    _multiplet_average(es,W,_degeneracy_tol(es))
            if ess is None:
                ess = np.empty((len(ks),)+es.shape[1:],dtype=es.dtype)
                Ws = np.empty((len(ks),)+W.shape[1:],dtype=W.dtype)
            ess[i0:i0+len(es)] = es; Ws[i0:i0+len(es)] = W
        self.ndiag += len(ks)
        return ess,Ws


# ------------------------------------------------------------ integration

def simplex_sum(dim,nk,es,W,energies,delta):
    """sum_n int dk wbar_n(k)/(E + i delta - E_n(k)) over the Brillouin
    zone (normalized to one), from the fields on the uniform nk^dim mesh,
    at every energy: a (len(energies),nc) complex array"""
    sim,frac = simplices(dim,nk)
    esT = np.ascontiguousarray(np.transpose(es)) # one row per band
    energies = np.asarray(energies,dtype=np.float64)
    out = np.zeros((len(energies),W.shape[2]),dtype=np.complex128)
    parallel.set_num_threads()
    for b,A in _vertex_resolvent_blocks(sim,esT,energies,delta):
        for n in range(len(esT)): # A[n] is (energies, mesh points)
            out[b] += A[n] @ W[:,n,:]
    return out*(frac/sim.shape[1])


def simplex_dos_from_fields(dim,nk,es,W,energies,delta):
    """the (projected) density of states from the fields on the mesh"""
    return simplex_channels_from_fields(dim,nk,es,W,energies,delta)[:,0]


def simplex_channels_from_fields(dim,nk,es,W,energies,delta):
    """the density of states of every channel c of the real weights
    W[k,n,c] (a site of the LDOS, or the one channel of the DOS), from the
    fields on the mesh: a (len(energies),nc) real array"""
    energies = np.asarray(energies,dtype=np.float64)
    if delta>0.:
        return -simplex_sum(dim,nk,es,W,energies,delta).imag/np.pi
    sim,frac = simplices(dim,nk)
    order = np.argsort(energies)
    parallel.set_num_threads()
    import numba
    nchunk = 8*numba.get_num_threads()
    out = _simplex_dos0(sim,np.ascontiguousarray(es),np.ascontiguousarray(W),
            energies[order],nchunk,_harm,_fact)*frac
    res = np.zeros((len(energies),W.shape[2]))
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

# The DOS runs one interpolation per field (qtci_fields), the LDOS one
# interpolation of all the fields at once, with the field index as an extra
# tensor site (qtci_stacked_fields). The stacked form needs fewer
# diagonalizations when the fields are few, but its rank grows with their
# number: it was twice as fast as one per field for the 72 fields of the
# LDOS of an 8-band supercell, and about half as fast for the 32 band
# energies of a 32-band supercell, see future_development/qtci_dos_green.md
from ..qtcitk.meshfields import qtci_fields
from ..qtcitk.meshfields import qtci_stacked_fields,stacked_maxbonddim


def dos_qtci_counted(h,energies,nk=128,delta=0.,operator=None,
        tolerance=1e-6,ordering=None):
    """Density of states with the triangle integration on a 2^R x 2^R mesh
    in 2D, the tetrahedron integration on a 2^R x 2^R x 2^R mesh in 3D (nk
    rounded up to a power of two), with the band energies (and the
    weights, with an operator) reconstructed from a QTCI of each field
    instead of computed at every point. ordering is the bit ordering of
    the quantics mesh, see qtcitk.meshfields"""
    check_broadening(delta)
    dim = h.dimensionality
    if dim not in (2,3):
        raise NotImplementedError("the qtci integration of the DOS is only "
                "implemented in 2D and 3D, got dimensionality "
                +str(dim)+"; use integration='triangle'")
    R = max(1,int(np.ceil(np.log2(nk))))
    fields = _Fields(h,kind=_mesh_kind(operator),operator=operator)
    nb = h.intra.shape[0]
    def evaluate(ks):
        es,W = fields(ks)
        if operator is None: return es
        return np.concatenate([es,W[:,:,0].real],axis=1)
    nf = nb if operator is None else 2*nb
    vals,ncalls,ranks = qtci_fields(evaluate,R,nf,tolerance=tolerance,
            dim=dim,ordering=ordering)
    es = np.ascontiguousarray(vals[:,:nb])
    if operator is None: W = np.ones((len(es),nb,1),dtype=np.complex128)
    else: W = vals[:,nb:,None].astype(np.complex128)
    energies = np.asarray(energies,dtype=np.float64)
    ds = simplex_dos_from_fields(dim,2**R,es,W,energies,delta)
    return energies,ds,fields.ndiag,ranks


def dos_qtci(h,energies,nk=128,delta=0.,operator=None,tolerance=1e-6):
    return dos_qtci_counted(h,energies,nk=nk,delta=delta,operator=operator,
            tolerance=tolerance)[0:2]


# ------------------------------------------------------------ local DOS

def ldos_triangle_counted(h,energies,nk=100,delta=0.,operator=None):
    """Local density of states of every site of the unit cell, its spin and
    Nambu components summed, from the simplex integration on the uniform
    nk^d mesh, at every energy: (energies, ldos, ndiag, mesh) with ldos a
    (len(energies),nsites) array and mesh the (dim, nk, band energies) that
    dos_from_band_mesh takes. The weight of band n on site i is the
    multiplet-averaged sum of |u_n,a|^2 over the orbitals a of site i, or
    with an operator of the local matrix element Re[conj(u_n,a) (O u_n)_a]
    (the convention of every mode of ldos.get_ldos). One diagonalization
    of the mesh gives every site and every energy, and the sum over sites
    is the triangle DOS to rounding, or with an operator the triangle DOS
    weighted by <u_n|O|u_n>"""
    check_broadening(delta)
    dim = h.dimensionality
    if dim not in (1,2,3):
        raise ValueError("the simplex LDOS needs a periodic Hamiltonian of "
                "dimensionality 1, 2 or 3, got "+str(dim))
    fields = _Fields(h,kind="site",operator=operator)
    es,W = fields(uniform_mesh(dim,nk))
    energies = np.asarray(energies,dtype=np.float64)
    ds = simplex_channels_from_fields(dim,nk,es,W,energies,delta)
    return energies,ds,fields.ndiag,(dim,nk,es)


def ldos_qtci_counted(h,energies,nk=128,delta=0.,operator=None,
        tolerance=1e-6):
    """The local DOS of ldos_triangle_counted on a 2^R x 2^R mesh (nk
    rounded up to a power of two), with the band energies and the weight of
    every band on every site reconstructed from one QTCI of all the fields
    at once, in 2D. The tolerance is relative to the largest value of any
    field. The band energies enter that interpolation measured from their
    mean over the Brillouin zone, Re Tr(H_0)/n, and the mean is added back
    afterwards, so a constant shift of every onsite energy does not loosen
    the tolerance of the weights"""
    check_broadening(delta)
    if h.dimensionality!=2:
        raise NotImplementedError("the qtci integration of the LDOS is only "
                "implemented in 2D, not in "+str(h.dimensionality)+"D; use "
                "integration='triangle'")
    R = max(1,int(np.ceil(np.log2(nk))))
    fields = _Fields(h,kind="site",operator=operator)
    nb = h.intra.shape[0]
    ns = fields.sites.shape[1]
    from .. import algebra
    # the mean of the band energies, exact: the hoppings to other cells
    # average to zero over the Brillouin zone
    e0 = float(np.trace(algebra.todense(h.intra)).real)/nb
    def evaluate(ks):
        es,W = fields(ks)
        return np.concatenate([es-e0,W.real.reshape(len(es),nb*ns)],axis=1)
    vals,ncalls,ranks = qtci_stacked_fields(evaluate,R,nb*(1+ns),
            tolerance=tolerance,maxbonddim=stacked_maxbonddim)
    es = np.ascontiguousarray(vals[:,:nb]+e0)
    W = np.ascontiguousarray(vals[:,nb:].reshape(len(es),nb,ns)
            ).astype(np.complex128)
    energies = np.asarray(energies,dtype=np.float64)
    ds = simplex_channels_from_fields(2,2**R,es,W,energies,delta)
    return energies,ds,fields.ndiag,(2,2**R,es)


def dos_from_band_mesh(mesh,energies,delta):
    """the DOS from the band energies a simplex LDOS call already has, the
    mesh tuple it returns, with no new diagonalization"""
    dim,nk,es = mesh
    W = np.ones((es.shape[0],es.shape[1],1),dtype=np.complex128)
    return simplex_dos_from_fields(dim,nk,es,W,energies,delta)


# ------------------------------------------------------------ Green's function

def green_triangle(h,energies,nk=100,delta=1e-2):
    """Bulk Green's function of the unit cell, (1/N) sum_k (z - H_k)^-1 with
    z = E + i delta, from the simplex integration of the multiplet-averaged
    band projectors on the uniform nk^d mesh, at every energy: a
    (len(energies),n,n) array. One diagonalization of the mesh serves every
    energy.

    The mesh keeps the eigenvectors, n^2 numbers per k-point, and not the
    projectors, n^3. The sum over a multiplet M of A_n P_n, with P_n the
    projector averaged over M, is the sum of P_n times A_n averaged over
    M, so the average is taken on the vertex resolvent A and the
    projectors are built for one chunk of the mesh at a time"""
    if not delta>0.:
        raise ValueError("the simplex Green's function needs a positive "
                "broadening delta, got delta="+str(delta)+"; at delta=0 "
                "its real part has logarithmic divergences")
    dim = h.dimensionality
    if dim not in (1,2,3):
        raise ValueError("the simplex Green's function needs a periodic "
                "Hamiltonian of dimensionality 1, 2 or 3, got "+str(dim))
    fields = _Fields(h,kind="green")
    es,U = fields(uniform_mesh(dim,nk)) # U[k,i,n]
    nkp,n,nb = U.shape
    sim,frac = simplices(dim,nk)
    energies = np.atleast_1d(np.asarray(energies,dtype=np.float64))
    tol = _degeneracy_tol(es)
    esT = np.ascontiguousarray(np.transpose(es)) # one row per band
    g = np.zeros((len(energies),n*n),dtype=np.complex128)
    chunk = max(1,_block//(nb*n*n)) # k-points per chunk of projectors
    parallel.set_num_threads()
    for b,A in _vertex_resolvent_blocks(sim,esT,energies,delta):
        A = np.ascontiguousarray(np.transpose(A,(2,0,1))) # A[k,n,ie]
        _multiplet_average(es,A,tol) # in place of averaging the projectors
        for k0 in range(0,nkp,chunk):
            Uc = U[k0:k0+chunk]
            P = np.einsum("kin,kjn->knij",Uc,Uc.conj(),optimize=True)
            g[b] += A[k0:k0+chunk].reshape(-1,A.shape[2]).T @ \
                    P.reshape(-1,n*n)
    g *= frac/sim.shape[1]
    return g.reshape(len(energies),n,n)
