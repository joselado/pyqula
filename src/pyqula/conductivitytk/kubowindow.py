# Band window of the Kubo-Greenwood optical conductivity: only the bands
# close to the Fermi level are diagonalized, the velocity is rotated only on
# them, and only their pairs are stored. The physics of the truncation, what
# it keeps exact and what it biases, is documented in
# conductivity.optical_conductivity; this module only holds the machinery.
#
# The window is a fixed range of band indices, the same at every k-point.
# Bands are labelled by their position in the sorted spectrum of H(k), as in
# the full calculation, so a pair index p means the same pair of bands at
# every vertex of a triangle, and in every call of the qtci cache. A window
# whose bands changed from one k-point to the next would make the triangle
# and qtci paths interpolate between different pairs.
#
# The two forms of the window are both turned into such a range:
#   ewindow = W (or (emin,emax)): every band that comes inside (emin,emax)
#     at some k-point of the mesh is kept at every k-point.
#   nv, nc: the nv bands below and the nc bands above the Fermi level are
#     kept. In a metal, where the number of occupied bands N(k) changes
#     across the mesh, the range is [min N - nv, max N + nc), so every band
#     that crosses the Fermi level is kept as well.
# The range is first estimated from the eigenvalues on a few k-points, and
# then checked on the whole mesh with a few probe bands below and above
# it; if a probe band enters the window, or the edge cuts a degenerate
# multiplet where the window asks for the edge band, the range grows. The
# probe bands are kept when the growth fits inside them, and the mesh is
# done again only when it does not.
import warnings
import numpy as np
import scipy.linalg as dlg
from scipy.sparse import csr_matrix, csc_matrix, hstack

from .. import algebra
from .. import limits


_full_eigh_dimension = 200 # below this, the batched numba eigh of all the
                           # bands is cheaper than the scipy subset solver
_kbatch = 64 # k-points diagonalized together on the batched path
_nprescan = 16 # k-points of the first estimate of the range
_nprobe = 4 # bands computed beyond each edge of the range, to check it


class BandWindow():
    """The window as the caller gives it: an energy interval around the
    Fermi level, or a number of bands below and above it"""
    def __init__(self,ewindow=None,nv=None,nc=None):
        if ewindow is not None and (nv is not None or nc is not None):
            raise ValueError("give the band window either as an energy "
                "interval ewindow or as numbers of bands nv, nc, not both "
                "(got ewindow="+str(ewindow)+", nv="+str(nv)+", nc="
                +str(nc)+")")
        self.energy = ewindow is not None
        if self.energy:
            if np.ndim(ewindow)==0:
                emin,emax = -abs(float(ewindow)),abs(float(ewindow))
            else:
                if len(ewindow)!=2: raise ValueError("ewindow must be a "
                    "number W, for (-W,W), or a pair (emin,emax), got "
                    +str(ewindow))
                emin,emax = float(ewindow[0]),float(ewindow[1])
            if not emin<0.<emax: raise ValueError("the energy window must "
                "contain the Fermi level, zero, got (emin,emax)=("
                +str(emin)+","+str(emax)+")")
            self.emin,self.emax = emin,emax
        else:
            nv = 0 if nv is None else int(nv)
            nc = 0 if nc is None else int(nc)
            if nv<0 or nc<0 or nv+nc==0: raise ValueError("nv and nc "
                "count bands below and above the Fermi level, they must "
                "be non-negative and not both zero, got nv="+str(nv)
                +", nc="+str(nc))
            self.nv,self.nc = nv,nc


def get_window(ewindow=None,nv=None,nc=None):
    """BandWindow for the keywords, or None if no window is asked for"""
    if ewindow is None and nv is None and nc is None: return None
    return BandWindow(ewindow=ewindow,nv=nv,nc=nc)


# ------------------------------------------------------------ setup

class WindowSetup():
    """Everything that does not depend on k, with the hoppings kept sparse:
    the dense copies of kubo._setup would cost n^2 per hopping"""
    def __init__(self,h):
        from . import kubo
        hm = h.get_multicell() # a copy
        self.n = hm.intra.shape[0]
        self.orders,self.jac,self.cellvol = kubo._cell_factors(h)
        self.rr = kubo._orbital_positions(h,self.n) # (n,3)
        self.intra = csr_matrix(hm.intra,dtype=np.complex128)
        tms,dirs = [],[]
        for t in hm.hopping:
            m = csr_matrix(t.m,dtype=np.complex128)
            m.eliminate_zeros()
            if m.nnz==0: continue
            tms.append(m)
            dirs.append(np.array(t.dir,dtype=np.float64))
        self.dirs = np.array(dirs).reshape(-1,3)
        self.tstack = hstack(tms,format="csr") if tms else None # (n,nhop*n)
        self.tms = tms
        absmax = lambda m: np.max(np.abs(m.data)) if m.nnz else 0.
        self.scale = max(absmax(self.intra),
                max((absmax(m) for m in tms),default=0.),1e-12)
        # a shift-invert solver above the dense limit, as elsewhere in pyqula
        self.sparse = self.n>limits.densedimension

    def hk(self,k):
        """Bloch Hamiltonian at k, sparse, with the phase exp(2 pi i k.R)
        of kubo._bloch_phases"""
        ph = np.exp(2j*np.pi*(self.dirs@np.array(k,dtype=np.float64)))
        m = self.intra.copy()
        for p,t in zip(ph,self.tms): m = m + p*t
        return m

    def velocities(self,k,es,U):
        """Cartesian velocity matrix elements <n|v_alpha|m> between the
        columns of U, eigenvectors of H(k) with energies es. With
        v_alpha = dH/dK_alpha + i[H,r_alpha] (kubo._velocities), the
        derivative is applied to U through the sparse hoppings, at cost
        nnz*nw, and the commutator is i(E_n - E_m)<n|r_alpha|m>, at cost
        n*nw^2, so no n x n matrix is formed."""
        nw = U.shape[1]
        dim = len(self.orders)
        ph = np.exp(2j*np.pi*(self.dirs@np.array(k,dtype=np.float64)))
        dk = [] # U^dagger dH/dk_i U, per periodic direction
        Uc = np.conjugate(U.T)
        for i in range(dim):
            if self.tstack is None:
                dk.append(np.zeros((nw,nw),dtype=np.complex128)); continue
            c = 2j*np.pi*self.dirs[:,i]*ph # d/dk_i of the Bloch phases
            Y = (c[:,None,None]*U[None,:,:]).reshape(-1,nw)
            dk.append(Uc@(self.tstack@Y))
        v = np.zeros((3,nw,nw),dtype=np.complex128)
        de = es[:,None]-es[None,:] # E_n - E_m
        for a in range(3):
            for i in range(dim): v[a] += self.jac[i,a]*dk[i]
            ra = (Uc*self.rr[None,:,a])@U # <n|r_alpha|m>
            v[a] += 1j*de*ra
        return v


# ------------------------------------------------------------ solvers

def _dense_range(ws,ks,a,b):
    """Bands a to b-1 (indices in the sorted spectrum) and their velocities
    on every k-point of ks"""
    nk,nw = len(ks),b-a
    es = np.zeros((nk,nw))
    vs = np.zeros((nk,3,nw,nw),dtype=np.complex128)
    if ws.n<=_full_eigh_dimension: # all the bands, batched
        from ..htk.eigenvectors import parallel_diagonalization
        for i0 in range(0,nk,_kbatch):
            kb = ks[i0:i0+_kbatch]
            hks = np.array([ws.hk(k).toarray() for k in kb])
            eb,wb = parallel_diagonalization(hks)
            for j,k in enumerate(kb):
                es[i0+j] = eb[j,a:b]
                vs[i0+j] = ws.velocities(k,es[i0+j],wb[j][:,a:b])
        return es,vs
    for ik,k in enumerate(ks): # only the bands of the range
        e,U = dlg.eigh(ws.hk(k).toarray(),subset_by_index=[a,b-1],
                driver="evr")
        es[ik] = e
        vs[ik] = ws.velocities(k,e,U)
    return es,vs


def _sparse_near_fermi(ws,k,nb,na):
    """At least nb eigenpairs below and na above the Fermi level, from a
    shift-invert solver around it; returns the nb highest below and the na
    lowest above, sorted"""
    m = csc_matrix(ws.hk(k))
    kk = nb+na+4
    while True:
        if kk>=ws.n-1: # ARPACK cannot, all the bands instead
            e,U = dlg.eigh(m.toarray())
        else: e,U = algebra.arpack_eigh(m,k=kk,which="LM",sigma=0.)
        below = np.nonzero(e<0.)[0]
        above = np.nonzero(e>=0.)[0]
        if (len(below)>=nb and len(above)>=na) or kk>=ws.n-1:
            sel = np.concatenate([below[len(below)-min(nb,len(below)):],
                    above[:min(na,len(above))]])
            return e[sel],U[:,sel]
        kk = 2*kk


def _sparse_counts(ws,ks,nb,na):
    """nb bands below and na above the Fermi level, counted from it, on
    every k-point of ks"""
    nk,nw = len(ks),nb+na
    es = np.zeros((nk,nw))
    vs = np.zeros((nk,3,nw,nw),dtype=np.complex128)
    for ik,k in enumerate(ks):
        e,U = _sparse_near_fermi(ws,k,nb,na)
        if len(e)<nw: raise ValueError("the band window needs "+str(nb)
                +" bands below and "+str(na)+" above the Fermi level at "
                "every k-point, and the spectrum at k="+str(list(k))+" has "
                "fewer; on the sparse path the window must lie inside the "
                "spectrum, give a narrower one")
        es[ik] = e
        vs[ik] = ws.velocities(k,e,U)
    return es,vs


# ------------------------------------------------------------ the range

def _prescan_ks(ks):
    """A few k-points of the mesh, spread over it, for the first estimate"""
    idx = np.unique(np.linspace(0,len(ks)-1,min(_nprescan,len(ks))).astype(int))
    return ks[idx]


def _first_range(ws,window,ks):
    """First estimate of the index range from all the eigenvalues on a few
    k-points (dense), or of the counts below and above the Fermi level
    (sparse)"""
    if ws.sparse: # counts from the Fermi level
        if not window.energy: return window.nv,window.nc
        nb,na = 1,1
        for k in _prescan_ks(ks):
            while True: # grow until the window is inside the bands found
                e,_ = _sparse_near_fermi(ws,k,nb+1,na+1)
                nb2 = max(nb,int(np.sum((e>window.emin)&(e<0.))))
                na2 = max(na,int(np.sum((e<window.emax)&(e>=0.))))
                if (nb2,na2)==(nb,na): break
                nb,na = nb2,na2
        return nb,na
    lo,hi = ws.n,0
    for k in _prescan_ks(ks):
        e = dlg.eigvalsh(ws.hk(k).toarray())
        if window.energy:
            inside = np.nonzero((e>window.emin)&(e<window.emax))[0]
            if len(inside)==0: continue
            lo,hi = min(lo,inside[0]),max(hi,inside[-1]+1)
        else:
            nocc = int(np.sum(e<0.))
            lo,hi = min(lo,nocc-window.nv),max(hi,nocc+window.nc)
    if hi<=lo: # no band inside the window on the prescan
        nocc = int(np.sum(dlg.eigvalsh(ws.hk(ks[0]).toarray())<0.))
        lo,hi = nocc-1,nocc+1
    return max(lo,0),min(hi,ws.n)


def _requested(window,es,cols,side):
    """Whether the band in column cols of es (one per k-point) is one that
    the window asks for at that k-point: inside (emin,emax), or among the
    nv bands below (side=-1) or the nc above (side=+1) the Fermi level"""
    e = es[np.arange(len(es)),cols] if np.ndim(cols) else es[:,cols]
    if window.energy: return (e>window.emin)&(e<window.emax)
    if side<0:
        nb = np.sum((es<0.)&(np.arange(es.shape[1])[None,:]>=cols),axis=1)
        return (e<0.)&(nb<=window.nv)
    na = np.sum((es>=0.)&(np.arange(es.shape[1])[None,:]<=cols),axis=1)
    return (e>=0.)&(na<=window.nc)


def _growth(window,es,nw,plo,phi,tol):
    """How many bands the range has to grow by, below and above, for the
    window to hold on these k-points. The columns of es are plo probe bands
    below the range, its nw bands, and phi probe bands above it.

    A probe band that enters the energy window at some k-point is added,
    and so is every band between it and the range. In the count form the
    range must hold nv bands below the Fermi level and nc above at every
    k-point. Then the edge of the range is not allowed to cut a degenerate
    multiplet at a k-point where the edge band is one the window asks for:
    there the band kept would be a combination fixed by the eigensolver.
    Where the edge band lies outside the window its pairs are among those
    the window drops anyway, which is what keeps a model with near
    degeneracies at every k-point (the two valleys of a twisted bilayer)
    from growing the range without end."""
    core = es[:,plo:plo+nw]
    if window.energy:
        glo = max([j for j in range(1,plo+1)
                if np.any(es[:,plo-j]>window.emin)],default=0)
        ghi = max([j for j in range(1,phi+1)
                if np.any(es[:,plo+nw-1+j]<window.emax)],default=0)
    else:
        nbelow = np.sum(core<0.,axis=1) # bands of the range below E_F
        glo = max(0,window.nv-int(np.min(nbelow)))
        ghi = max(0,window.nc-int(np.min(nw-nbelow)))
    c = plo-glo # column of the new lowest band
    while 0<c<=plo and np.any((np.abs(es[:,c]-es[:,c-1])<tol)
            &_requested(window,es,c,-1)):
        glo += 1; c -= 1
    c = plo+nw-1+ghi # column of the new highest band
    while plo+nw-1<=c<es.shape[1]-1 and np.any(
            (np.abs(es[:,c+1]-es[:,c])<tol)&_requested(window,es,c,1)):
        ghi += 1; c += 1
    return glo,ghi


def bands_and_velocities(h,ks,window,tol_rel=1e-6,ws=None,fixed=None):
    """Bands of the window and their velocity matrix elements on every
    k-point of ks: es[k,n], vs[k,alpha,n,m], as kubo._bands_and_velocities
    returns for all the bands, together with the cell volume, the energy
    scale, and the range that was used.

    The range is found here, growing it until the window holds on the whole
    of ks, unless fixed (a range from an earlier call) is given; then it is
    only checked, with a warning if a band outside it enters the window.
    The range is (lo,hi), band indices lo to hi-1 of the sorted spectrum,
    or on the sparse path (nb,na), the nb bands below the Fermi level and
    the na above it."""
    if ws is None: ws = WindowSetup(h)
    tol = tol_rel*ws.scale
    ks = np.array(ks,dtype=np.float64)
    rng = _first_range(ws,window,ks) if fixed is None else fixed
    while True:
        if ws.sparse:
            nb,na = rng
            for plo in range(_nprobe,-1,-1): # fewer probes at the ends
                try:
                    es,vs = _sparse_counts(ws,ks,nb+plo,na+plo)
                    break
                except ValueError:
                    if plo==0: raise
            phi = plo
            lo,hi = 0,nb+na
        else:
            lo,hi = rng
            a,b = max(lo-_nprobe,0),min(hi+_nprobe,ws.n) # probes each side
            es,vs = _dense_range(ws,ks,a,b)
            plo,phi = lo-a,b-hi
        glo,ghi = _growth(window,es,hi-lo,plo,phi,tol)
        end_lo,end_hi = plo<_nprobe,phi<_nprobe # the spectrum ends there
        if end_lo: glo = min(glo,plo)
        if end_hi: ghi = min(ghi,phi)
        if fixed is not None:
            if glo or ghi: warnings.warn("the band window "+str(tuple(rng))
                +" fixed on the coarse scan misses bands that enter the "
                "window on later k-points; give a wider window")
            glo = ghi = 0
        if ws.sparse: rng = (nb+glo,na+ghi)
        else: rng = (lo-glo,hi+ghi)
        if (glo<plo or end_lo) and (ghi<phi or end_hi):
            # the bands added are probes already computed, and the probes
            # left beyond them verify the new edges: keep them
            i0,i1 = plo-glo,plo+hi-lo+ghi
            es = np.ascontiguousarray(es[:,i0:i1])
            vs = np.ascontiguousarray(vs[:,:,i0:i1,i0:i1])
            break
        # else grow beyond the probes and do the mesh again
    return es,vs,ws.cellvol,ws.scale,tuple(int(x) for x in rng)
