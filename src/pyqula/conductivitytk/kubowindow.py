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
# then checked on the whole mesh with one probe band below and one above
# it; if a probe band enters the window, or is degenerate with the edge
# band, the range grows and the mesh is done again.
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


def _violations(window,es,lo,hi,tol,probe_lo,probe_hi):
    """How many bands the range has to grow by, below and above, for the
    window to hold on these k-points. es holds the bands of the range with
    a probe band on each side where there is one (probe_lo, probe_hi). A
    probe band that enters the window, or is degenerate with the edge band
    of the range, is added. In the count form, there must be nv bands of
    the range below the Fermi level and nc above at every k-point."""
    grow_lo = grow_hi = 0
    nw = hi-lo
    core = es[:,int(probe_lo):int(probe_lo)+nw]
    if window.energy:
        if probe_lo and np.any(es[:,0]>window.emin): grow_lo = 1
        if probe_hi and np.any(es[:,-1]<window.emax): grow_hi = 1
    else:
        nbelow = np.sum(core<0.,axis=1) # bands of the range below E_F
        grow_lo = max(0,window.nv-int(np.min(nbelow)))
        grow_hi = max(0,window.nc-int(np.min(nw-nbelow)))
    # do not cut a degenerate multiplet at the edges of the range
    if probe_lo and np.any(np.abs(core[:,0]-es[:,0])<tol): grow_lo = max(grow_lo,1)
    if probe_hi and np.any(np.abs(es[:,-1]-core[:,-1])<tol): grow_hi = max(grow_hi,1)
    return grow_lo,grow_hi


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
            es,vs = _sparse_counts(ws,ks,nb+1,na+1) # one probe each side
            plo = phi = True
            lo,hi = 0,nb+na
        else:
            lo,hi = rng
            a,b = max(lo-1,0),min(hi+1,ws.n) # one probe band each side
            es,vs = _dense_range(ws,ks,a,b)
            plo,phi = a<lo,b>hi
        glo,ghi = _violations(window,es,lo,hi,tol,plo,phi)
        if ws.sparse: grown = (nb+glo,na+ghi)
        else: grown = (max(lo-glo,0),min(hi+ghi,ws.n))
        i0 = int(plo) # drop the probes
        es = np.ascontiguousarray(es[:,i0:i0+hi-lo])
        vs = np.ascontiguousarray(vs[:,:,i0:i0+hi-lo,i0:i0+hi-lo])
        if grown==tuple(rng): break
        if fixed is not None:
            warnings.warn("the band window "+str(tuple(rng))+" fixed on the "
                "coarse scan misses bands that enter the window on later "
                "k-points; give a wider window")
            break
        rng = tuple(int(x) for x in grown) # grow and do the mesh again
    return es,vs,ws.cellvol,ws.scale,tuple(int(x) for x in rng)
