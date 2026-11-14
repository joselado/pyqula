# Simplex (linear triangle) integration of the Kubo-Greenwood optical
# conductivity, and its quantics tensor cross interpolation (QTCI) variant.
# The formula, the units and the sign conventions are those of kubo.py and
# conductivity.py; this module only replaces the k-sum.
#
# The grid method of kubo.py samples 1/(omega + i delta - dE_nm(k)) on the
# mesh, so it needs a mesh whose energy spacing is below delta. Here the
# transition energy dE_nm(k) is interpolated linearly inside each simplex of
# the mesh (segments in 1D, triangles in 2D, tetrahedra in 3D), the weight
# W^ab_nm(k) = ratio_nm v^a_nm v^b_mn is averaged over its vertices, and the
# resolvent is integrated over the simplex exactly. The mesh then only has to resolve the smooth
# functions dE and W, not the broadening, and every frequency comes from the
# same mesh.
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
    nv = simplices.shape[1] # vertices per simplex, 2, 3 or 4
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
    """Simplices of the periodic nk (1D), nk x nk (2D) or nk^3 (3D) mesh,
    as vertex indices into the row-major list of klist.kmesh, and the
    volume fraction of the Brillouin zone each one covers. The 3D
    tetrahedra are those of dostk/simplexdos.py"""
    if dim==3:
        from ..dostk.simplexdos import simplices as simplices3d
        return simplices3d(3,nk)
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
#   psi_K(x) at the vertex energies, K = 1, 2, 3 in 1D, 2D, 3D (exact
#   branches),
#   the Taylor series of psi_K[...] over a close cluster of nodes (an edge,
#   a face or the whole simplex) around its mean.
# In other words the weighted joint density of states, a sum of linear
# B-splines, is stored by its knots. The coefficients are built once, the
# vertex ones indexed by k-point, so the simplices sharing a vertex share
# one source, and the sum is then evaluated per frequency over the sources.
#
# The branch of a simplex is its Newton table, read from the top: the
# divided difference over the sorted nodes e_lo..e_hi of spread s is
# (psi_K[e_(lo+1)..e_hi] - psi_K[e_lo..e_(hi-1)])/s, and a cluster with
# s < _alpha delta is not split further but kept as one point source. For
# a tetrahedron this gives at most two faces, three edges and four
# vertices; two, three or four equal (or nearly equal) nodes end in a
# point source, whose series has only its first term when the nodes are
# exactly equal.
#
# The branch cannot depend on omega here, so the rule has to hold at every
# frequency at once. A cluster of m+1 nodes of spread s has its nodes
# within m s/(m+1) of their mean, and its Taylor term of order n is bounded
# by (m s/((m+1) |c - z|))^n up to a power of n. |c - z| >= delta for every
# real omega, so for s < _alpha delta the series converges at a rate of at
# least 3 _alpha/4 for any omega (2/3 for a triangle); it runs until two
# terms in a row are below 1e-17 of the sum, which takes up to about 80
# terms on resonance for a triangle, about 130 for a tetrahedron, and a few
# far from it. The series keeps its first order term, which vanishes for
# the exact mean, because the stored center is the rounded mean. What is
# left is the exact branch for a cluster of spread just above _alpha delta,
# evaluated at a frequency far from its mean, which loses eps (|x|/s)^m of
# that cluster, m = 3 for a whole tetrahedron. The rule used before,
# s < 1e-2 delta with a second order series, lost 1e4 times more in 2D.
#
# Only a rule per frequency removes this last loss, so the simplices where
# it matters are kept out of the vertex form. While building the sources
# we bound, for every simplex and pair, the rounding error of its vertex
# form over the actual frequency window (_vertex_error: eps times the sum
# over the leaves of the Newton table of |coefficient| |psi_K[leaf]| at the
# farthest frequency), and when it is above _tol of the peak of the
# simplex, the nodes of its Newton table that cancel too much are point
# sources evaluated per frequency with simplexdos._dd, as the direct
# kernel does, see _decompose. In 2D the loss is
# quadratic: on the spinful Haldane model at delta = 1e-3 the rule catches
# 0.3%, 1.1% and 3.2% of the forward triangle pairs at nk = 128, 256
# and 512, and without it the sum is still within 1.5e-12 of a longdouble
# reference instead of 7e-13. At delta = 1e-5 it matters: 2.4e-10 without
# it at nk = 128, 2.8e-12 with it. In 3D the loss is cubic. Without this
# rule the source kernel lost up to 1e-9 of the peak of a tetrahedron at
# delta = 1e-2 and all of it at delta = 1e-6.
#
# Most of the simplices that this rule keeps whole are far from every
# frequency of the window, the reverse terms at -dE above all. Those go
# to energy bins instead: a simplex whose nodes lie within _rho of the
# distance from the center C of its bin to the window is expanded around
# C, K! psi_K[e] = sum_k K! k!/(k+K)! h_k(e - C) (z - C)^-(k+1), which
# converges at a rate _rho or better at every frequency of the window and
# has no cancellation, and the bin keeps the sum of these moments. A bin
# costs _kmax terms per frequency, whatever the number of simplices in it.
# A simplex adds only the orders it needs to its bin: its term of order k
# is below r^k of the first, with r its own rate, so it stops at r^k <
# 1e-18, the size of the last of the _kmax terms at the rate _rho. On the
# 2D models of the tests at nk = 512 and the window [0, 7] t, the binned
# triangles (all the reverse terms) stop after 8 terms on average.
# On the 3D test model at nk = 48 and delta = 1e-2 this leaves 7% of the
# simplex pairs with nodes evaluated per frequency, against 35% without
# the bins.
#
# A simplex with nodes evaluated per frequency is close to some frequency
# of the window, but far from most of them. Such a simplex was first kept
# whole, which in 3D made the tetrahedra with a close edge or face expensive,
# since they are wide (a median distance of 0.23 t from the center of
# their bin on the layered model of the tests at nk = 24). It is now split
# along its Newton table as long as the parts stay within its rounding
# budget, so that only the close edges and faces are evaluated per
# frequency, and these are shared by the neighboring tetrahedra. With
# _budget = 1/100 the 7000 kept tetrahedra of that model become 4742 edges
# and 4679 faces after the merge, and the rest goes to the vertex form.
#
# A point source, a Taylor series or a node evaluated per frequency, is
# narrow, of spread below delta for the first, so far from it the
# expansion around the center of its energy bin holds as well, with the
# logarithmic terms of orders below K (_add_node_moments). The point
# sources form sets, by bin and with the two kinds apart, and a set keeps
# the moments of its sources and the largest distance R of their nodes
# from its center C. At a frequency with (w - C)^2 + delta^2 >= (R/_rho)^2
# all of them converge at a rate _rho or better and the set gives their
# sum, with the orders that its own rate R/|z - C| needs; at the others
# each source is evaluated one by one. One decision per set and frequency
# serves both, so no source is counted twice or left out. The energy bins
# of the far simplices take the orders that their own rate needs at each
# frequency in the same way.
#
# The vertex sources cancel at every frequency: the sum of |coefficient|
# |psi_K| over them is 5e3 times the largest entry on the 3D layered model
# of the tests at nk = 24, and 8e3 at nk = 48. A plain running sum loses
# eps times this times a factor that grows with the number of sources, so
# the error of the source kernel grew from 2e-13 to 7e-13 of the largest
# entry between nk = 24 and 48, against the 3e-14 to 7e-14 of the direct
# kernel. The sum per frequency is now compensated (_kadd), which brings
# it to the level of the direct kernel on that model. In 2D the source
# kernel stays above the direct one (3.5e-13 against 4e-14 at 128^2 and
# delta = 1e-3), from the vertex form below _tol. What is left is the
# rounding of each term and the vertex form of the simplices below _tol,
# up to 4e-13 on a 3D metal, where _tol = 3e-12 would give 1e-14 at 1.7
# to 2.8 times the cost (future_development/qtci_optical_response.md).
# With the compensated sum _budget = 1/100 is as precise as keeping the
# simplices whole (_budget = 0) on every model measured, and 1/10 or 1 is
# not on all of them.

_alpha = 1. # see above
_nmax = 200 # largest order of the Taylor series of a point source
_tol = 3e-11 # largest rounding error of a simplex in its vertex form,
# relative to its peak, before its nodes that cancel too much are
# evaluated per frequency
_budget = 1e-2 # fraction of _tol left to the rounding of such a simplex,
# once its nodes that cancel too much are evaluated per frequency
_direct = 10 # type of a point source evaluated by simplexdos._dd
_ntasks = 64 # tasks of the sum over sources, see _sigma_sources
_nbins = 512 # energy bins of the far simplices
_rho = 0.5 # a simplex goes to its bin when its nodes are within _rho times
# the distance from the bin center to the frequency window
_kmax = 60 # order of the expansion of a bin, _rho^_kmax < 1e-17


@jit(nopython=True,cache=True)
def _sort_nodes(e,k,nv):
    """insertion sort of the first nv energies, carrying the vertex indices"""
    for i in range(1,nv):
        v = e[i]
        kv = k[i]
        j = i-1
        while j>=0 and e[j]>v:
            e[j+1] = e[j]
            k[j+1] = k[j]
            j -= 1
        e[j+1] = v
        k[j+1] = kv


@jit(nopython=True,cache=True)
def _leaf_bound(e,lo,hi,K,harm,wlo,whi,delta):
    """m! times a bound of |psi_K[e_lo..e_hi]| over the frequencies
    [wlo,whi], m = hi-lo < K: psi_(K-m) at the farthest frequency, and
    -ln x at least at |x| = delta"""
    m = hi-lo
    X = max(abs(e[lo]-wlo),abs(e[hi]-whi),abs(e[lo]-whi),abs(e[hi]-wlo),
            delta)
    if m==K-1: return max(abs(np.log(X)),abs(np.log(delta)))+np.pi # -ln x
    # x^(K-m-1) (ln x - H)/(K-m-1)!, largest at X
    P = abs(np.log(X))+np.pi+harm[K-m-1]
    for i in range(K-m-1): P *= X/(i+1)
    return P


@jit(nopython=True,cache=True)
def _node_sum(e,lo0,hi0,f0,K,tolc,fact,harm,wlo,whi,delta,slo,shi,sco):
    """sum over the leaves (vertices and close clusters) of the Newton
    table of f0 psi_K[e_lo0..e_hi0] of |coefficient| times the bound of
    |psi_K[leaf]| over the window; eps times this bounds the rounding
    error of its vertex form"""
    top = 1
    slo[0] = lo0
    shi[0] = hi0
    sco[0] = f0
    err = 0.
    while top>0:
        top -= 1
        lo = slo[top]
        hi = shi[top]
        f = sco[top]
        sp = e[hi]-e[lo]
        if lo==hi or sp<tolc: # a leaf
            err += abs(f)*_leaf_bound(e,lo,hi,K,harm,wlo,whi,delta)/fact[hi-lo]
            continue
        slo[top] = lo+1
        shi[top] = hi
        sco[top] = f/sp
        slo[top+1] = lo
        shi[top+1] = hi-1
        sco[top+1] = -f/sp
        top += 2
    return err


@jit(nopython=True,cache=True)
def _vertex_error(e,nv,tolc,fact,harm,wlo,whi,delta,slo,shi,sco):
    """bound on the rounding error of the vertex form of K! psi_K[e], the
    Newton table of _simplex_sources, over the frequencies [wlo,whi]:
    eps times the sum over the leaves of the table (vertices and close
    clusters) of |coefficient| times a bound of |psi_K[leaf]| over the
    window, relative to pi/(s + pi delta), a lower bound of the peak of
    the simplex (the density of its energies reaches 1/s at least). A
    simplex that is one close cluster has no cancellation."""
    K = nv-1
    if e[K]-e[0]<tolc: return 0.
    err = _node_sum(e,0,K,fact[K],K,tolc,fact,harm,wlo,whi,delta,slo,shi,sco)
    return 4.4e-16*err*(e[K]-e[0]+np.pi*delta)/np.pi


@jit(nopython=True,cache=True)
def _node_mean(e,lo,hi):
    """mean of the nodes e_lo..e_hi, the center of a point source"""
    cm = 0.
    for i in range(lo,hi+1): cm += e[i]
    return cm/(hi-lo+1)


@jit(nopython=True,cache=True)
def _decompose(e,nv,tolc,fact,harm,wlo,whi,delta,btol,slo,shi,sco,sbu,sh,
        work):
    """the nodes of the Newton table of K! psi_K[e], K = nv-1, that are
    evaluated per frequency, for a simplex whose vertex form loses more
    than _tol of its peak somewhere on [wlo,whi]. A node is numbered as in a
    binary heap, the whole simplex 1 and the children of node h, without
    its first node and without its last, 2h and 2h+1; the result has the
    bit h-1 set for every node evaluated per frequency. The rounding budget
    of the simplex is btol of its peak, btol = _budget _tol by default,
    in the units of _node_sum.
    Top down, a node whose vertex form stays within its budget is left in
    vertex form, a node whose two children, evaluated per frequency, would
    each stay within half of it is split with half of it for each child,
    and the others (an edge at least, whose children are vertices) are
    evaluated per frequency. Since the budgets of the leaves add up to that
    of the simplex, the vertex form of the rest and the rounding of the
    split nodes stay within it."""
    K = nv-1
    top = 1
    slo[0] = 0
    shi[0] = K
    sco[0] = fact[K]
    sbu[0] = btol*np.pi/(4.4e-16*(e[K]-e[0]+np.pi*delta))
    sh[0] = 1
    code = 0
    while top>0:
        top -= 1
        lo = slo[top]
        hi = shi[top]
        f = sco[top]
        b = sbu[top]
        h = sh[top]
        s = e[hi]-e[lo]
        if lo==hi or s<tolc: continue # a vertex or a close cluster
        if _node_sum(e,lo,hi,f,K,tolc,fact,harm,wlo,whi,delta,work[0],
                work[1],work[2])<=b:
            continue # in vertex form
        if hi-lo==1 or abs(f)/s*max(
                _leaf_bound(e,lo+1,hi,K,harm,wlo,whi,delta),
                _leaf_bound(e,lo,hi-1,K,harm,wlo,whi,delta)
                )/fact[hi-lo-1]>0.5*b:
            code |= 1<<(h-1) # evaluated per frequency
            continue
        slo[top] = lo+1
        shi[top] = hi
        sco[top] = f/s
        sbu[top] = 0.5*b
        sh[top] = 2*h
        slo[top+1] = lo
        shi[top+1] = hi-1
        sco[top+1] = -f/s
        sbu[top+1] = 0.5*b
        sh[top+1] = 2*h+1
        top += 2
    return code


@jit(nopython=True,cache=True)
def _simplex_sources(mode,e,kv,nv,wbar,tolc,Av,p,n,ep,tp,yp,ca,fact,
        slo,shi,sco,sh,code):
    """sources of one simplex and one pair, with the sorted node energies
    e[0..nv-1] at the k-points kv. mode=0 only counts the point sources,
    mode=1 writes them to (ep,tp,yp,ca) from index n on, mode=2 adds the
    vertex coefficients of psi_K, K = nv-1, to Av[k,p,c]. A point source of
    type m is the cluster of m+1 nodes ep + yp[0..m], and stands for psi_K
    over them; of type _direct+m it is evaluated per frequency. slo, shi,
    sco and sh are the stack of the Newton table, and code the nodes
    evaluated per frequency, see _decompose (0 for none). Returns the new
    n."""
    K = nv-1
    top = 1
    slo[0] = 0
    shi[0] = K
    sco[0] = fact[K] # K! psi_K[e_0..e_K] is the simplex average
    sh[0] = 1
    while top>0:
        top -= 1
        lo = slo[top]
        hi = shi[top]
        f = sco[top]
        h = sh[top]
        if lo==hi: # a vertex
            if mode==2:
                for c in range(wbar.shape[0]): Av[kv[lo],p,c] += f*wbar[c]
            continue
        s = e[hi]-e[lo]
        if s<tolc: # a close cluster, its Taylor series
            if mode==1:
                cm = _node_mean(e,lo,hi)
                ep[n] = cm
                tp[n] = hi-lo
                for i in range(4):
                    yp[n,i] = e[lo+i]-cm if lo+i<=hi else 0.
                for c in range(wbar.shape[0]): ca[n,c] = f*wbar[c]
            n += 1
            continue
        if h<=7 and (code>>(h-1))&1: # evaluated per frequency
            n = _direct_node(mode,e,lo,hi,f,wbar,n,ep,tp,yp,ca)
            continue
        # (psi_K[e_(lo+1)..e_hi] - psi_K[e_lo..e_(hi-1)])/s
        slo[top] = lo+1
        shi[top] = hi
        sco[top] = f/s
        sh[top] = 2*h
        slo[top+1] = lo
        shi[top+1] = hi-1
        sco[top+1] = -f/s
        sh[top+1] = 2*h+1
        top += 2
    return n


@jit(nopython=True,cache=True)
def _direct_node(mode,e,lo,hi,f,wbar,n,ep,tp,yp,ca):
    """f psi_K[e_lo..e_hi] as a point source evaluated per frequency, see
    _simplex_sources. Returns the new n."""
    if mode==1:
        cm = _node_mean(e,lo,hi)
        ep[n] = cm
        tp[n] = _direct+hi-lo
        for i in range(4): yp[n,i] = e[lo+i]-cm if lo+i<=hi else 0.
        for c in range(wbar.shape[0]): ca[n,c] = f*wbar[c]
    return n+1


@jit(nopython=True,cache=True)
def _bin_of(cm,bins,nb):
    """energy bin of a simplex of mean energy cm, among nb bins of centers
    bins[0] + (j+1/2) bins[1]"""
    j = int((cm-bins[0])/bins[1])
    return min(max(j,0),nb-1)


@jit(nopython=True,cache=True)
def _bins_of(cm,bins,nb):
    """_bin_of of every entry of cm"""
    out = np.zeros(cm.shape[0],dtype=np.int64)
    for i in range(cm.shape[0]): out[i] = _bin_of(cm[i],bins,nb)
    return out


@jit(nopython=True,cache=True)
def _chunk_sources(mode,i0,i1,n,simplices,dE,W,trans,diag,tolc,fact,harm,
        wlo,whi,delta,tol,btol,flag,Af,Ar,ep,tp,yp,ca,bins,M,R):
    """the sources of the simplices i0..i1-1, see _sources_jit, from point
    index n on. mode=0 also decides the branch of every simplex and pair,
    flag[it,p,0] forward and flag[it,p,1] reverse: 0 for the vertex form,
    2 when it goes to its energy bin, and -code when the nodes code of its
    Newton table (see _decompose) are evaluated per frequency. mode=3 adds
    the binned simplices to the moments M[j,k,c] of the bins, whose
    centers are bins[0] + (j+1/2) bins[1], and the largest distance R[j]
    of their nodes from the center. Returns the new n."""
    nv = simplices.shape[1]
    npair = dE.shape[1]
    nc = W.shape[2]
    wbar = np.zeros(nc,dtype=np.complex128)
    wrev = np.zeros(nc,dtype=np.complex128)
    e = np.zeros(nv)
    kv = np.zeros(nv,dtype=np.int64)
    slo = np.zeros(2*nv,dtype=np.int64) # stack of the Newton table
    shi = np.zeros(2*nv,dtype=np.int64)
    sco = np.zeros(2*nv)
    sbu = np.zeros(2*nv)
    sh = np.zeros(2*nv,dtype=np.int64)
    work = (np.zeros(2*nv,dtype=np.int64),np.zeros(2*nv,dtype=np.int64),
            np.zeros(2*nv)) # stack of _node_sum
    for it in range(i0,i1): # loop over simplices
        for p in range(npair): # loop over stored pairs
            if mode>0: # the branches are known, skip what this mode leaves
                skip = True
                for r in range(1 if diag[p] else 2):
                    if _wanted(mode,flag[it,p,r]): skip = False
                if skip: continue
            for c in range(nc): wbar[c] = 0.
            for v in range(nv):
                for c in range(nc): wbar[c] += W[simplices[it,v],p,c]
            for c in range(nc): wbar[c] = wbar[c]/nv
            for c in range(nc): wrev[c] = wbar[trans[c]]
            for r in range(2): # forward, and the pair (m,n) at energy -dE
                if r==1 and diag[p]: break
                if mode>0 and not _wanted(mode,flag[it,p,r]): continue
                sg = 1. if r==0 else -1.
                for v in range(nv):
                    kv[v] = simplices[it,v]
                    e[v] = sg*dE[kv[v],p]
                _sort_nodes(e,kv,nv)
                C = 0.
                j = 0
                if mode==0 or mode==3: # the bin of the simplex
                    cm = 0.
                    for v in range(nv): cm += e[v]
                    cm /= nv
                    j = _bin_of(cm,bins,M.shape[0])
                    C = bins[0]+(j+0.5)*bins[1]
                if mode==0:
                    rad = max(abs(e[0]-C),abs(e[nv-1]-C))
                    D = max(wlo-C,C-whi,0.)
                    if rad*rad<=_rho*_rho*(D*D+delta*delta): flag[it,p,r] = 2
                    elif _vertex_error(e,nv,tolc,fact,harm,wlo,whi,delta,
                            slo,shi,sco)>tol: # some nodes per frequency
                        flag[it,p,r] = -_decompose(e,nv,tolc,fact,harm,wlo,
                                whi,delta,btol,slo,shi,sco,sbu,sh,work)
                    else: flag[it,p,r] = 0
                if mode==3 and flag[it,p,r]==2:
                    rad = max(abs(e[0]-C),abs(e[nv-1]-C))
                    R[j] = max(R[j],rad)
                    D = max(wlo-C,C-whi,0.)
                    _add_moments(e,nv,C,wbar if r==0 else wrev,M,j,fact,
                            rad/np.sqrt(D*D+delta*delta))
                    continue
                if flag[it,p,r]==2: continue
                n = _simplex_sources(mode,e,kv,nv,wbar if r==0 else wrev,
                        tolc,Af if r==0 else Ar,p,n,ep,tp,yp,ca,fact,
                        slo,shi,sco,sh,-min(flag[it,p,r],0))
    return n


@jit(nopython=True,cache=True)
def _wanted(mode,f):
    """whether a simplex pair of branch f has something for the pass mode:
    the count (0) takes all, the point sources (1) and the vertex sources
    (2) those not binned, the moments (3) the binned"""
    if mode==0: return True
    if mode<=2: return f!=2
    return f==2


@jit(nopython=True,cache=True)
def _add_moments(e,nv,C,wbar,M,j,fact,r):
    """add the simplex with nodes e to the moments of bin j, of center C:
    K! psi_K[e] = sum_k K! k!/(k+K)! h_k(e - C) u^(k+1), u = 1/(z - C),
    which converges for |z - C| > max |e - C|. h_k of K+1 numbers below
    rad in size has (k+K)!/(k! K!) terms, so the term of order k is below
    r^k |u|, with r = rad |u| the rate of the window, and the series
    stops when r^k < 1e-18, as _kmax terms do at the largest rate _rho"""
    K = nv-1
    y0 = e[0]-C; y1 = e[1]-C
    y2 = e[2]-C if nv>2 else 0.
    y3 = e[3]-C if nv>3 else 0.
    H0 = 1.; H1 = 1.; H2 = 1.; H3 = 1. # h_k(y_0..y_i), i = 0..3
    g = 1. # K! k!/(k+K)!
    kmax = M.shape[1]
    if r<1e-18: kmax = 1
    elif r<_rho: kmax = min(kmax,int(np.ceil(np.log(1e-18)/np.log(r))))
    for k in range(kmax):
        if k>0:
            H0 = y0*H0
            H1 = H0+y1*H1
            H2 = H1+y2*H2
            H3 = H2+y3*H3
            g *= k/(k+K)
        hk = H1 if nv==2 else (H2 if nv==3 else H3)
        f = g*hk
        for c in range(wbar.shape[0]): M[j,k,c] += f*wbar[c]


@jit(nopython=True,cache=True)
def _add_node_moments(e,lo,hi,K,C,f,wbar,M,L,j,fact):
    """add f psi_K[e_lo..e_hi] to the moments of bin j, of center C, to
    the order of M, the rate _rho: with y = e - C, m = hi-lo and
    u = 1/(z - C), psi_K[e_lo..e_hi] = sum_n psi_K^(m+n)(C) h_n(y)/(m+n)!,
    whose terms of order i = m+n < K are psi_(K-i)(C - z) h_n(y)/i!, kept
    in L[j,i], and of order i = K+k are k!/(k+K)! h_n(y) u^(k+1), kept in
    M[j,k]; for m = K these are the terms of _add_moments. A point source
    has m >= 1 (a single node is a vertex source), so L[j,0] stays zero"""
    m = hi-lo
    y0 = e[lo]-C
    y1 = e[lo+1]-C if m>0 else 0.
    y2 = e[lo+2]-C if m>1 else 0.
    y3 = e[lo+3]-C if m>2 else 0.
    H0 = 1.; H1 = 1.; H2 = 1.; H3 = 1. # h_n(y_0..y_i), i = 0..3
    g = 1. # k!/(k+K)!
    for i in range(K): g /= (i+1)
    kmax = M.shape[1]
    for n in range(K-m+kmax):
        if n>0:
            H0 = y0*H0
            H1 = H0+y1*H1
            H2 = H1+y2*H2
            H3 = H2+y3*H3
        hn = H0 if m==0 else (H1 if m==1 else (H2 if m==2 else H3))
        i = m+n
        if i<K: # a logarithmic term
            a = f*hn/fact[i]
            for c in range(wbar.shape[0]): L[j,i,c] += a*wbar[c]
            continue
        k = i-K
        if k>0: g *= k/(k+K)
        a = f*g*hn
        for c in range(wbar.shape[0]): M[j,k,c] += a*wbar[c]


@jit(nopython=True,parallel=True,cache=True)
def _sources_jit(mode,simplices,dE,W,trans,diag,tolc,fact,harm,wlo,whi,
        delta,tol,btol,chunk,start,flag,ep,tp,yp,ca,bins,nbins):
    """Count (mode=0) or write (mode=1) the point sources of the whole
    simplex sum, in parallel over chunks of simplices; the point sources of
    chunk ic go from start[ic] on. Clusters of spread below tolc are Taylor
    expanded. A simplex whose vertex form would lose more than tol of its
    peak at a frequency of [wlo,whi] has the nodes of its Newton table that
    cancel too much evaluated per frequency, as point sources, and a
    simplex far from the window goes to the moments of its energy bin;
    the count decides this and stores it in flag. Returns the number of point sources of every chunk."""
    ns = simplices.shape[0]
    nchunk = (ns+chunk-1)//chunk
    count = np.zeros(nchunk,dtype=np.int64)
    A0 = np.zeros((1,1,1),dtype=np.complex128) # the vertices are not written
    M0 = np.zeros((nbins,1,1),dtype=np.complex128) # nor the moments
    for ic in prange(nchunk): # parallel loop over chunks of simplices
        n0 = start[ic] if mode==1 else 0
        n = _chunk_sources(mode,ic*chunk,min(ns,(ic+1)*chunk),n0,simplices,
                dE,W,trans,diag,tolc,fact,harm,wlo,whi,delta,tol,btol,flag,
                A0,A0,ep,tp,yp,ca,bins,M0,np.zeros(nbins))
        count[ic] = n-n0
    return count


@jit(nopython=True,parallel=True,cache=True)
def _moments_jit(simplices,dE,W,trans,diag,fact,harm,wlo,whi,delta,flag,
        bins,M,R):
    """Moments M[g,j,k,c] of the binned simplices and the radii R[g,j]
    of the bins, group g of the simplices in its own copy, in parallel over
    the groups; [wlo,whi] is the frequency window, which sets how many
    moments each simplex needs"""
    ns = simplices.shape[0]
    ng = M.shape[0]
    e0 = np.zeros(1); t0 = np.zeros(1,dtype=np.int64)
    y0 = np.zeros((1,4)); c0 = np.zeros((1,1),dtype=np.complex128)
    A0 = np.zeros((1,1,1),dtype=np.complex128)
    for g in prange(ng): # parallel loop over groups of simplices
        _chunk_sources(3,(g*ns)//ng,((g+1)*ns)//ng,0,simplices,dE,W,trans,
                diag,0.,fact,harm,wlo,whi,delta,0.,0.,flag,A0,A0,e0,t0,y0,c0,
                bins,M[g],R[g])


@jit(nopython=True,parallel=True,cache=True)
def _vertex_sources_jit(simplices,dE,W,trans,diag,tolc,fact,harm,flag,
        Af,Ar):
    """Add the vertex coefficients of the whole simplex sum to Af (forward,
    energy dE) and Ar (reverse, energy -dE), with the branches stored in
    flag. In parallel over the stored pairs, whose entries are their own,
    and serial over the simplices, since the simplices sharing a vertex add
    to one entry, so the result does not depend on the number of threads."""
    for p in prange(dE.shape[1]): # parallel loop over stored pairs
        e0 = np.zeros(1); t0 = np.zeros(1,dtype=np.int64)
        y0 = np.zeros((1,4)); c0 = np.zeros((1,1),dtype=np.complex128)
        M0 = np.zeros((1,1,1),dtype=np.complex128)
        _chunk_sources(2,0,simplices.shape[0],0,simplices,dE[:,p:p+1],
                W[:,p:p+1],trans,diag[p:p+1],tolc,fact,harm,0.,0.,1.,0.,0.,
                flag[:,p:p+1],Af[:,p:p+1],Ar[:,p:p+1],e0,t0,y0,c0,
                np.zeros(2),M0,np.zeros(1))


@jit(nopython=True,cache=True)
def _vertex_value(K,x):
    """psi_K(x), the K-th antiderivative of 1/(z-E) at x = E - z"""
    lx = np.log(x)
    if K==1: return -lx
    if K==2: return -x*(lx-1.)
    return -0.5*x*x*(lx-1.5)


@jit(nopython=True,cache=True)
def _vertex_value_log(K,x,lx):
    """psi_K(x), K = 1 or 2, with lx = ln x given"""
    if K==1: return -lx
    return -x*(lx-1.)


@jit(nopython=True,cache=True)
def _point_value(m,K,x,y0,y1,y2,y3,harm):
    """psi_K over the m+1 nodes c + y_0..y_m, at x = c - z, by its Taylor
    series around c, sum_n psi_K^(m+n)(c) h_n(y)/(m+n)!, see
    _simplex_sources. With u = -1/x and w = y u, h_n(y) = h_n(w) u^(-n),
    and every term is u^(m+1-K) a_n h_n(w), with
      a_n = (j-K)!/j!                               for j = m+n >= K,
      a_n = -(-1)^(K-j-1) (ln x - H_(K-j-1))/((K-j-1)! j!)  for j < K,
    so the recursion runs on w and no power of u appears alone"""
    u = -1./x
    w0 = y0*u; w1 = y1*u; w2 = y2*u; w3 = y3*u
    H0 = 1.+0j; H1 = 1.+0j; H2 = 1.+0j; H3 = 1.+0j # h_n(w_0..w_i)
    lx = 0j
    if m<K: lx = np.log(x)
    acc = 0j
    small = 0
    for n in range(_nmax+1):
        if n>0:
            H0 = w0*H0
            H1 = H0+w1*H1
            H2 = H1+w2*H2
            H3 = H2+w3*H3
        if m==1: hn = H1
        elif m==2: hn = H2
        else: hn = H3
        j = m+n
        if j<K: # a logarithmic term, (K-j-1)! j! is at most 2 here
            r = K-j-1
            sg = -1. if r%2==0 else 1.
            fr = 1.
            for i in range(2,r+1): fr *= i
            fj = 1.
            for i in range(2,j+1): fj *= i
            acc += sg*(lx-harm[r])/(fr*fj)*hn
            continue
        a = 1.
        for i in range(j-K+1,j+1): a *= i # j!/(j-K)!
        term = hn/a
        acc += term
        if n>0: # stop after two negligible terms in a row
            if abs(term)<=1e-17*abs(acc):
                small += 1
                if small==2: break
            else: small = 0
    q = m+1-K # the power of u in front
    pre = 1.+0j
    if q>=0:
        for i in range(q): pre *= u
    else:
        for i in range(-q): pre *= -x
    return pre*acc


@jit(nopython=True,cache=True)
def _orders(r,kmax):
    """orders of a moment expansion of rate r at this frequency, so that
    r^k < 1e-18, at most kmax"""
    if r<1e-18: return 1
    if r<_rho: return min(kmax,int(np.ceil(np.log(1e-18)/np.log(r))))
    return kmax


@jit(nopython=True,cache=True)
def _kadd(acc,err,c,x):
    """acc[c] += x, with the rounding error added to err[c] (Knuth's
    two-sum, exact for each of the real and imaginary parts)"""
    a = acc[c]
    s = a+x
    b = s-a
    err[c] += (a-(s-b))+(x-b)
    acc[c] = s


@jit(nopython=True,parallel=True,cache=True)
def _evaluate_sources_jit(ev,cv,K,ep,tp,yp,ca,omegas,delta,nchunk,harm,
        fact,Cb,Mb,Rb,jp,Cs,Rs,Ms,Ls):
    """Sum of all sources at every frequency. The parallel loop runs over
    (frequency, chunk of sources), each task writing its own partial sum,
    so the result does not depend on the number of threads. A point
    source in the set jp, of center Cs and radius Rs, is evaluated one by
    one near its set, and through the moments (Ms,Ls) of the set where all
    of its sources converge at a rate _rho or better,
    (w - Cs)^2 + delta^2 >= (Rs/_rho)^2. The sums are compensated (acc
    and its rounding error err, see _kadd), two sources at a time (blk):
    the sources cancel, by a factor of 5e3 on the 3D layered model of the
    tests at nk = 24 and 8e3 at nk = 48, and the rounding of a plain
    running sum grew with their number."""
    nw = omegas.shape[0]
    nc = cv.shape[1]
    nvs = ev.shape[0]
    nps = ep.shape[0]
    nkb = Cs.shape[0]
    part = np.zeros((nw,nchunk,2,nc),dtype=np.complex128)
    for task in prange(nw*nchunk): # parallel loop over frequency and chunk
        iw = task//nchunk
        ic = task%nchunk
        z = omegas[iw] + 1j*delta
        acc = np.zeros(nc,dtype=np.complex128)
        err = np.zeros(nc,dtype=np.complex128)
        blk = np.zeros(nc,dtype=np.complex128) # two sources, plainly
        nb = 0
        t = np.zeros(5)
        hbuf = np.zeros(_dos_nterms+1)
        d = np.zeros(5,dtype=np.complex128)
        far = np.zeros(nkb,dtype=np.bool_) # one decision for both loops
        for jb in range(nkb):
            dw = omegas[iw]-Cs[jb]
            far[jb] = dw*dw+delta*delta>=(Rs[jb]/_rho)**2
        i0 = (ic*nvs)//nchunk
        i1 = ((ic+1)*nvs)//nchunk
        for i in range(i0,i1): # vertex sources
            f = _vertex_value(K,ev[i]-z)
            for c in range(nc): blk[c] += cv[i,c]*f
            nb += 1
            if nb==2:
                for c in range(nc):
                    _kadd(acc,err,c,blk[c])
                    blk[c] = 0.
                nb = 0
        i0 = (ic*nps)//nchunk
        i1 = ((ic+1)*nps)//nchunk
        for i in range(i0,i1): # point sources
            if far[jp[i]]: continue # in the moments of its set
            if tp[i]>=_direct: # the rule per frequency
                m = tp[i]-_direct
                x = ep[i]-omegas[iw]
                for v in range(m+1): t[v] = x+yp[i,v] # sorted, as yp
                f = _dd(K,t,m+1,-delta,harm,fact,hbuf,d)
            else: f = _point_value(tp[i],K,ep[i]-z,yp[i,0],yp[i,1],yp[i,2],
                    yp[i,3],harm)
            for c in range(nc): blk[c] += ca[i,c]*f
            nb += 1
            if nb==2:
                for c in range(nc):
                    _kadd(acc,err,c,blk[c])
                    blk[c] = 0.
                nb = 0
        for c in range(nc): _kadd(acc,err,c,blk[c])
        if ic==0: # the bins, sum_k M_k u^(k+1) by Horner
            for jb in range(Cb.shape[0]):
                u = 1./(z-Cb[jb])
                kk = _orders(Rb[jb]*abs(u),Mb.shape[1])
                for c in range(nc):
                    hs = 0j
                    for k in range(kk-1,-1,-1): hs = hs*u+Mb[jb,k,c]
                    _kadd(acc,err,c,hs*u)
            for jb in range(nkb): # the point sources of the far sets
                if not far[jb]: continue
                u = 1./(z-Cs[jb])
                kk = _orders(Rs[jb]*abs(u),Ms.shape[1])
                for c in range(nc):
                    hs = 0j
                    for k in range(kk-1,-1,-1): hs = hs*u+Ms[jb,k,c]
                    _kadd(acc,err,c,hs*u)
                # the psi_(K-i)(Cs - z) terms, all with one logarithm; a
                # point source has two nodes or more, so Ls[:,0] is zero
                x = Cs[jb]-z
                lx = np.log(x)
                for i in range(1,Ls.shape[1]):
                    f = _vertex_value_log(K-i,x,lx)
                    for c in range(nc): _kadd(acc,err,c,Ls[jb,i,c]*f)
        for c in range(nc):
            part[iw,ic,0,c] = acc[c]
            part[iw,ic,1,c] = err[c]
    out = np.zeros((nw,nc),dtype=np.complex128)
    for iw in range(nw):
        acc = np.zeros(nc,dtype=np.complex128)
        err = np.zeros(nc,dtype=np.complex128)
        for ic in range(nchunk):
            for c in range(nc):
                _kadd(acc,err,c,part[iw,ic,0,c])
                err[c] += part[iw,ic,1,c]
        for c in range(nc): out[iw,c] = acc[c]+err[c]
    return out


@jit(nopython=True,parallel=True,cache=True)
def _point_moments_jit(ep,tp,yp,ca,order,start,Cs,K,fact,Ms,Ls,Rs):
    """Moments (Ms,Ls)[s] of the point sources of set s, order[start[s]]
    to order[start[s+1]-1], around its center Cs[s], with the largest
    distance Rs[s] of their nodes from it, in parallel over the sets, each
    one in a fixed order, so that the result does not depend on the number
    of threads. A point source of type m or _direct+m stands for psi_K
    over the m+1 nodes ep + yp[0..m]"""
    for s in prange(Cs.shape[0]): # parallel loop over sets
        e = np.zeros(4)
        for q in range(start[s],start[s+1]):
            i = order[q]
            m = tp[i]%_direct
            for v in range(m+1): e[v] = ep[i]+yp[i,v]
            _add_node_moments(e,0,m,K,Cs[s],1.,ca[i],Ms,Ls,s,fact)
            Rs[s] = max(Rs[s],abs(e[0]-Cs[s]),abs(e[m]-Cs[s]))


def _merge(keys,*cs):
    """Merge the sources whose keys (one row of exactly equal numbers per
    source) coincide, adding their coefficients, and drop those whose
    coefficients all vanish. Returns the merged keys and coefficients.
    A 1D array of keys is one key per source, and comes back 1D."""
    if keys.ndim==1: # one key per source, as shgsimplex passes them
        out = _merge(keys[:,None],*cs)
        return (out[0][:,0],)+out[1:]
    if keys.shape[0]==0: return (keys,)+cs
    # a stable sort, so that equal keys keep their order on every machine:
    # the default sort is deterministic too, but numpy picks it by CPU and
    # version, and the order of equal keys moves the last digits
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


def simplex_sources(simplices,dE,W,trans,diag,delta,window,alpha=None,
        tol=None,budget=None):
    """All the sources of the simplex sum for the frequencies in
    window = (wmin,wmax), see _sources_jit: vertex sources (energies ev,
    coefficients cv of psi_K), point sources (energies ep, types tp,
    shapes yp, coefficients ca), energy bins of the far simplices (centers
    Cb, moments Mb, radii Rb), and the sets of the point sources, the set
    jp of every point source, with the centers Cs, the radii Rs and the
    moments Ms and Ls of the sets, see _point_moments_jit. tol is the
    largest rounding error of a simplex in its vertex form relative to its
    peak, and budget the fraction of it left to a simplex that is split,
    see _decompose"""
    if alpha is None: alpha = _alpha
    if tol is None: tol = _tol
    if budget is None: budget = _budget
    npts,npair = dE.shape
    ns = simplices.shape[0]
    nc = W.shape[2]
    tolc = float(alpha*delta)
    flag = np.zeros((ns,npair,2),dtype=np.int8) # the branches, see
    # _chunk_sources
    chunk = 256
    nchunk = (ns+chunk-1)//chunk
    args = (simplices,dE,W,trans,diag,tolc,_dos_fact,_dos_harm,
            float(window[0]),float(window[1]),float(delta),float(tol),
            float(budget*tol),chunk)
    # energy bins of the far simplices, over the range of +-dE
    emax = max(float(np.max(np.abs(dE))) if dE.size else 0.,1e-300)
    bins = np.array([-emax,2.*emax/_nbins])
    e0 = np.zeros(0); t0 = np.zeros(0,dtype=np.int64)
    y0 = np.zeros((0,4)); c0 = np.zeros((0,nc),dtype=np.complex128)
    count = _sources_jit(0,*args,np.zeros(nchunk,dtype=np.int64),flag,
            e0,t0,y0,c0,bins,_nbins) # count the points, decide the branches
    start = np.concatenate([[0],np.cumsum(count)[:-1]]).astype(np.int64)
    npoint = int(np.sum(count))
    ep = np.zeros(npoint); tp = np.zeros(npoint,dtype=np.int64)
    yp = np.zeros((npoint,4)); ca = np.zeros((npoint,nc),dtype=np.complex128)
    _sources_jit(1,*args,start,flag,ep,tp,yp,ca,bins,_nbins)
    # a fixed number of groups, so that the result does not depend on the
    # number of threads
    ng = max(1,min(16,ns//4096))
    M = np.zeros((ng,_nbins,_kmax,nc),dtype=np.complex128)
    R = np.zeros((ng,_nbins))
    _moments_jit(simplices,dE,W,trans,diag,_dos_fact,_dos_harm,
            float(window[0]),float(window[1]),float(delta),flag,bins,M,R)
    M = M.sum(axis=0)
    used = np.nonzero(np.any(M!=0.,axis=(1,2)))[0]
    Cb = bins[0]+(used+0.5)*bins[1]
    Mb = np.ascontiguousarray(M[used])
    Rb = R.max(axis=0)[used]
    A = np.zeros((2,npts,npair,nc),dtype=np.complex128) # forward, reverse
    _vertex_sources_jit(simplices,dE,W,trans,diag,tolc,_dos_fact,_dos_harm,
            flag,A[0],A[1])
    A = A.reshape(-1,nc)
    # e.g. the reverse term of a diagonal pair, or of a binned simplex
    nz = A.view(np.float64).any(axis=1)
    ev = np.concatenate([dE.ravel(),-dE.ravel()])[nz]
    ev,cv = _merge(ev[:,None],A[nz])
    del A
    # the point sources merge when their type, energy and shape coincide,
    # as in the copies of a spinful model and the edges and faces that
    # neighboring simplices share
    keys,ca = _merge(np.concatenate([tp[:,None].astype(np.float64),ep[:,None],
            yp],axis=1),ca)
    tp = keys[:,0].astype(np.int64)
    ep = keys[:,1].copy()
    yp = np.ascontiguousarray(keys[:,2:])
    # the sets of the point sources: the Taylor series and those evaluated
    # per frequency apart, since the latter are wider, and by energy bin
    sk,jp = np.unique((tp>=_direct)*_nbins+_bins_of(ep,bins,_nbins),
            return_inverse=True)
    jp = jp.ravel().astype(np.int64)
    Cs = bins[0]+(sk%_nbins+0.5)*bins[1]
    order = np.argsort(jp,kind="stable")
    start = np.searchsorted(jp[order],np.arange(len(sk)+1)).astype(np.int64)
    K = simplices.shape[1]-1
    Ms = np.zeros((len(sk),_kmax,nc),dtype=np.complex128)
    Ls = np.zeros((len(sk),K,nc),dtype=np.complex128)
    Rs = np.zeros(len(sk))
    _point_moments_jit(ep,tp,yp,ca,order,start,Cs,K,_dos_fact,Ms,Ls,Rs)
    return (ev[:,0].copy(),cv,ep,tp,yp,ca,Cb,Mb,Rb,jp,Cs,Rs,Ms,Ls)


def _sigma_sources(simplices,dE,W,trans,diag,omegas,delta,tol=None,
        budget=None):
    """The output of _sigma_simplices, from the sources; tol and budget
    are those of simplex_sources"""
    omegas = np.asarray(omegas,dtype=np.float64)
    window = (np.min(omegas),np.max(omegas)) if len(omegas) else (0.,0.)
    ev,cv,ep,tp,yp,ca,Cb,Mb,Rb,jp,Cs,Rs,Ms,Ls = simplex_sources(simplices,dE,
            W,trans,diag,delta,window,tol=tol,budget=budget)
    K = simplices.shape[1]-1 # 1, 2 or 3 in 1D, 2D or 3D
    nw = len(omegas)
    nsrc = len(ev)+len(ep)
    # enough tasks to keep up to _ntasks/4 threads busy when there are few
    # frequencies. A fixed count, not the number of threads, so that the
    # order of the partial sums, and so the result to the last bit, does
    # not depend on the number of threads
    nchunk = max(1,min(nsrc//1024,-(-_ntasks//max(nw,1))))
    return _evaluate_sources_jit(ev,np.ascontiguousarray(cv),K,ep,tp,yp,
            np.ascontiguousarray(ca),omegas,delta,nchunk,_dos_harm,_dos_fact,
            Cb,Mb,Rb,jp,Cs,Rs,Ms,Ls)


_kernels = {
    "sources": _sigma_sources, # sources built once, then summed per frequency
    "simplices": _sigma_simplices, # every simplex at every frequency
}


def _sigma_from_fields(dim,nk,pairs,dE,W,comps,trans,energies,delta,cellvol,
        kernel=None):
    """Assemble the 3x3 tensor from the stored pair fields on the mesh.
    kernel="simplices" is the direct sum over simplices, kept as the
    reference that the tests compare the default against; the default
    (kernel=None) is the source kernel, for segments, triangles and
    tetrahedra alike."""
    if kernel is None: kernel = "sources"
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
    # the velocities are 48 n^2 bytes per k-point, and the source build
    # below, where the memory peaks, needs only the pair fields
    del es,vs
    pairs,dE,W = _prune(pairs,dE,W)
    return _sigma_from_fields(h.dimensionality,nk,pairs,dE,W,comps,trans,
            energies,delta,cellvol)


def sigma_qtci(h,energies,nk,T,delta,intraband,interband,degeneracy_tol,
        tolerance=1e-6,window=None):
    """Optical conductivity tensor with the simplex integration on a
    2^R x 2^R mesh (nk rounded up to a power of two), where the smooth
    fields dE and W are not computed on every point but reconstructed from
    a quantics tensor cross interpolation of all of them at once (the
    field index is one more tensor site). The transition energies and the
    weights are two families of that interpolation, each divided by its
    largest value on the coarse scan first, so the tolerance is relative
    to the largest field of each family, and does not depend on the unit
    of length, see conductivity.optical_conductivity"""
    check_broadening(delta)
    if h.dimensionality!=2:
        raise NotImplementedError("integration='qtci' of the optical "
                "conductivity is only implemented in 2D, got dimensionality "
                +str(h.dimensionality)+"; use integration='triangle', in 1D "
                "a fine mesh is already cheap")
    from ..qtcitk.meshfields import qtci_stacked_fields,stacked_maxbonddim
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
    # coarse scan: fixes the pairs and components that matter
    # (qtci_stacked_fields repeats it on the same points to seed the pivots)
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
    # two families of fields, the transition energies and the weights: the
    # stacked TCI measures its tolerance against the largest field of all,
    # and W goes as a length squared while dE does not, so on a shared
    # scale a large lattice constant would loosen the tolerance of dE and a
    # small one that of W. qtci_stacked_fields divides each family by its
    # largest value on the coarse scan, so the result does not depend on
    # the unit of length
    isdE = np.array([kind=="dE" for (kind,p,c) in fields],dtype=np.bool_)
    families = [np.flatnonzero(isdE),np.flatnonzero(~isdE)]
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
    # one TCI of all the fields: it shares the pivots in k, so it needs
    # fewer diagonalizations than one TCI per field when the fields are few,
    # and its rank grows with their number, hence the larger rank cap
    # (future_development/qtci_optical_response.md)
    vals,ncalls,ranks = qtci_stacked_fields(evaluate,R,len(fields),
            tolerance=tolerance,ncoarse=nc0,maxbonddim=stacked_maxbonddim,
            families=families)
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
