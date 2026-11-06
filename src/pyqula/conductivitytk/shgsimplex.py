# Simplex (linear triangle) integration of the second-harmonic conductivity
# of shg.py, and its quantics tensor cross interpolation (QTCI) variant.
# The formula, the units and the sign conventions are those of shg.py
# (velocity gauge, Passos et al., PRB 97, 235446 (2018)); this module only
# replaces the k-sum, in the same way as kubosimplex.py does for the linear
# conductivity.
#
# Structure of the integrand
# --------------------------
# Writing out the density-matrix recursion of shg._sigma2_velocity_jit, with
# z = w + i delta, E_nm = E_n - E_m, f_nm = f_n - f_m, and symmetrizing in
# (b,c) the loop of three velocity elements
#   L_nml = (h^a_nm h^b_ml h^c_ln + h^a_nm h^c_ml h^b_ln)/2,
# the k-integrand is -(q^3/z^2) times the sum of five families of terms:
#
#   (i)   L_nml f_nl          / ((z - E_ln)(2z - E_mn))     l != n
#   (ii)  -L_nml f_lm         / ((z - E_ml)(2z - E_mn))     l != m
#   (iii) (1/2) h^a_nm h^bc_mn f_nm / (2z - E_mn)
#   (iv)  sym_bc[h^ab_nm h^c_mn] f_nm / (z - E_mn)
#   (v)   (1/2) h^abc_nn f_n                                (no resonance)
#
# The prefactor 1/z^2 does not depend on k. The weights in front (matrix
# element products times occupation differences) are smooth in k, and every
# band index loop is closed, so they are gauge invariant. They are averaged
# over the degenerate multiplets of each band index at every k, as in
# kubosimplex.py, and over the vertices of each simplex. The band energies
# are interpolated linearly inside each simplex, so every energy difference
# is linear there, and the resonant factors are integrated over the simplex
# exactly.
#
# Single resonances, (iii) and (iv): the simplex average of g(x) = 1/(z-x)
# with x linear in k is, by Hermite-Genocchi, d! psi_d[x_0..x_d], the d-th
# divided difference of the d-th antiderivative of g over the vertex values
# (psi_1 = -ln(x-z), psi_2 = -((x-z)ln(x-z) - (x-z)), as in kubosimplex.py).
# 1/(2z - B) = (1/2) g(B/2) is the same with the nodes B/2, so the 2w shell
# carries the broadening 2 delta, as in shg.py.
#
# Products, (i) and (ii): two resonances, at w (A = E_ln or E_ml) and at 2w
# (B = E_mn), with C = B/2,
#   1/((z-A)(2z-B)) = (1/2) g[A,C] = (1/2) int_0^1 ds g'((1-s)A + sC).
# Three cases:
#   m = n, so B = 0:       1/(2z) g(A)                         exact
#   l = m in (i) or l = n in (ii), so B = A (one pair of bands):
#                          (1/z) (g(A) - (1/2) g(A/2))         exact
#   three distinct bands:  the general product below
# The two exact cases have k-independent coefficients after partial
# fractions in z, so they reduce to single resonances. In a two-band model
# these are all the terms there are, so for two bands the method is the
# triangle method of the linear conductivity and nothing else is
# approximated. The cases are told apart per simplex from the vertex
# energies (B = 0 or B = A at every vertex, within the degeneracy
# tolerance), so a degenerate copy of a band (spin) falls into the exact
# case as well.
#
# For three distinct bands, a partial fraction in z gives the coefficient
# 1/(A - C) = 2/(2E_l - E_m - E_n), which diverges on the double-resonance
# line where the w and the 2w shells cross, so it cannot be interpolated as
# a smooth weight. Instead, the divided difference g[A,C] is written as the
# integral over s in [0,1], and the simplex times [0,1] is a prism on which
# x(k,s) = (1-s)A(k) + sC(k) is bilinear, not linear. The prism is cut into
# d+1 simplices of dimension d+1 (a staircase triangulation), x is
# interpolated linearly on each of them from its exact values at the 2(d+1)
# vertices of the prism, and g' is integrated exactly on each one, again by
# Hermite-Genocchi: the average of g' over a (d+1)-simplex is (d+1)!
# psi_d[d+2 nodes]. With the factor 1/2 and the volume 1/(d+1) of each
# piece the product averages to (d!/2) sum_pieces psi_d[nodes]. The bottom
# (s=0) and the top (s=1) faces are exact, so both resonant shells sit
# exactly where the linear interpolation of the bands puts them, and the
# double resonance is a confluent case of the divided difference and needs
# nothing special.
#
# What is approximated is the interior of the prism: the bilinear x differs
# from its piecewise linear interpolant by O(h |grad(A - C)|), with h the
# mesh spacing, and this does not grow as delta goes to zero. With a single
# staircase the errors of the simplices do not cancel and the sum
# converges at first order in h (measured on the three-band TMD model: the
# error halves when nk doubles). The staircase depends on the order of the
# vertices, and the average over all orderings (six for a triangle, sharing
# twelve distinct pieces) cancels that first-order part: the measured
# convergence is then of second order, as for the exact terms. Cutting the
# prism into slabs in s does not do this (measured, and dropped). An exact
# treatment needs the integral of a product of two resolvents of two
# independent linear functions over a triangle, a dilogarithm with a
# 1/Jacobian that degenerates when the two gradients are parallel; it is not
# built.
#
# What is not cured by any of this: the 1/z^2 prefactor. Its divergence
# cancels between the five families only after the k-integral (a sum rule
# that holds by integration by parts), which the uniform grid satisfies to
# exponential accuracy and the simplex sum only to O(h^2), so at small w the
# simplex result carries an O(h^2/|z|^2) error. Below the absorption edge
# the grid is the better choice anyway, since the integrand is smooth there.
import numpy as np
from numba import jit, prange

from . import kubo
from . import shg
from .kubosimplex import simplices
from .. import parallel


def check_broadening(delta):
    """The simplex integral needs a finite broadening: at delta = 0 the
    logarithms have their branch point on the real axis"""
    if not delta>0.:
        raise ValueError("the triangle and qtci integrations of the "
                "second-harmonic conductivity need a positive broadening "
                "delta, got delta="+str(delta))


# ------------------------------------------------------------ term catalogue
# A term is (kind, a0, a1, b0, b1): its resonant factor depends on the
# energy differences A = E_a1 - E_a0 and B = E_b1 - E_b0.
_PRODUCT = 0 # 1/((z-A)(2z-B))
_SINGLE = 1  # 1/(z-A)
_DOUBLE = 2  # 1/(2z-A)
_FLAT = 3    # no resonance


def _term_catalogue(nb):
    """Every term of the five families of the module comment for nb bands,
    merged when two of them have the same resonant factor. Returns the term
    keys (nterm,5) and the index maps from the band indices to the terms."""
    keys = {}
    def index(key):
        if key not in keys: keys[key] = len(keys)
        return keys[key]
    Ti = -np.ones((nb,nb,nb),dtype=np.int64)
    Tii = -np.ones((nb,nb,nb),dtype=np.int64)
    Tiii = -np.ones((nb,nb),dtype=np.int64)
    Tiv = -np.ones((nb,nb),dtype=np.int64)
    for n in range(nb):
        for m in range(nb):
            bpair = (n,m) if m!=n else (0,0) # B = 0 when m = n
            for l in range(nb):
                if l!=n: Ti[n,m,l] = index((_PRODUCT,n,l)+bpair)
                if l!=m: Tii[n,m,l] = index((_PRODUCT,l,m)+bpair)
            if m!=n:
                Tiii[n,m] = index((_DOUBLE,n,m,0,0))
                Tiv[n,m] = index((_SINGLE,n,m,0,0))
    Tv = index((_FLAT,0,0,0,0))
    terms = np.array(sorted(keys,key=lambda k: keys[k]),dtype=np.int64)
    return terms,Ti,Tii,Tiii,Tiv,Tv


@jit(nopython=True,parallel=True,cache=True)
def _weights_jit(es,f,h1,h2,h3,trip,Ti,Tii,Tiii,Tiv,Tv,nterm,tol):
    """Weights W[k,term,t] of the terms of the module comment on a batch of
    k-points, symmetrized in (b,c) and averaged over the degenerate
    multiplets of every band index"""
    nk = es.shape[0]
    nb = es.shape[1]
    nt = trip.shape[0]
    W = np.zeros((nk,nterm,nt),dtype=np.complex128)
    for ik in prange(nk): # loop over k-points
        start = np.zeros(nb,dtype=np.int64) # first band of each multiplet
        for i in range(1,nb):
            if es[ik,i]-es[ik,i-1]<tol: start[i] = start[i-1]
            else: start[i] = i
        stop = np.zeros(nb,dtype=np.int64) # one past its last band
        stop[nb-1] = nb
        for i in range(nb-2,-1,-1):
            if start[i+1]==start[i]: stop[i] = stop[i+1]
            else: stop[i] = i+1
        fk = f[ik]
        for t in range(nt): # loop over Cartesian triplets
            a = trip[t,0] ; b = trip[t,1] ; c = trip[t,2]
            for n in range(nb):
                W[ik,Tv,t] += 0.5*h3[ik,a,b,c,n,n]*fk[n]
                for m in range(nb):
                    sn = stop[n]-start[n] ; sm = stop[m]-start[m]
                    if m!=n: # pair terms (iii) and (iv)
                        acc3 = 0j ; acc4 = 0j
                        for i in range(start[n],stop[n]):
                            for j in range(start[m],stop[m]):
                                acc3 += h1[ik,a,i,j]*h2[ik,b,c,j,i]
                                acc4 += h2[ik,a,b,i,j]*h1[ik,c,j,i] \
                                      + h2[ik,a,c,i,j]*h1[ik,b,j,i]
                        norm = (fk[n]-fk[m])/(sn*sm)
                        W[ik,Tiii[n,m],t] += 0.5*acc3*norm
                        W[ik,Tiv[n,m],t] += 0.5*acc4*norm
                    for l in range(nb): # three-band loops (i) and (ii)
                        if l==n and l==m: continue # both vanish
                        sl = stop[l]-start[l]
                        acc = 0j
                        for i in range(start[n],stop[n]):
                            for j in range(start[m],stop[m]):
                                for p in range(start[l],stop[l]):
                                    acc += h1[ik,a,i,j]*(
                                        h1[ik,b,j,p]*h1[ik,c,p,i]
                                        + h1[ik,c,j,p]*h1[ik,b,p,i])
                        L = 0.5*acc/(sn*sm*sl)
                        if l!=n: W[ik,Ti[n,m,l],t] += L*(fk[n]-fk[l])
                        if l!=m: W[ik,Tii[n,m,l],t] -= L*(fk[l]-fk[m])
    return W


# ------------------------------------------------------------ divided differences

_ratio = 0.25 # a cluster of nodes within this fraction of its distance to
              # the pole is done by its Taylor series, which converges as
              # _ratio^k; a wider one by the recursion, which then divides
              # by a span at least _ratio times the distance to the pole


@jit(nopython=True,cache=True)
def _psi(d,y):
    """d-th antiderivative of g(x) = 1/(z-x), as a function of y = x - z"""
    if d==1: return -np.log(y)
    return -(y*np.log(y)-y)


@jit(nopython=True,cache=True)
def _taylor_coefficient(p,d,y):
    """psi_d^(p)(x)/p!, with y = x - z"""
    if p==0: return _psi(d,y)
    if p<d: return _psi(d-p,y) # p = 1 < d = 2: psi_2' = psi_1
    u = -1./y # 1/(z-x)
    # psi_d^(p) = (p-d)!/(z-x)^(p-d+1), divided by p!
    fac = 1.
    for q in range(p-d+1,p+1): fac *= q
    return u**(p-d+1)/fac


@jit(nopython=True,cache=True)
def _dd_taylor(x,i,j,d,z,hbuf):
    """psi_d[x_i..x_j] by its Taylor series around the mean of the nodes,
    f[x_i..x_j] = sum_k f^(r+k)(m)/(r+k)! h_k(x - m), with h_k the complete
    homogeneous symmetric polynomials of the deviations from the mean"""
    r = j-i
    m = 0.
    for q in range(i,j+1): m += x[q]
    m = m/(r+1)
    y = m-z
    span = x[j]-x[i]
    rho = span/abs(y)
    if rho<1e-12: K = 0
    else: K = min(hbuf.shape[0]-1,int(np.log(1e-17)/np.log(rho))+4)
    for k in range(K+1): hbuf[k] = 0.
    hbuf[0] = 1.
    for q in range(i,j+1): # h_k over one more variable
        dv = x[q]-m
        for k in range(1,K+1): hbuf[k] += dv*hbuf[k-1]
    out = 0j
    k0 = 0 # first k with r+k >= d, below it the coefficients are logarithms
    while r+k0<d and k0<=K:
        if k0!=1: out += _taylor_coefficient(r+k0,d,y)*hbuf[k0]
        k0 += 1
    if k0>K: return out
    u = -1./y # 1/(z-m)
    up = _taylor_coefficient(r+k0,d,y) # psi_d^(p)/p! at p = r+k0
    for k in range(k0,K+1):
        p = r+k
        if k!=1: out += up*hbuf[k] # h_1 = sum of the deviations = 0
        # psi_d^(p+1)/(p+1)! = psi_d^(p)/p! * u (p-d+1)/(p+1)
        up = up*u*(p-d+1)/(p+1)
    return out


@jit(nopython=True,cache=True)
def _small(x,i,j,z):
    """whether the Taylor series applies to the cluster of nodes x_i..x_j"""
    m = 0.
    for q in range(i,j+1): m += x[q]
    m = m/(j-i+1)
    return x[j]-x[i]<_ratio*abs(m-z)


@jit(nopython=True,cache=True)
def _dd_psi(x,n,d,z,F,hbuf):
    """psi_d[x_0..x_n] for real nodes x (sorted in place), stable for
    coinciding nodes; F is a work array of at least (n+1,n+1)"""
    for i in range(1,n+1): # insertion sort
        v = x[i] ; q = i-1
        while q>=0 and x[q]>v:
            x[q+1] = x[q] ; q -= 1
        x[q+1] = v
    if _small(x,0,n,z): return _dd_taylor(x,0,n,d,z,hbuf)
    for i in range(n+1): F[i,i] = _psi(d,x[i]-z)
    for r in range(1,n+1):
        for i in range(n+1-r):
            j = i+r
            if _small(x,i,j,z): F[i,j] = _dd_taylor(x,i,j,d,z,hbuf)
            else: F[i,j] = (F[i+1,j]-F[i,j-1])/(x[j]-x[i])
    return F[0,n]


@jit(nopython=True,cache=True)
def _single_average(e,nv,z,x,F,hbuf):
    """average of 1/(z - E) over a simplex with nv vertices (nv = d+1) and
    linear E taking the values e at the vertices: d! psi_d[e]"""
    d = nv-1
    for v in range(nv): x[v] = e[v]
    fac = 1. if d==1 else 2.
    return fac*_dd_psi(x,d,d,z,F,hbuf)


@jit(nopython=True,cache=True)
def _piece(x,ea,ec,i0,i1,i2,i3,na,z,F,hbuf):
    """psi_d[nodes] of one piece of the prism: the first na of the vertex
    indices i0..i3 are bottom vertices (values ea), the rest top vertices
    (values ec)"""
    idx = (i0,i1,i2,i3)
    n = 0
    for q in range(4):
        if idx[q]<0: break
        x[q] = ea[idx[q]] if q<na else ec[idx[q]]
        n += 1
    return _dd_psi(x,n-1,n-2,z,F,hbuf)


@jit(nopython=True,cache=True)
def _product_average(ea,ec,nv,z,x,F,hbuf):
    """average of (1/2) g[A,C] = 1/((z-A)(2z-2C)) over a simplex, with A and
    C linear: (d!/2) times the sum over the d+1 pieces of a triangulation
    of the prism simplex x [0,1] of psi_d[d+2 nodes], averaged over the
    staircase triangulations of all the vertex orderings (an ordering
    (p,q,r) of a triangle has the pieces {a_p,a_q,a_r,c_p}, {a_q,a_r,c_p,
    c_q} and {a_r,c_p,c_q,c_r}). One ordering alone leaves an error of
    first order in the mesh spacing that does not cancel between the
    simplices; the average over the orderings removes it (measured, see the
    module comment)"""
    if nv==2: # a segment: the square of (a0,a1,c0,c1), two orderings
        out = _piece(x,ea,ec,0,1,0,-1,2,z,F,hbuf) \
            + _piece(x,ea,ec,1,0,1,-1,1,z,F,hbuf) \
            + _piece(x,ea,ec,1,0,1,-1,2,z,F,hbuf) \
            + _piece(x,ea,ec,0,0,1,-1,1,z,F,hbuf)
        return 0.25*out # (1/2) times the average of the two orderings
    out = 0j
    for p in range(3): # pieces shared by two orderings each
        out += 2.*_piece(x,ea,ec,0,1,2,p,3,z,F,hbuf) # {a0,a1,a2,c_p}
        out += 2.*_piece(x,ea,ec,p,0,1,2,1,z,F,hbuf) # {a_p,c0,c1,c2}
    for p in range(3): # the middle piece of each of the six orderings
        for q in range(3):
            if q==p: continue
            r = 3-p-q
            out += _piece(x,ea,ec,q,r,p,q,2,z,F,hbuf) # {a_q,a_r,c_p,c_q}
    return out/6.


@jit(nopython=True,parallel=True,cache=True)
def _shg_simplex_jit(sims,es,Wt,terms,omegas,delta,tol):
    """Sum over simplices and terms of the vertex-averaged weight times the
    exact simplex average of the resonant factor, for every frequency and
    every Cartesian triplet. Not normalized: the caller multiplies by
    -q^3/z^2, the simplex volume fraction and 1/V_cell"""
    ns = sims.shape[0]
    nv = sims.shape[1]
    nterm = terms.shape[0]
    nt = Wt.shape[2]
    nw = omegas.shape[0]
    out = np.zeros((nw,nt),dtype=np.complex128)
    for iw in prange(nw): # parallel loop over frequencies
        z = omegas[iw] + 1j*delta
        acc = np.zeros(nt,dtype=np.complex128)
        x = np.zeros(4) # nodes of one divided difference
        F = np.zeros((4,4),dtype=np.complex128)
        hbuf = np.zeros(48)
        ea = np.zeros(nv) ; eb = np.zeros(nv) ; ec = np.zeros(nv)
        for it in range(ns): # loop over simplices
            for p in range(nterm): # loop over terms
                kind = terms[p,0]
                if kind==_FLAT:
                    for t in range(nt): acc[t] += Wt[it,p,t]
                    continue
                isb0 = True ; isba = True # B = 0, B = A at every vertex
                for v in range(nv):
                    kv = sims[it,v]
                    ea[v] = es[kv,terms[p,2]] - es[kv,terms[p,1]]
                    eb[v] = es[kv,terms[p,4]] - es[kv,terms[p,3]]
                    if abs(eb[v])>tol: isb0 = False
                    if abs(eb[v]-ea[v])>tol: isba = False
                if kind==_SINGLE: K = _single_average(ea,nv,z,x,F,hbuf)
                elif kind==_DOUBLE:
                    for v in range(nv): ec[v] = 0.5*ea[v]
                    K = 0.5*_single_average(ec,nv,z,x,F,hbuf)
                elif isb0: K = _single_average(ea,nv,z,x,F,hbuf)/(2.*z)
                elif isba:
                    for v in range(nv): ec[v] = 0.5*ea[v]
                    K = (_single_average(ea,nv,z,x,F,hbuf)
                         - 0.5*_single_average(ec,nv,z,x,F,hbuf))/z
                else:
                    for v in range(nv): ec[v] = 0.5*eb[v]
                    K = _product_average(ea,ec,nv,z,x,F,hbuf)
                for t in range(nt): acc[t] += Wt[it,p,t]*K
        out[iw] = acc
    return out


# ------------------------------------------------------------ fields

def _fields(tms,dirs,d,hkgen,ks,T,trip,cat,tol):
    """Band energies es[k,n] and term weights W[k,term,t] on a set of
    k-points, in batches"""
    terms,Ti,Tii,Tiii,Tiv,Tv = cat
    nb = Ti.shape[0]
    es = np.zeros((len(ks),nb))
    W = np.zeros((len(ks),len(terms),len(trip)),dtype=np.complex128)
    parallel.set_num_threads() # set the number of threads for numba
    for i0 in range(0,len(ks),shg._kbatch): # loop over batches of k-points
        kb = ks[i0:i0+shg._kbatch]
        e,h1,h2,h3 = shg._eigenbasis_derivatives(tms,dirs,d,hkgen,kb)
        f = kubo._fermi(e,T)
        es[i0:i0+len(kb)] = e
        W[i0:i0+len(kb)] = _weights_jit(e,f,h1,h2,h3,trip,Ti,Tii,Tiii,Tiv,
                Tv,len(terms),tol)
    return es,W


def _prune(terms,W,rtol=1e-12):
    """Drop the terms whose weight vanishes on every k-point, e.g. those
    between two occupied bands of an insulator"""
    wmax = np.max(np.abs(W),axis=(0,2))
    keep = wmax>rtol*max(np.max(wmax),1e-300)
    return terms[keep],W[:,keep]


def _sigma_from_fields(dim,nk,es,terms,W,energies,delta,cellvol,tol):
    """Simplex sum and normalization, from the fields on the mesh"""
    sims,frac = simplices(dim,nk)
    Wt = np.zeros((len(sims),)+W.shape[1:],dtype=np.complex128)
    for v in range(sims.shape[1]): Wt += W[sims[:,v]]
    Wt = Wt/sims.shape[1] # vertex average of the weights
    parallel.set_num_threads() # set the number of threads for numba
    s = _shg_simplex_jit(sims,np.ascontiguousarray(es),Wt,terms,energies,
            delta,tol)
    z = energies + 1j*delta
    return s*(-shg._charge**3)/(z*z)[:,None]*frac/cellvol


def _symmetrized_triplets(trip):
    """distinct (a,b,c) with b <= c among the requested components"""
    sym = []
    for (a,b,c) in trip:
        key = (a,min(b,c),max(b,c))
        if key not in sym: sym.append(key)
    return sym


def sigma_triangle(h,energies,nk,T,delta,trip,degeneracy_tol,tolerance=None):
    """Second-harmonic conductivity of the Cartesian triplets trip, with the
    simplex integration on the uniform nk mesh. Returns an array
    (len(energies),len(trip)), symmetrized in the last two indices"""
    check_broadening(delta)
    sym = _symmetrized_triplets(trip)
    tms,dirs,d,hkgen,cellvol,scale = shg._derivative_tensors(h)
    tol = degeneracy_tol*scale
    nb = tms.shape[1]
    cat = _term_catalogue(nb)
    ks = kubo._kmesh(h,nk)
    es,W = _fields(tms,dirs,d,hkgen,ks,T,np.array(sym,dtype=np.int64),cat,tol)
    terms,W = _prune(cat[0],W)
    s = _sigma_from_fields(h.dimensionality,nk,es,terms,W,energies,delta,
            cellvol,tol)
    return _unpack(s,sym,trip)


def _unpack(s,sym,trip):
    out = np.zeros((s.shape[0],len(trip)),dtype=np.complex128)
    for i,(a,b,c) in enumerate(trip):
        out[:,i] = s[:,sym.index((a,min(b,c),max(b,c)))]
    return out


def sigma_qtci(h,energies,nk,T,delta,trip,degeneracy_tol,tolerance=1e-6,
        info=None):
    """The same on a 2^R x 2^R mesh (nk rounded up to a power of two), with
    the band energies and the term weights reconstructed from a quantics
    tensor cross interpolation of each one, all of them sharing one cache of
    diagonalizations. info, if a dict, receives the number of
    diagonalizations and the ranks"""
    check_broadening(delta)
    if h.dimensionality!=2:
        raise NotImplementedError("integration='qtci' of the second-harmonic "
                "conductivity is only implemented in 2D, got dimensionality "
                +str(h.dimensionality)+"; use integration='triangle', in 1D "
                "a fine mesh is already cheap")
    from ..qutecipytk import crossinterpolate2
    from ..qutecipytk.tensortrain.core import tensortrain
    from ..qutecipytk.tensortrain.batcheval import BatchEvaluator
    import itertools
    sym = _symmetrized_triplets(trip)
    symarr = np.array(sym,dtype=np.int64)
    tms,dirs,d,hkgen,cellvol,scale = shg._derivative_tensors(h)
    tol = degeneracy_tol*scale
    nb = tms.shape[1]
    R = max(1,int(np.ceil(np.log2(nk)))) # bits per direction
    n = 2**R
    # coarse scan: fixes the terms that matter and seeds the pivots
    nc0 = 2**min(R,5)
    ks0 = kubo._kmesh(h,nc0)
    cat = _term_catalogue(nb)
    es0,W0 = _fields(tms,dirs,d,hkgen,ks0,T,symarr,cat,tol)
    keep = np.max(np.abs(W0),axis=(0,2))
    keep = keep>1e-12*max(np.max(keep),1e-300)
    terms = cat[0][keep]
    W0 = W0[:,keep]
    cache = {} # grid index (i,j) -> (es, W of the kept terms)
    def ensure(ij):
        new = [q for q in dict.fromkeys(ij) if q not in cache]
        if not new: return
        ks = np.zeros((len(new),3))
        ks[:,:2] = np.array(new,dtype=np.float64)/n
        e,W = _fields(tms,dirs,d,hkgen,ks,T,symarr,cat,tol)
        W = W[:,keep]
        for q,a,b in zip(new,e,W): cache[q] = (a,b)
    def value(kind,p,c,e,W):
        if kind=="E": return e[...,p]
        if kind=="re": return W[...,p,c].real
        return W[...,p,c].imag
    fields = [("E",p,None) for p in range(nb)]
    for p in range(len(terms)):
        for c in range(len(sym)):
            fields += [("re",p,c),("im",p,c)]
    weights = 2**np.arange(R-1,-1,-1)
    def decode(keys): # interleaved bits (x,y) per scale, most significant first
        q = np.array(keys,dtype=np.int64).reshape(-1,R,2)
        return list(zip((q[:,:,0]@weights).tolist(),(q[:,:,1]@weights).tolist()))
    def encode(i,j):
        out = []
        for s in range(R): out += [(i>>(R-1-s))&1,(j>>(R-1-s))&1]
        return out
    localdims = [2]*(2*R)
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
            combos = list(itertools.product(*[range(dd) for dd in cd]))
            keys = [tuple(l)+cc+tuple(r) for l in L for cc in combos for r in Rr]
            return self.evalmany(keys).reshape(len(L),*cd,len(Rr))
    npts = n*n
    es = np.zeros((npts,nb))
    W = np.zeros((npts,len(terms),len(sym)),dtype=np.complex128)
    xaxes = list(range(0,2*R,2)); yaxes = list(range(1,2*R,2))
    stride = n//nc0 # coarse point (i,j) is the fine point (i,j)*stride
    ranks = {}
    wscale = np.max(np.abs(W0),axis=(0,2)) # size of each term's weight
    done = [] # (coarse values, fine values) of the fields interpolated so far
    for (kind,p,c) in fields:
        f0 = value(kind,p,c,es0,W0)
        if np.max(np.abs(f0))==0.: continue # identically zero
        # a part that is only rounding noise (the imaginary part of a weight
        # that is real) would be interpolated to full rank: it is zero
        if kind!="E" and np.max(np.abs(f0))<1e-10*wscale[p]: continue
        full = None
        for (g0,gfull) in done: # equal or opposite to a field already done
            # on all the points of the coarse scan: the same function (a
            # symmetry of the model), interpolated once
            for sgn in (1.,-1.):
                if np.max(np.abs(f0-sgn*g0))<1e-12*np.max(np.abs(f0)):
                    full = sgn*gfull ; break
            if full is not None: break
        if full is None:
            i0 = int(np.argmax(np.abs(f0)))
            piv = [encode((i0//nc0)*stride,(i0%nc0)*stride)]
            tci,rk,errors = crossinterpolate2(np.float64,Field(kind,p,c),
                    localdims,initialpivots=piv,tolerance=tolerance,
                    maxbonddim=400,pivotsearch="full")
            ranks[(kind,p,c)] = max(rk) if len(rk) else 0
            full = tensortrain(tci).fulltensor()
            full = np.transpose(full,xaxes+yaxes).reshape(npts) # row-major
            done.append((f0,full))
        if kind=="E": es[:,p] = full
        elif kind=="re": W[:,p,c] += full
        else: W[:,p,c] += 1j*full
    if isinstance(info,dict):
        info["diagonalizations"] = len(cache)
        info["ranks"] = ranks
    s = _sigma_from_fields(2,n,es,terms,W,energies,delta,cellvol,tol)
    return _unpack(s,sym,trip)
