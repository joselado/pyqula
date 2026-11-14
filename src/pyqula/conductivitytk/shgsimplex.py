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

_ratio = 0.25 # the direct sum does a cluster of nodes of span s at a
              # distance y of its mean from the pole by its Taylor series,
              # which converges as (s/y)^k, when s < _ratio y, and a wider
              # one by the recursion, which then divides by a span of at
              # least _ratio y, per frequency. The source kernel compares s
              # with _ratio delta instead, see there


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
    out = u # u^(p-d+1) by products, a complex power is a log and an exp
    for q in range(p-d): out = out*u
    return out/fac


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
    rho = span/abs(y) # below _ratio, see _small
    # equal nodes still take the first order: their rounded mean can be off
    # by eps abs(m), (0.7 + 0.7 + 0.7)/3 is not 0.7, see h_1 below
    if rho<1e-12: K = 1
    else: K = min(hbuf.shape[0]-1,int(np.log(1e-17)/np.log(rho))+4)
    for k in range(K+1): hbuf[k] = 0.
    hbuf[0] = 1.
    for q in range(i,j+1): # h_k over one more variable
        dv = x[q]-m
        for k in range(1,K+1): hbuf[k] += dv*hbuf[k-1]
    out = 0j
    k0 = 0 # first k with r+k >= d, below it the coefficients are logarithms
    while r+k0<d and k0<=K:
        out += _taylor_coefficient(r+k0,d,y)*hbuf[k0]
        k0 += 1
    if k0>K: return out
    u = -1./y # 1/(z-m)
    up = _taylor_coefficient(r+k0,d,y) # psi_d^(p)/p! at p = r+k0
    for k in range(k0,K+1):
        p = r+k
        # h_1, the sum of the deviations, vanishes for the exact mean but not
        # for the rounded m, which is off by eps abs(m): on resonance, where
        # abs(y) is about delta, skipping it lost eps abs(m)/delta (2e-10 of
        # a triangle at m = 0.5 and delta = 1e-6), so it is kept
        out += up*hbuf[k]
        # psi_d^(p+1)/(p+1)! = psi_d^(p)/p! * u (p-d+1)/(p+1)
        up = up*u*(p-d+1)/(p+1)
    return out


@jit(nopython=True,cache=True)
def _small(x,i,j,z):
    """whether the Taylor series applies to the cluster of nodes x_i..x_j:
    its span is below _ratio times the distance of its mean m to the pole.
    The rule depends on w. A rule on delta alone, span < _ratio delta,
    sends a nearly flat cluster far from resonance to the recursion, which
    then loses eps (abs(m - z)/span)^r in a divided difference of order r
    (4e-4 of a triangle of spread delta at abs(m - z) = 0.5 and delta =
    1e-6, and more than the whole value of a prism piece)"""
    m = 0.
    for q in range(i,j+1): m += x[q]
    m = m/(j-i+1)
    s = x[j]-x[i] # compared in squares, abs(m - z) needs a square root
    return s*s<_ratio*_ratio*((m-z.real)**2+z.imag*z.imag)


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


# ------------------------------------------------------------ source kernel
#
# The source kernel fixes the branches once for all frequencies: a cluster
# of nodes goes to its Taylor series when its span is below _ratio delta,
# whatever w (the distance of a real node to the pole is at least delta, so
# the series converges at every w), and the exact cases B = 0 and B = A are
# decided from the vertex energies. Every divided difference is linear in
# the values it combines, so the whole sum over simplices and terms is a
# sum of fixed coefficients times a few functions of y = e - z at fixed
# energies e:
#   psi_d(y) at a node (psi_2 on triangles, psi_1 on segments), where the
#     nodes are the energy differences E_j - E_i at the vertices or their
#     halves (the 2w shell and the top face of the prism);
#   psi_d[a,b] for a pair of nodes closer than _ratio*delta;
#   a Taylor series at the mean m of a larger cluster of close nodes, in
#     psi_2, psi_1 and the powers of v = delta/(z - m).
# We build the coefficients once. For one divided difference this is the
# recursion of _dd_psi run backwards: the coefficient of the top entry
# F[0,n] is pushed down the table to the entries it is made of, and it
# ends either on a node, F[i,i], or on a cluster that _dd_psi does by its
# Taylor series. The node coefficients are indexed by k-point and by node
# type, so the simplices that share a vertex share one source, and sources
# with exactly the same energy are merged. The terms of the cases B = 0 and
# B = A carry a factor 1/z, kept as a second family of coefficients. Then
# each frequency costs one sum over the sources. This is the same sum as
# _shg_simplex_jit, term by term, with the additions in another order, the
# pairs below in closed form, and the branch of a cluster decided on delta
# and not per frequency.
#
# That last difference costs precision, and it is not repaired here. A
# cluster wider than _ratio delta goes to the recursion at every w, and far
# from its mean, at a distance abs(x) much larger than its span s, the
# recursion loses eps (abs(x)/s)^r in a divided difference of order r. At
# the delta of an optical spectrum this is small: on the three-band model
# of the tests at nk = 64 and delta = 1e-3 the sum is within 5e-11 of the
# direct sum, which decides per frequency. It matters for nearly flat
# bands at a small delta: three bands of spread delta on a 2x2 mesh at
# delta = 1e-6 lose 9e-10 to 5e-9 of the largest value, which is 4e-4 to
# all of the value off resonance. Summing the
# simplices whose fixed branches cancel too much at every frequency cures
# that, but it cost 2.6 times the old time at delta = 1e-3 for no gain
# there, so it was not kept (future_development/qtci_shg.md). For a small
# delta or flat bands, kernel="simplices" of _sigma_from_fields is exact.
#
# Most clusters are pairs, and their number grows as nk^3 at fixed delta
# (the triangles times the chance that two nodes are closer than
# _ratio*delta), so a pair is not stored as its Taylor series (some 25
# coefficients) but by its two nodes, and evaluated in closed form: with
# y = a - z, h = b - a and s = h/(2y + h),
#   psi_2[a,b] = -(ln y + s + (1 + s) S'),  psi_1[a,b] = -2 S/(2y + h),
#   S = atanh(s)/s = 1 + S',  S' = s^2/3 + s^4/5 + ...,
# where abs(s) < 0.15, so S' converges fast and nothing cancels.

_maxcluster = 80 # clusters of one simplex and one term: at most six per
                 # divided difference, twelve divided differences
_maxbasis = 48 # psi_2, psi_1 and the powers of v of one Taylor cluster


@jit(nopython=True,cache=True)
def _cluster_sources(x,i,j,d,delta,c,fam,cm,cf,cl,cc,ncl,hbuf):
    """c times psi_d[x_i..x_j] for a cluster of close nodes. A cluster q
    keeps its energy cm[q,0], its multiplier cm[q,1], its order cm[q,2] and
    its nodes cm[q,3:], its family cf[q] and its number of basis functions
    cl[q], with cl[q] = 0 for a pair. A Taylor cluster (three nodes or
    more) keeps the coefficients cc[q] of the basis (psi_2, psi_1, v, v^2,
    ...) of the Taylor series of _dd_taylor, for a unit multiplier, with
    v = delta/(z - m); the deviations are scaled by delta, so no large power
    of 1/delta is formed. The same cluster appears in several pieces of a
    prism; it is then stored once, with the multipliers added. Returns the
    new number of clusters"""
    r = j-i
    for q in range(ncl): # the same nodes already met in this simplex
        if cf[q]!=fam or cm[q,2]!=r: continue
        same = True
        for t in range(r+1):
            if cm[q,3+t]!=x[i+t]: same = False
        if same:
            cm[q,1] += c
            return ncl
    cm[ncl,1] = c
    cm[ncl,2] = r
    for t in range(r+1): cm[ncl,3+t] = x[i+t]
    cf[ncl] = fam
    if r==1: # a pair, evaluated in closed form
        cm[ncl,0] = x[i]
        cl[ncl] = 0
        return ncl+1
    m = 0.
    for q in range(i,j+1): m += x[q]
    m = m/(r+1)
    span = x[j]-x[i]
    rho = span/delta # abs(m - z) >= delta: the series converges at every w
    if rho<1e-12: K = 1 # equal nodes take the first order, see _dd_taylor
    else: K = min(hbuf.shape[0]-1,int(np.log(1e-17)/np.log(rho))+4)
    for k in range(K+1): hbuf[k] = 0.
    hbuf[0] = 1.
    for q in range(i,j+1): # h_k of the scaled deviations
        dv = (x[q]-m)/delta
        for k in range(1,K+1): hbuf[k] += dv*hbuf[k-1]
    L = max(2,r+K-d+3)
    for b in range(L): cc[ncl,b] = 0.
    scale = 1./delta**(r-d+1)
    for k in range(K+1):
        # k = 1 is kept: h_1 vanishes only for the exact mean, see _dd_taylor
        p = r+k
        if p<d: # p = 1 < d = 2 (not reached for r >= 2)
            cc[ncl,1] += hbuf[k]
            continue
        fac = 1. # p!/(p-d)!
        for q in range(p-d+1,p+1): fac *= q
        cc[ncl,p-d+2] += scale*hbuf[k]/fac # coefficient of v^(p-d+1)
    cm[ncl,0] = m
    cl[ncl] = L
    return ncl+1


@jit(nopython=True,cache=True)
def _dd_sources(x,lab,n,d,delta,s,fam,cn,cm,cf,cl,cc,ncl,C,hbuf):
    """s times psi_d[x_0..x_n] as sources, with the branches on delta of
    this kernel (_dd_psi decides per frequency instead):
    the coefficient of psi_d at node q is added to cn[lab[q],fam], the
    clusters are appended from index ncl on. Returns the new ncl"""
    for i in range(1,n+1): # insertion sort, carrying the labels along
        v = x[i] ; lv = lab[i] ; q = i-1
        while q>=0 and x[q]>v:
            x[q+1] = x[q] ; lab[q+1] = lab[q] ; q -= 1
        x[q+1] = v ; lab[q+1] = lv
    for i in range(n+1):
        for j in range(n+1): C[i,j] = 0.
    C[0,n] = s
    for r in range(n,-1,-1): # from the top of the table down
        for i in range(n+1-r):
            j = i+r
            c = C[i,j]
            if c==0.: continue
            if r==0: cn[lab[i],fam] += c # a node
            elif x[j]-x[i]<_ratio*delta: # a cluster, the rule on delta
                ncl = _cluster_sources(x,i,j,d,delta,c,fam,cm,cf,cl,cc,ncl,
                        hbuf)
            else: # F[i,j] = (F[i+1,j]-F[i,j-1])/(x_j-x_i)
                g = c/(x[j]-x[i])
                C[i+1,j] += g
                C[i,j-1] -= g
    return ncl


@jit(nopython=True,cache=True)
def _piece_sources(ea,ec,i0,i1,i2,i3,na,nv,delta,s,cn,cm,cf,cl,cc,ncl,
        x,lab,C,hbuf):
    """sources of s times one piece of the prism, see _piece. Label v is
    the bottom node of vertex v, label nv+v its top node"""
    idx = (i0,i1,i2,i3)
    n = 0
    for q in range(4):
        if idx[q]<0: break
        if q<na:
            x[q] = ea[idx[q]] ; lab[q] = idx[q]
        else:
            x[q] = ec[idx[q]] ; lab[q] = nv+idx[q]
        n += 1
    return _dd_sources(x,lab,n-1,n-2,delta,s,0,cn,cm,cf,cl,cc,ncl,C,hbuf)


@jit(nopython=True,cache=True)
def _term_sources(kind,ea,eb,ec,nv,tol,delta,cn,cm,cf,cl,cc,x,lab,C,hbuf):
    """Sources of one simplex and one term, with the branches of
    _shg_simplex_jit, for a unit weight. Node coefficients go to
    cn[label,family] (family 0: plain, family 1: times 1/z), labels v and
    nv+v are the full and the halved node of vertex v. Returns the number
    of clusters and whether the halved nodes are those of B (True) or of
    A (False)"""
    for v in range(2*nv):
        cn[v,0] = 0. ; cn[v,1] = 0.
    d = nv-1
    fac = 1. if d==1 else 2.
    isb0 = True ; isba = True # B = 0, B = A at every vertex
    for v in range(nv):
        if abs(eb[v])>tol: isb0 = False
        if abs(eb[v]-ea[v])>tol: isba = False
    ncl = 0
    if kind==_SINGLE: # fac psi_d[A]
        for v in range(nv):
            x[v] = ea[v] ; lab[v] = v
        ncl = _dd_sources(x,lab,d,d,delta,fac,0,cn,cm,cf,cl,cc,ncl,C,hbuf)
        return ncl,False
    if kind==_DOUBLE: # (1/2) fac psi_d[A/2]
        for v in range(nv):
            x[v] = 0.5*ea[v] ; lab[v] = nv+v
        ncl = _dd_sources(x,lab,d,d,delta,0.5*fac,0,cn,cm,cf,cl,cc,ncl,C,
                hbuf)
        return ncl,False
    if isb0 or isba: # (1/2z) fac psi_d[A], or (1/z) fac (psi_d[A] - ...
        s = 0.5*fac if isb0 else fac
        for v in range(nv):
            x[v] = ea[v] ; lab[v] = v
        ncl = _dd_sources(x,lab,d,d,delta,s,1,cn,cm,cf,cl,cc,ncl,C,hbuf)
        if not isb0: # ... - (1/2) psi_d[A/2])
            for v in range(nv):
                x[v] = 0.5*ea[v] ; lab[v] = nv+v
            ncl = _dd_sources(x,lab,d,d,delta,-0.5*fac,1,cn,cm,cf,cl,cc,ncl,
                    C,hbuf)
        return ncl,False
    for v in range(nv): ec[v] = 0.5*eb[v]
    if nv==2: # the pieces of _product_average
        ncl = _piece_sources(ea,ec,0,1,0,-1,2,nv,delta,0.25,cn,cm,cf,cl,cc,
                ncl,x,lab,C,hbuf)
        ncl = _piece_sources(ea,ec,1,0,1,-1,1,nv,delta,0.25,cn,cm,cf,cl,cc,
                ncl,x,lab,C,hbuf)
        ncl = _piece_sources(ea,ec,1,0,1,-1,2,nv,delta,0.25,cn,cm,cf,cl,cc,
                ncl,x,lab,C,hbuf)
        ncl = _piece_sources(ea,ec,0,0,1,-1,1,nv,delta,0.25,cn,cm,cf,cl,cc,
                ncl,x,lab,C,hbuf)
        return ncl,True
    for p in range(3):
        ncl = _piece_sources(ea,ec,0,1,2,p,3,nv,delta,2./6.,cn,cm,cf,cl,cc,
                ncl,x,lab,C,hbuf)
        ncl = _piece_sources(ea,ec,p,0,1,2,1,nv,delta,2./6.,cn,cm,cf,cl,cc,
                ncl,x,lab,C,hbuf)
    for p in range(3):
        for q in range(3):
            if q==p: continue
            r = 3-p-q
            ncl = _piece_sources(ea,ec,q,r,p,q,2,nv,delta,1./6.,cn,cm,cf,
                    cl,cc,ncl,x,lab,C,hbuf)
    return ncl,True


@jit(nopython=True,cache=True)
def _row_sources(write,ir,rows,sims,es,W,terms,tslot,tol,delta,Av,
        qa,qh,qf,qw,ep,pf,pstart,plen,cp,pw,oq,ot,oc):
    """Sources of all the simplices of one row of the mesh and all the
    terms. With write=False it only counts the pairs, the Taylor clusters
    and their coefficients; with write=True it adds the node coefficients
    to Av, and writes the pairs (nodes qa, qa+qh, family qf, weights qw)
    from index oq on and the Taylor clusters (energy ep, family pf,
    coefficients cp[pstart:pstart+plen], weights pw) from index ot and
    coefficient oc on. Returns the three counts"""
    nv = sims.shape[1]
    nterm = terms.shape[0]
    nt = W.shape[2]
    cn = np.zeros((2*nv,2))
    cm = np.zeros((_maxcluster,7)) ; cf = np.zeros(_maxcluster,dtype=np.int64)
    cl = np.zeros(_maxcluster,dtype=np.int64)
    cc = np.zeros((_maxcluster,_maxbasis))
    x = np.zeros(4) ; lab = np.zeros(4,dtype=np.int64)
    C = np.zeros((4,4)) ; hbuf = np.zeros(_maxbasis)
    ea = np.zeros(nv) ; eb = np.zeros(nv) ; ec = np.zeros(nv)
    wbar = np.zeros(nt,dtype=np.complex128)
    nq = 0 ; ntay = 0 ; nc = 0
    for s in range(rows.shape[1]): # loop over the simplices of the row
        it = rows[ir,s]
        for p in range(nterm): # loop over terms
            kind = terms[p,0]
            if kind==_FLAT: continue # a constant, added by the caller
            for v in range(nv):
                kv = sims[it,v]
                ea[v] = es[kv,terms[p,2]] - es[kv,terms[p,1]]
                eb[v] = es[kv,terms[p,4]] - es[kv,terms[p,3]]
            ncl,useb = _term_sources(kind,ea,eb,ec,nv,tol,delta,cn,cm,cf,cl,
                    cc,x,lab,C,hbuf)
            if not write:
                for c in range(ncl):
                    if cl[c]==0: nq += 1
                    else:
                        ntay += 1 ; nc += cl[c]
                continue
            for t in range(nt): wbar[t] = 0.
            for v in range(nv):
                for t in range(nt): wbar[t] += W[sims[it,v],p,t]
            for t in range(nt): wbar[t] = wbar[t]/nv
            hs = tslot[p,2] if useb else tslot[p,1]
            for v in range(nv):
                kv = sims[it,v]
                for fam in range(2):
                    if cn[v,fam]!=0.:
                        for t in range(nt):
                            Av[kv,tslot[p,0],fam,t] += cn[v,fam]*wbar[t]
                    if cn[nv+v,fam]!=0.:
                        for t in range(nt):
                            Av[kv,hs,fam,t] += cn[nv+v,fam]*wbar[t]
            for c in range(ncl):
                if cl[c]==0: # a pair
                    g = oq+nq
                    qa[g] = cm[c,3] ; qh[g] = cm[c,4]-cm[c,3] ; qf[g] = cf[c]
                    for t in range(nt): qw[g,t] = cm[c,1]*wbar[t]
                    nq += 1
                else: # a Taylor cluster
                    g = ot+ntay
                    ep[g] = cm[c,0] ; pf[g] = cf[c]
                    pstart[g] = oc+nc ; plen[g] = cl[c]
                    for b in range(cl[c]): cp[oc+nc+b] = cc[c,b]
                    for t in range(nt): pw[g,t] = cm[c,1]*wbar[t]
                    ntay += 1 ; nc += cl[c]
    return nq,ntay,nc


@jit(nopython=True,parallel=True,cache=True)
def _count_sources_jit(rows,sims,es,W,terms,tslot,tol,delta):
    """number of pairs, of Taylor clusters and of their coefficients, per
    row of the mesh"""
    nrow = rows.shape[0]
    counts = np.zeros((nrow,3),dtype=np.int64)
    A0 = np.zeros((1,1,2,1),dtype=np.complex128)
    e0 = np.zeros(1) ; i0 = np.zeros(1,dtype=np.int64)
    c0 = np.zeros((1,1),dtype=np.complex128)
    for ir in prange(nrow): # parallel loop over rows
        a,b,c = _row_sources(False,ir,rows,sims,es,W,terms,tslot,tol,delta,
                A0,e0,e0,i0,c0,e0,i0,i0,i0,e0,c0,0,0,0)
        counts[ir,0] = a ; counts[ir,1] = b ; counts[ir,2] = c
    return counts


@jit(nopython=True,parallel=True,cache=True)
def _write_sources_jit(rowids,rows,sims,es,W,terms,tslot,tol,delta,Av,
        qa,qh,qf,qw,ep,pf,pstart,plen,cp,pw,offsets):
    """write the sources of the rows rowids, which share no vertex, so that
    two threads never add to the same entry of Av"""
    for q in prange(rowids.shape[0]): # parallel loop over rows
        ir = rowids[q]
        _row_sources(True,ir,rows,sims,es,W,terms,tslot,tol,delta,Av,
                qa,qh,qf,qw,ep,pf,pstart,plen,cp,pw,
                offsets[ir,0],offsets[ir,1],offsets[ir,2])


def _mesh_rows(dim,nk):
    """The simplices of simplices(dim,nk) grouped by row of the mesh (all
    the simplices of row i have their vertices in the rows i and i+1), and
    three sets of rows such that two rows of one set share no vertex"""
    i = np.arange(nk)
    if dim==1: rows = i[:,None]
    else:
        t1 = i[:,None]*nk + i[None,:]
        rows = np.concatenate([t1,nk*nk+t1],axis=1)
    last = nk-1 if nk%2==1 else -1 # in an odd ring, row nk-1 touches row 0
    phases = [i[(i%2==0) & (i!=last)],i[i%2==1],i[i==last]]
    return np.ascontiguousarray(rows,dtype=np.int64),phases


def _node_slots(terms):
    """The node types, an ordered band pair (i,j) and whether the energy
    E_j - E_i is halved, and for each term the slots of its full A node,
    its halved A node and its halved B node (-1 where there is none)"""
    slots = {}
    def slot(i,j,half):
        if (i,j,half) not in slots: slots[(i,j,half)] = len(slots)
        return slots[(i,j,half)]
    tslot = -np.ones((len(terms),3),dtype=np.int64)
    for p,(kind,a0,a1,b0,b1) in enumerate(terms):
        if kind==_FLAT: continue
        tslot[p,0] = slot(a0,a1,0)
        if kind==_SINGLE: continue
        tslot[p,1] = slot(a0,a1,1)
        if kind==_PRODUCT and (b0,b1)!=(0,0): tslot[p,2] = slot(b0,b1,1)
    return np.array(list(slots),dtype=np.int64).reshape(-1,3),tslot


def shg_sources(dim,nk,es,terms,W,delta,tol):
    """All the sources of the simplex sum, see the comment above, as a
    dict: nodes (energies ev, coefficients cv[i,family,t]), pairs (nodes qa
    and qa+qh, families qf, weights qw), Taylor clusters (energies ep,
    families pf, coefficients cp[pstart:pstart+plen], weights pw) and the
    constant of the terms without resonance"""
    from .kubosimplex import _merge
    sims,_ = simplices(dim,nk)
    rows,phases = _mesh_rows(dim,nk)
    npts = es.shape[0]
    nt = W.shape[2]
    slots,tslot = _node_slots(terms)
    es = np.ascontiguousarray(es) ; W = np.ascontiguousarray(W)
    const = np.zeros(nt,dtype=np.complex128)
    for p in np.nonzero(terms[:,0]==_FLAT)[0]: # sum of the vertex averages
        wt = np.zeros((len(sims),nt),dtype=np.complex128)
        for v in range(sims.shape[1]): wt += W[sims[:,v],p]
        const += np.sum(wt/sims.shape[1],axis=0)
    parallel.set_num_threads() # set the number of threads for numba
    counts = _count_sources_jit(rows,sims,es,W,terms,tslot,tol,delta)
    offsets = np.zeros((len(rows),3),dtype=np.int64)
    offsets[1:] = np.cumsum(counts,axis=0)[:-1]
    nq,ntay,nc = np.sum(counts,axis=0)
    Av = np.zeros((npts,len(slots),2,nt),dtype=np.complex128)
    s = dict(qa=np.zeros(nq),qh=np.zeros(nq),qf=np.zeros(nq,dtype=np.int64),
            qw=np.zeros((nq,nt),dtype=np.complex128),
            ep=np.zeros(ntay),pf=np.zeros(ntay,dtype=np.int64),
            pstart=np.zeros(ntay,dtype=np.int64),
            plen=np.zeros(ntay,dtype=np.int64),cp=np.zeros(nc),
            pw=np.zeros((ntay,nt),dtype=np.complex128),const=const)
    for rowids in phases:
        if len(rowids)==0: continue
        _write_sources_jit(rowids.astype(np.int64),rows,sims,es,W,terms,
                tslot,tol,delta,Av,s["qa"],s["qh"],s["qf"],s["qw"],s["ep"],
                s["pf"],s["pstart"],s["plen"],s["cp"],s["pw"],offsets)
    E = np.zeros((npts,len(slots))) # node energies, as in _shg_simplex_jit
    for q,(i,j,half) in enumerate(slots):
        E[:,q] = es[:,j]-es[:,i]
        if half: E[:,q] = 0.5*E[:,q]
    Av = Av.reshape(npts*len(slots),2*nt)
    nz = np.any(Av!=0.,axis=1)
    ev,cv = _merge(E.ravel()[nz],Av[nz])
    del Av
    s["ev"] = ev ; s["cv"] = np.ascontiguousarray(cv.reshape(-1,2,nt))
    return s


@jit(nopython=True,parallel=True,cache=True)
def _evaluate_sources_jit(ev,cv,vkind,qa,qh,qf,qw,ep,pf,pstart,plen,cp,pw,
        const,omegas,delta,nchunk):
    """Sum of all sources at every frequency, the output of
    _shg_simplex_jit. The parallel loop runs over (frequency, chunk of
    sources), each task writing its own partial sum, so the result does not
    depend on the number of threads"""
    nw = omegas.shape[0]
    nt = cv.shape[2]
    nvs = ev.shape[0]
    nqs = qa.shape[0]
    nps = ep.shape[0]
    part = np.zeros((nw,nchunk,2,nt),dtype=np.complex128)
    for task in prange(nw*nchunk): # parallel loop over frequency and chunk
        iw = task//nchunk
        ic = task%nchunk
        z = omegas[iw] + 1j*delta
        acc = np.zeros((2,nt),dtype=np.complex128)
        i0 = (ic*nvs)//nchunk
        i1 = ((ic+1)*nvs)//nchunk
        for i in range(i0,i1): # nodes
            y = ev[i]-z
            if vkind==2: f = -(y*np.log(y)-y) # psi_2
            else: f = -np.log(y) # psi_1
            for fam in range(2):
                for t in range(nt): acc[fam,t] += cv[i,fam,t]*f
        i0 = (ic*nqs)//nchunk
        i1 = ((ic+1)*nqs)//nchunk
        for i in range(i0,i1): # pairs, see the comment above
            y = qa[i]-z
            u = 2.*y+qh[i]
            sv = qh[i]/u
            s2 = sv*sv
            sp = 0j # S - 1
            tk = s2
            for k in range(1,40):
                sp += tk/(2*k+1)
                if abs(tk)<1e-17: break
                tk = tk*s2
            if vkind==2: f = -(np.log(y)+sv+(1.+sv)*sp) # psi_2[a,b]
            else: f = -2.*(1.+sp)/u # psi_1[a,b]
            fam = qf[i]
            for t in range(nt): acc[fam,t] += qw[i,t]*f
        i0 = (ic*nps)//nchunk
        i1 = ((ic+1)*nps)//nchunk
        for i in range(i0,i1): # Taylor clusters
            y = ep[i]-z
            ly = np.log(y)
            v = -delta/y # delta/(z - m)
            o = pstart[i]
            f = cp[o]*(-(y*ly-y)) + cp[o+1]*(-ly) # psi_2 and psi_1
            vq = v
            for b in range(2,plen[i]):
                f += cp[o+b]*vq
                vq = vq*v
            fam = pf[i]
            for t in range(nt): acc[fam,t] += pw[i,t]*f
        for fam in range(2):
            for t in range(nt): part[iw,ic,fam,t] = acc[fam,t]
    out = np.zeros((nw,nt),dtype=np.complex128)
    for iw in range(nw):
        z = omegas[iw] + 1j*delta
        for t in range(nt):
            a0 = 0j ; a1 = 0j
            for ic in range(nchunk):
                a0 += part[iw,ic,0,t] ; a1 += part[iw,ic,1,t]
            out[iw,t] = const[t] + a0 + a1/z
    return out


def _sum_sources(dim,nk,es,terms,W,energies,delta,tol):
    """Same output as _sum_simplices, from the sources"""
    import numba
    s = shg_sources(dim,nk,es,terms,W,delta,tol)
    vkind = 2 if dim==2 else 1
    nw = len(energies)
    nsrc = len(s["ev"])+len(s["qa"])+len(s["ep"])
    # enough tasks to keep every thread busy when there are few frequencies
    nchunk = max(1,min(nsrc//1024,-(-4*numba.get_num_threads()//max(nw,1))))
    return _evaluate_sources_jit(s["ev"],s["cv"],vkind,s["qa"],s["qh"],
            s["qf"],s["qw"],s["ep"],s["pf"],s["pstart"],s["plen"],s["cp"],
            s["pw"],s["const"],energies,delta,nchunk)


def _sum_simplices(dim,nk,es,terms,W,energies,delta,tol):
    """The direct sum, every simplex at every frequency"""
    sims,_ = simplices(dim,nk)
    Wt = np.zeros((len(sims),)+W.shape[1:],dtype=np.complex128)
    for v in range(sims.shape[1]): Wt += W[sims[:,v]]
    Wt = Wt/sims.shape[1] # vertex average of the weights
    return _shg_simplex_jit(sims,np.ascontiguousarray(es),Wt,terms,energies,
            delta,tol)


_kernels = {
    "sources": _sum_sources, # sources built once, then summed per frequency
    "simplices": _sum_simplices, # every simplex at every frequency
}


# ------------------------------------------------------------ fields

def _fields(tms,dirs,d,hkgen,ks,T,trip,cat,tol):
    """Band energies es[k,n] and term weights W[k,term,t] on a set of
    k-points, in batches"""
    terms,Ti,Tii,Tiii,Tiv,Tv = cat
    nb = Ti.shape[0]
    es = np.zeros((len(ks),nb))
    W = np.zeros((len(ks),len(terms),len(trip)),dtype=np.complex128)
    vertices = shg.triplet_vertices(trip) # the only ones _weights_jit reads
    parallel.set_num_threads() # set the number of threads for numba
    for i0 in range(0,len(ks),shg._kbatch): # loop over batches of k-points
        kb = ks[i0:i0+shg._kbatch]
        e,h1,h2,h3 = shg._eigenbasis_derivatives(tms,dirs,d,hkgen,kb,
                vertices=vertices)
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


def _sigma_from_fields(dim,nk,es,terms,W,energies,delta,cellvol,tol,
        kernel="sources"):
    """Simplex sum and normalization, from the fields on the mesh.
    kernel="simplices" is the direct sum over simplices, kept as the
    reference that the tests compare the default against"""
    if kernel not in _kernels:
        raise ValueError("unknown simplex kernel "+repr(kernel)+
                ", accepted values are "+", ".join(_kernels))
    frac = 1./(2*nk*nk) if dim==2 else 1./nk # as in simplices(dim,nk)
    energies = np.asarray(energies,dtype=np.float64)
    parallel.set_num_threads() # set the number of threads for numba
    s = _kernels[kernel](dim,nk,es,terms,W,energies,delta,tol)
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
    tensor cross interpolation of each one (qtcitk/meshfields.qtci_fields),
    all of them sharing one cache of diagonalizations. info, if a dict,
    receives the number of diagonalizations and the ranks"""
    check_broadening(delta)
    if h.dimensionality!=2:
        raise NotImplementedError("integration='qtci' of the second-harmonic "
                "conductivity is only implemented in 2D, got dimensionality "
                +str(h.dimensionality)+"; use integration='triangle', in 1D "
                "a fine mesh is already cheap")
    from ..qtcitk.meshfields import qtci_fields
    sym = _symmetrized_triplets(trip)
    symarr = np.array(sym,dtype=np.int64)
    tms,dirs,d,hkgen,cellvol,scale = shg._derivative_tensors(h)
    tol = degeneracy_tol*scale
    nb = tms.shape[1]
    R = max(1,int(np.ceil(np.log2(nk)))) # bits per direction
    n = 2**R
    # coarse scan: fixes the terms and the fields that matter (qtci_fields
    # repeats it on the same points to seed the pivots)
    nc0 = 2**min(R,5)
    ks0 = kubo._kmesh(h,nc0)
    cat = _term_catalogue(nb)
    es0,W0 = _fields(tms,dirs,d,hkgen,ks0,T,symarr,cat,tol)
    keep = np.max(np.abs(W0),axis=(0,2))
    keep = keep>1e-12*max(np.max(keep),1e-300)
    terms = cat[0][keep]
    W0 = W0[:,keep]
    def value(kind,p,c,e,W):
        if kind=="E": return e[...,p]
        if kind=="re": return W[...,p,c].real
        return W[...,p,c].imag
    candidates = [("E",p,None) for p in range(nb)]
    for p in range(len(terms)):
        for c in range(len(sym)):
            candidates += [("re",p,c),("im",p,c)]
    # the fields to interpolate, and where every other field comes from
    wscale = np.max(np.abs(W0),axis=(0,2)) # size of each term's weight
    fields = [] # (kind,p,c) of the fields given to qtci_fields
    coarse = [] # their values on the coarse scan
    alias = [] # (kind,p,c,index in fields,sign) of every nonzero field
    for (kind,p,c) in candidates:
        f0 = value(kind,p,c,es0,W0)
        if np.max(np.abs(f0))==0.: continue # identically zero
        # a part that is only rounding noise (the imaginary part of a weight
        # that is real) would be interpolated to full rank: it is zero
        if kind!="E" and np.max(np.abs(f0))<1e-10*wscale[p]: continue
        found = None
        for i,g0 in enumerate(coarse): # equal or opposite to a field already
            # kept on all the points of the coarse scan: the same function
            # (a symmetry of the model), interpolated once
            for sgn in (1.,-1.):
                if np.max(np.abs(f0-sgn*g0))<1e-12*np.max(np.abs(f0)):
                    found = (i,sgn) ; break
            if found is not None: break
        if found is None:
            found = (len(fields),1.)
            fields.append((kind,p,c))
            coarse.append(f0)
        alias.append((kind,p,c)+found)
    def evaluate(ks): # the kept fields at the k-points ks
        e,W = _fields(tms,dirs,d,hkgen,ks,T,symarr,cat,tol)
        W = W[:,keep]
        out = np.zeros((len(ks),len(fields)))
        for f,(kind,p,c) in enumerate(fields): out[:,f] = value(kind,p,c,e,W)
        return out
    vals,ncalls,rk = qtci_fields(evaluate,R,len(fields),tolerance=tolerance,
            ncoarse=nc0)
    npts = n*n
    es = np.zeros((npts,nb))
    W = np.zeros((npts,len(terms),len(sym)),dtype=np.complex128)
    for (kind,p,c,f,sgn) in alias:
        full = sgn*vals[:,f]
        if kind=="E": es[:,p] = full
        elif kind=="re": W[:,p,c] += full
        else: W[:,p,c] += 1j*full
    if isinstance(info,dict):
        info["diagonalizations"] = ncalls
        info["ranks"] = dict(zip(fields,rk))
    s = _sigma_from_fields(2,n,es,terms,W,energies,delta,cellvol,tol)
    return _unpack(s,sym,trip)
