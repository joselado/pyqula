# Linear simplex (segment in 1D, triangle in 2D, tetrahedron in 3D)
# integration of the bare susceptibility of a crystal,
#   chi_ij(q,w) = int dk sum_nm M^ij_nm(k) (f(E_n(k)) - f(E_m(k+q)))
#                                       / (E_n(k) - E_m(k+q) - w + i delta),
#   M^ij_nm(k) = <n,k|A_i|m,k+q><m,k+q|B_j|n,k>,
# over the Brillouin zone normalized to one, in the convention of
# chitk/chiAB.py (integration="triangle" of chiAB_q).
#
# On a uniform mesh the two Fermi steps and the resonance are sampled
# point by point, so at low temperature the mesh needs nk of order v/T,
# and of order v/delta for the dynamic response. Here the band energies
# E_n(k) and E_m(k+q) and the matrix elements M(k) are interpolated
# linearly inside each simplex of the mesh, and the steps and the
# resolvent are integrated exactly. The mesh then only has to resolve the
# bands and the matrix elements, which are smooth. This is the analytic
# tetrahedron method of the Lindhard function (A. H. MacDonald, S. H.
# Vosko and P. T. Coleridge, J. Phys. C 12, 2991 (1979); J. Rath and A. J.
# Freeman, PRB 11, 2109 (1975)) with the complex energies of Ph. Lambin
# and J.-P. Vigneron, PRB 29, 3430 (1984).
#
# The construction at T=0, for one simplex and one pair (n,m):
# f_n - f_m = theta_n (1 - theta_m) - theta_m (1 - theta_n), so the
# integral is over the region A where n is occupied and m empty, minus the
# region B where m is occupied and n empty. Each region is the
# intersection of the simplex with two half-spaces, a convex polytope,
# which is tiled by sub-simplices in two stages: the part E_n < mu is cut
# out first, and each of its pieces is cut again by E_m > mu, with the
# corner, prism and truncated-corner decompositions of simplexdm. There
# are at most 2 x 2 triangles or 3 x 3 tetrahedra per region, all with
# positive volume. The tiling never leaves its region, so inside A the
# transition energy D = E_n - E_m is negative and inside B positive: the
# static response w = 0 has no resonance inside a piece, and its delta = 0
# limit is a real, finite integral. Since M is linear inside the simplex,
# each piece gives a weight per vertex j of the simplex,
#   w_j = int_piece lambda_j / (D - w + i delta),
# with lambda_j the barycentric coordinate of the vertex. On a piece of
# dimension d with node values x_s = D_s - w - i delta, the Hermite-
# Genocchi formula with one repeated node gives the average of mu_s g(D),
# mu_s the barycentric coordinates of the piece, as
#   d! G[x_0,...,x_d,x_s],  G^(d+1) = g,
# so with g = 1/(z - D) it is d! psi_(d+1)[x_0..x_d,x_s] in the notation
# of dostk/simplexdos.py, and the response is minus its complex conjugate.
# The divided differences are those of dostk/simplexdos.py (its Newton
# table, with the clusters of close nodes replaced by their Taylor series).
#
# A pair whose transition energy vanishes on the whole simplex (n = m at
# q = 0, the spin partners of a spin-degenerate band, q a reciprocal
# lattice vector) is the static limit q -> 0 of the intraband continuum,
# as in chiAB.lindhard_pair: (f_n - f_m)/(E_n - E_m) tends to f'(E_n), so
# the pair gives f'(E_n) i delta/(i delta - w), integrated with the Fermi
# surface weights int_T lambda_j delta(mu - E_n) of the simplex.
#
# At temperature T, f(E) = int dx (-f'(x)) theta(x - E), and the same x
# enters both steps, so the response at T is the T=0 response at the
# chemical potential x convolved with -f'(x). As in dmtk/simplexdm.py the
# integral over x is split at the vertex energies of both bands, at 0, at
# a fixed set of multiples of T up to 40 T, and at the energies where a
# cut through an edge of the simplex meets the resonance D = w, and each
# piece is done by Gauss-Legendre quadrature in the variable f(x), on
# nodes graded towards both ends of the piece (see _quadrature). A pair
# whose two bands keep the same side of the window on every vertex has an
# x-independent T=0 response there, which is evaluated once. So does a
# piece of x between the energy ranges of the two bands on the simplex
# (_interval_kind), where the whole simplex is in A or in B: its nodes
# are replaced by one evaluation times f(a) - f(b), and these pieces of
# each pair are summed into one evaluation for A and one for B.
#
# The weights are summed over the simplices that share each vertex, and
# averaged over the degenerate multiplets of n at k and of m at k+q, which
# makes the result independent of the basis eigh picks inside a multiplet.
# The bands are ordered by energy at each k, so two bands that cross are
# interpolated with a kink, as in the simplex DOS. At a mesh q where the
# bands at k+q cross on the mesh points (a folded supercell), chi is then
# not continuous in q: off the mesh q it depends on the direction of
# approach, and its value at the mesh q is the mean of these limits. The
# step is a mesh error of second order, 1.5e-3 of the largest entry at
# nk = 16 on a 2x2 honeycomb supercell, so a q-path through such points
# shows a small step that a finer mesh removes. A basis per simplex removes
# it at no gain in accuracy, see future_development/
# tetrahedron_susceptibility.md, "A basis per simplex for crossing bands".
import numpy as np
from numba import jit, prange

from .. import parallel
from ..dostk.simplexdos import _psi, _taylor_dd, _close, _sort_small
from ..dostk.simplexdos import _harm, _fact, _nterms
from ..dmtk.simplexdm import _tbreaks, _fermi
_ngl = 12 # Gauss-Legendre nodes per piece of the finite-T integral, on
          # graded nodes (see _quadrature)
_gx,_gw = np.polynomial.legendre.leggauss(_ngl)


# ------------------------------------------------------------ tilings
#
# The part e < 0 of a simplex with vertex values e sorted in increasing
# order, nneg of them negative, as sub-simplices whose vertices are either
# a vertex a of the simplex, (a,a), or the zero of e on the edge (a,b),
# a < nneg <= b. Indexed [p-2, nneg, piece, vertex, 2].

def _tilings():
    R = np.zeros((3,4,3,4,2),dtype=np.int64)
    NR = np.zeros((3,4),dtype=np.int64)
    rec = {
        (2,1): [[(0,0),(0,1)]],
        (3,1): [[(0,0),(0,1),(0,2)]],
        (3,2): [[(0,0),(1,1),(1,2)],[(0,0),(1,2),(0,2)]],
        (4,1): [[(0,0),(0,1),(0,2),(0,3)]],
        # a prism (v0,c02,c03)-(v1,c12,c13)
        (4,2): [[(0,0),(0,2),(0,3),(1,1)],[(0,2),(0,3),(1,1),(1,2)],
                [(0,3),(1,1),(1,2),(1,3)]],
        # a prism (v0,v1,v2)-(c03,c13,c23), the simplex minus a corner
        (4,3): [[(0,0),(1,1),(2,2),(0,3)],[(1,1),(2,2),(0,3),(1,3)],
                [(2,2),(0,3),(1,3),(2,3)]],
    }
    for (p,nneg),pieces in rec.items():
        NR[p-2,nneg] = len(pieces)
        for ip,piece in enumerate(pieces):
            for v,(a,b) in enumerate(piece):
                R[p-2,nneg,ip,v,0] = a
                R[p-2,nneg,ip,v,1] = b
    return R,NR

_R,_NR = _tilings()
_maxpieces = 18 # 9 per region in 3D


@jit(nopython=True,cache=True)
def _clip_neg(V,e,p,R,NR,out,n0):
    """append to out[n0:] the simplices that tile the part e < 0 of the
    simplex whose vertices are the rows of V (barycentric coordinates in
    the parent simplex), e being the values of a linear function at those
    vertices. Returns the new number of pieces"""
    idx = np.empty(p,dtype=np.int64)
    es = np.empty(p)
    for i in range(p):
        idx[i] = i
        es[i] = e[i]
    for i in range(1,p): # insertion sort, carrying the indices
        x = es[i]; ix = idx[i]
        j = i-1
        while j>=0 and es[j]>x:
            es[j+1] = es[j]; idx[j+1] = idx[j]
            j -= 1
        es[j+1] = x; idx[j+1] = ix
    nneg = 0
    for i in range(p):
        if es[i]<0.: nneg += 1
    if nneg==0: return n0
    if nneg==p:
        for v in range(p):
            for j in range(p): out[n0,v,j] = V[v,j]
        return n0+1
    for ip in range(NR[p-2,nneg]):
        for v in range(p):
            a = R[p-2,nneg,ip,v,0]
            b = R[p-2,nneg,ip,v,1]
            if a==b:
                for j in range(p): out[n0,v,j] = V[idx[a],j]
            else: # es[a] < 0 <= es[b]
                t = -es[a]/(es[b]-es[a])
                for j in range(p):
                    out[n0,v,j] = (1.-t)*V[idx[a],j] + t*V[idx[b],j]
        n0 += 1
    return n0


@jit(nopython=True,cache=True)
def _regions(en,em,p,R,NR,I,P1,out,sgn,tmp):
    """tile the region A (en < 0 < em) with sign +1 and the region B
    (em < 0 < en) with sign -1. Returns the number of pieces"""
    n = 0
    for reg in range(2):
        if reg==0: first = en; second = em; s = 1.
        else: first = em; second = en; s = -1.
        n1 = _clip_neg(I,first,p,R,NR,P1,0)
        for a in range(n1):
            for v in range(p):
                acc = 0.
                for j in range(p): acc += P1[a,v,j]*second[j]
                tmp[v] = -acc
            n0 = n
            n = _clip_neg(P1[a],tmp,p,R,NR,out,n)
            for i in range(n0,n): sgn[i] = s
    return n


@jit(nopython=True,cache=True)
def _det(C,p):
    """determinant of the p x p barycentric matrix C, the volume of the
    piece in units of the parent simplex up to its sign"""
    if p==2: return C[0,0]*C[1,1]-C[0,1]*C[1,0]
    if p==3:
        return (C[0,0]*(C[1,1]*C[2,2]-C[1,2]*C[2,1])
               -C[0,1]*(C[1,0]*C[2,2]-C[1,2]*C[2,0])
               +C[0,2]*(C[1,0]*C[2,1]-C[1,1]*C[2,0]))
    out = 0.
    for c in range(4): # cofactor expansion along the first row
        m = np.empty((3,3))
        for r in range(1,4):
            cc = 0
            for k in range(4):
                if k==c: continue
                m[r-1,cc] = C[r,k]
                cc += 1
        d3 = (m[0,0]*(m[1,1]*m[2,2]-m[1,2]*m[2,1])
             -m[0,1]*(m[1,0]*m[2,2]-m[1,2]*m[2,0])
             +m[0,2]*(m[1,0]*m[2,1]-m[1,1]*m[2,0]))
        if c%2==0: out += C[0,c]*d3
        else: out -= C[0,c]*d3
    return out


# ------------------------------------------------------------ the kernel

@jit(nopython=True,cache=True)
def _ddK_col(K,t,P,s,harm,fact,h,d,dcol):
    """divided difference psi_K[t_0+is,...,t_(P-1)+is] for sorted real
    parts t, with the Newton table of dostk.simplexdos._dd and its Taylor
    series for clusters of close nodes, for any number of nodes. The values
    psi_K(t_i + is) of the first column are given in dcol, which the
    repeated-node tables of one piece share"""
    c = 0.
    for i in range(P): c += t[i]
    if _close(t[P-1]-t[0],c/P,s): # the whole set at once
        return _taylor_dd(K,t,0,P-1,s,harm,fact,h)
    for i in range(P): d[i] = dcol[i]
    for m in range(1,P): # Newton table, in place
        for i in range(P-m):
            sp = t[i+m]-t[i]
            cc = 0.
            for j in range(i,i+m+1): cc += t[j]
            if _close(sp,cc/(m+1),s):
                d[i] = _taylor_dd(K,t,i,i+m,s,harm,fact,h)
            else: d[i] = (d[i+1]-d[i])/sp
    return d[0]


@jit(nopython=True,cache=True)
def _vertex_averages(t,p,s,harm,fact,h,d,tb,ib,pb,tt,dcol,a):
    """a[s'] = (p-1)! psi_p[t_0..t_(p-1),t_s'] + i s for every node s', the
    averages of mu_s'/(z - D) over a piece with node values t + i s. The
    psi_p of the nodes is evaluated once for all p tables"""
    for i in range(p):
        tb[i] = t[i]; ib[i] = i
    for i in range(1,p): # insertion sort, carrying the indices
        x = tb[i]; ix = ib[i]
        j = i-1
        while j>=0 and tb[j]>x:
            tb[j+1] = tb[j]; ib[j+1] = ib[j]
            j -= 1
        tb[j+1] = x; ib[j+1] = ix
    c = 0.
    for i in range(p): c += tb[i]
    close = _close(tb[p-1]-tb[0],c/p,s)
    if not close:
        for i in range(p): pb[i] = _psi(p,complex(tb[i],s),harm,fact)
    for pos in range(p): # the node tb[pos] repeated
        for i in range(pos+1):
            tt[i] = tb[i]; dcol[i] = pb[i]
        for i in range(pos,p):
            tt[i+1] = tb[i]; dcol[i+1] = pb[i]
        if close: dd = _taylor_dd(p,tt,0,p,s,harm,fact,h)
        else: dd = _ddK_col(p,tt,p+1,s,harm,fact,h,d,dcol)
        a[ib[pos]] = fact[p-1]*dd


@jit(nopython=True,cache=True)
def _add_pieces(C,sgn,npc,D,p,w,delta,coef,out,ws2,harm,fact):
    """out[j] += coef sum over the pieces of sgn int_piece lambda_j/(D - w
    + i delta), per unit volume of the parent simplex"""
    t,tb,ib,pb,tt,dcol,a,h,d = ws2
    for ic in range(npc):
        vol = abs(_det(C[ic],p))
        if vol==0.: continue
        for s in range(p):
            acc = 0.
            for j in range(p): acc += C[ic,s,j]*D[j]
            t[s] = acc-w
        _vertex_averages(t,p,-delta,harm,fact,h,d,tb,ib,pb,tt,dcol,a)
        for s in range(p): # the barycentric weight of the node s
            b = -np.conj(a[s])*(coef*sgn[ic]*vol)
            for j in range(p): out[j] += C[ic,s,j]*b


@jit(nopython=True,cache=True)
def _fermi_surface(en,p,coef,out,ws2,harm,fact):
    """out[j] += coef int_T lambda_j delta(E_n) per unit volume, as
    -Im/pi of the resolvent average at zero broadening"""
    t,tb,ib,pb,tt,dcol,a,h,d = ws2
    emin = en[0]; emax = en[0]
    for i in range(1,p):
        emin = min(emin,en[i]); emax = max(emax,en[i])
    if emin>=0. or emax<=0.: return
    for i in range(p): t[i] = en[i]
    _vertex_averages(t,p,-0.,harm,fact,h,d,tb,ib,pb,tt,dcol,a)
    for j in range(p): out[j] += coef*(-a[j].imag/np.pi)


@jit(nopython=True,cache=True)
def _static_factor(w,delta):
    """i delta/(i delta - w) of a degenerate pair, its delta -> 0 limit
    being 1 at w = 0 and 0 elsewhere"""
    if delta>0.: return 1j*delta/(1j*delta-w)
    if w==0.: return 1.+0j
    return 0j


@jit(nopython=True,cache=True)
def _kernel0(en0,em0,D,p,x,omegas,iw0,iw1,delta,degenerate,coef,out,acc,ws):
    """the T=0 weights at chemical potential x for the frequencies
    omegas[iw0:iw1], times coef, added to out[:,iw]. The tiling is done
    once for all of them. ws holds the work arrays"""
    I,P1,C,sgn,tmp,en,em,ws2,R,NR = ws
    for i in range(p):
        en[i] = en0[i]-x
        em[i] = em0[i]-x
    if degenerate:
        for v in range(p): acc[v] = 0.
        _fermi_surface(en,p,-1.+0j,acc,ws2,_harm,_fact)
        for iw in range(iw0,iw1):
            g = coef*_static_factor(omegas[iw],delta)
            for v in range(p): out[v,iw] += acc[v]*g
        return
    npc = _regions(en,em,p,R,NR,I,P1,C,sgn,tmp)
    if npc==0: return
    for iw in range(iw0,iw1):
        for v in range(p): acc[v] = 0.
        _add_pieces(C,sgn,npc,D,p,omegas[iw],delta,coef,acc,ws2,_harm,_fact)
        for v in range(p): out[v,iw] += acc[v]


@jit(nopython=True,cache=True)
def _knots(en,em,D,p,w,T,tbreaks,resonance,kn):
    """sorted breakpoints of the finite-T integral over x: the window
    clipped to the energies of the simplex, the vertex energies, 0, the
    multiples of T, and with resonance the energies where a cut through an
    edge meets the resonance D = w. Returns their number"""
    L = tbreaks[-1]*T
    lo = en[0]; hi = en[0]
    for i in range(p):
        lo = min(lo,en[i],em[i]); hi = max(hi,en[i],em[i])
    lo = max(lo,-L); hi = min(hi,L)
    if hi<=lo: return 0
    m = 0
    kn[m] = lo; m += 1
    kn[m] = hi; m += 1
    for i in range(p):
        if en[i]>lo and en[i]<hi: kn[m] = en[i]; m += 1
        if em[i]>lo and em[i]<hi: kn[m] = em[i]; m += 1
    if 0.>lo and 0.<hi: kn[m] = 0.; m += 1
    for tb in tbreaks:
        for s in (-tb*T,tb*T):
            if s>lo and s<hi: kn[m] = s; m += 1
    if resonance:
        for i in range(p): # the resonance on the edges
            for j in range(i+1,p):
                a = D[i]-w; b = D[j]-w
                if a*b<0.:
                    u = a/(a-b)
                    e1 = en[i]+u*(en[j]-en[i])
                    e2 = em[i]+u*(em[j]-em[i])
                    if e1>lo and e1<hi: kn[m] = e1; m += 1
                    if e2>lo and e2<hi: kn[m] = e2; m += 1
    kn[:m].sort()
    return m


@jit(nopython=True,cache=True)
def _interval(a,b,T):
    """the piece [a,b] of the integral over x in the variable f(x): the
    sign of the mirror x = -u, the ends f(u1) and the span f(u0) - f(u1)"""
    if b<=0.: # mirror: x = -u, u in [-b,-a]
        sg = -1.; u0 = -b; u1 = -a
    else:
        sg = 1.; u0 = a; u1 = b
    f0 = _fermi(u0,T); f1 = _fermi(u1,T)
    return sg,f1,f0-f1


@jit(nopython=True,cache=True)
def _graded_node(q,sg,f1,span,T):
    """the node q of a piece of the integral over x: whether it is kept,
    the chemical potential x and the weight of the quadrature"""
    # graded nodes, phi = f1 + span g(u) with the septic smoothstep
    # g, whose g' = 140 u^3 (1-u)^3 vanishes at both ends. The T=0
    # weights can have a logarithmic singularity at a breakpoint
    # (a mesh edge on which D = 0, the perfect nesting of the
    # square lattice at (pi,pi)), which plain Gauss-Legendre nodes
    # resolve only slowly: on that case at T = 5e-2, nk = 64 and
    # 128, 8 plain nodes were off by 6e-6 relative, 12 graded ones
    # by 7e-8, against 48 nodes
    u = 0.5*(1.+_gx[q])
    g = u**4*(35.-84.*u+70.*u*u-20.*u**3)
    dg = 140.*u**3*(1.-u)**3
    phi = f1+span*g
    if phi<=0. or phi>=1.: return False,0.,0.
    x = sg*T*(np.log1p(-phi)-np.log(phi))
    return True,x,0.5*_gw[q]*span*dg


@jit(nopython=True,cache=True)
def _interval_kind(en,em,p,a,b):
    """how the T=0 weights depend on x inside the piece (a,b) of the
    integral over x, whose ends are knots, so that no vertex energy lies
    inside it: -1 when a band has vertices on both sides of x and the
    tiling moves with x, 1 when the whole simplex is in A (every E_n < x
    < every E_m), 2 when it is in B, and 0 when A and B are empty. In the
    last three cases the weights are the same at every x of the piece"""
    x = 0.5*(a+b)
    nn = 0; nm = 0
    for i in range(p):
        if en[i]<x: nn += 1
        if em[i]<x: nm += 1
    if (nn>0 and nn<p) or (nm>0 and nm<p): return -1
    if nn==p and nm==0: return 1
    if nm==p and nn==0: return 2
    return 0


@jit(nopython=True,cache=True)
def _quadrature(en,em,D,p,omegas,iw0,iw1,delta,T,degenerate,tbreaks,
        resonance,out,acc,ws,kn):
    """the convolution of the T=0 weights with -f'(x) for the frequencies
    omegas[iw0:iw1], on the breakpoints of the frequency omegas[iw0]. A
    piece of x where the weights do not depend on x (_interval_kind >= 0)
    is the weights at one x times the weight f(a) - f(b) of the piece,
    which is the sum of the weights of its nodes to rounding, so the
    pieces of A and those of B are each evaluated once"""
    m = _knots(en,em,D,p,omegas[iw0],T,tbreaks,resonance,kn)
    wA = 0.; wB = 0.; xA = 0.; xB = 0.
    for i in range(m-1):
        a = kn[i]; b = kn[i+1]
        if b<=a: continue
        kind = _interval_kind(en,em,p,a,b)
        if kind==0: continue
        sg,f1,span = _interval(a,b,T)
        if span<=0.: continue
        if kind==1:
            wA += span; xA = 0.5*(a+b)
            continue
        if kind==2:
            wB += span; xB = 0.5*(a+b)
            continue
        _interval_nodes(en,em,D,p,a,b,omegas,iw0,iw1,delta,T,degenerate,1.,
                out,acc,ws)
    if wA>0.:
        _kernel0(en,em,D,p,xA,omegas,iw0,iw1,delta,degenerate,wA,out,acc,ws)
    if wB>0.:
        _kernel0(en,em,D,p,xB,omegas,iw0,iw1,delta,degenerate,wB,out,acc,ws)


@jit(nopython=True,cache=True)
def _interval_nodes(en,em,D,p,a,b,omegas,iw0,iw1,delta,T,degenerate,coef,
        out,acc,ws):
    """the graded quadrature over the piece (a,b) of x, times coef"""
    sg,f1,span = _interval(a,b,T)
    if span<=0.: return
    for q in range(_gx.shape[0]):
        ok,x,wq = _graded_node(q,sg,f1,span,T)
        if not ok: continue
        _kernel0(en,em,D,p,x,omegas,iw0,iw1,delta,degenerate,coef*wq,out,
                acc,ws)


@jit(nopython=True,cache=True)
def _constant_window(en,em,p,L):
    """whether both bands keep one side of the window [-L,L] on every
    vertex, so that the T=0 weights do not depend on x inside it"""
    for b in range(2):
        e = en if b==0 else em
        below = 0; above = 0
        for i in range(p):
            if e[i]<=-L: below += 1
            elif e[i]>=L: above += 1
        if not (below==p or above==p): return False
    return True


@jit(nopython=True,cache=True)
def _kernelT(en,em,D,p,omegas,delta,T,degenerate,tbreaks,out,acc,ws,kn):
    """the weights at temperature T, every frequency, added to out[:,iw].
    When delta < T the breakpoints include the resonance of each frequency
    and the quadrature is done frequency by frequency; otherwise the
    features in x are at least T wide and the frequencies share the
    nodes"""
    nw = omegas.shape[0]
    if _constant_window(en,em,p,tbreaks[-1]*T):
        _kernel0(en,em,D,p,0.,omegas,0,nw,delta,degenerate,1.,out,acc,ws)
        return
    if delta<T:
        for iw in range(nw):
            _quadrature(en,em,D,p,omegas,iw,iw+1,delta,T,degenerate,tbreaks,
                    True,out,acc,ws,kn)
    else:
        _quadrature(en,em,D,p,omegas,0,nw,delta,T,degenerate,tbreaks,
                False,out,acc,ws,kn)


@jit(nopython=True,cache=True)
def _pair_fields(sim,E1,E2,it,n,m,tol,L,En,Em,D):
    """the energies En, Em and the transition energy D of the pair (n,m)
    on the vertices of the simplex it. Returns whether the pair can
    contribute at a chemical potential of the window [-L,L], and whether
    its transition energy vanishes on the whole simplex"""
    p = sim.shape[1]
    nmin = 1e300; nmax = -1e300
    mmin = 1e300; mmax = -1e300
    dmax = 0.
    for v in range(p):
        En[v] = E1[sim[it,v],n]
        Em[v] = E2[sim[it,v],m]
        D[v] = En[v]-Em[v]
        nmin = min(nmin,En[v]); nmax = max(nmax,En[v])
        mmin = min(mmin,Em[v]); mmax = max(mmax,Em[v])
        dmax = max(dmax,abs(D[v]))
    degenerate = dmax<tol
    # skip a pair with an empty A and B at every chemical potential of
    # the window [-L,L]
    if degenerate: return not (nmin>=L or nmax<=-L),True
    # A needs nmin < x < mmax, B mmin < x < nmax
    a = nmin<L and mmax>-L and nmin<mmax
    b = mmin<L and nmax>-L and mmin<nmax
    return (a or b),False


@jit(nopython=True,parallel=True,cache=True)
def _weights_jit(sim,E1,E2,omegas,delta,T,tol,tbreaks,out):
    """out[it,n,m,v,iw]: the weight of the vertex v of the simplex it for
    the pair (n at k, m at k+q) at the frequency iw, per unit volume of the
    simplex"""
    ns = sim.shape[0]; p = sim.shape[1]; nb = E1.shape[1]
    nw = omegas.shape[0]
    L = tbreaks[-1]*T
    for it in prange(ns):
        I = np.zeros((p,p))
        for i in range(p): I[i,i] = 1.
        P1 = np.zeros((3,p,p))
        C = np.zeros((_maxpieces,p,p))
        sgn = np.zeros(_maxpieces)
        tmp = np.zeros(p); en = np.zeros(p); em = np.zeros(p)
        ws2 = (np.zeros(p),np.zeros(p),np.zeros(p,dtype=np.int64),
               np.zeros(p,dtype=np.complex128),np.zeros(p+1),
               np.zeros(p+1,dtype=np.complex128),
               np.zeros(p,dtype=np.complex128),np.zeros(_nterms+1),
               np.zeros(p+1,dtype=np.complex128))
        ws = (I,P1,C,sgn,tmp,en,em,ws2,_R,_NR)
        kn = np.zeros(4+2*p+p*(p-1)+2*tbreaks.shape[0])
        En = np.zeros(p); Em = np.zeros(p); D = np.zeros(p)
        acc = np.zeros(p,dtype=np.complex128)
        res = np.zeros((p,nw),dtype=np.complex128)
        for n in range(nb):
            for m in range(nb):
                active,degenerate = _pair_fields(sim,E1,E2,it,n,m,tol,L,
                        En,Em,D)
                if not active: continue
                for v in range(p):
                    for iw in range(nw): res[v,iw] = 0.
                if T==0.:
                    _kernel0(En,Em,D,p,0.,omegas,0,nw,delta,degenerate,1.,
                            res,acc,ws)
                else:
                    _kernelT(En,Em,D,p,omegas,delta,T,degenerate,tbreaks,
                            res,acc,ws,kn)
                for v in range(p):
                    for iw in range(nw): out[it,n,m,v,iw] = res[v,iw]


@jit(nopython=True,parallel=True,cache=True)
def _multiplet_average_pairs(E1,E2,W,tol):
    """replace W[k,n,m,:] by its mean over the block of the degenerate
    multiplet of n at k and the one of m at k+q"""
    nk = W.shape[0]; nb = W.shape[1]; nw = W.shape[3]
    for ik in prange(nk):
        s1 = np.zeros(nb,dtype=np.int64) # first band of the multiplet
        s2 = np.zeros(nb,dtype=np.int64)
        for i in range(1,nb):
            s1[i] = s1[i-1] if E1[ik,i]-E1[ik,i-1]<tol else i
            s2[i] = s2[i-1] if E2[ik,i]-E2[ik,i-1]<tol else i
        e1 = np.zeros(nb,dtype=np.int64) # one past its last band
        e2 = np.zeros(nb,dtype=np.int64)
        e1[nb-1] = nb; e2[nb-1] = nb
        for i in range(nb-2,-1,-1):
            e1[i] = e1[i+1] if s1[i+1]==s1[i] else i+1
            e2[i] = e2[i+1] if s2[i+1]==s2[i] else i+1
        for n in range(nb):
            if s1[n]!=n: continue
            for m in range(nb):
                if s2[m]!=m: continue
                g = (e1[n]-n)*(e2[m]-m)
                if g==1: continue
                for iw in range(nw):
                    acc = 0j
                    for a in range(n,e1[n]):
                        for b in range(m,e2[m]): acc += W[ik,a,b,iw]
                    acc = acc/g
                    for a in range(n,e1[n]):
                        for b in range(m,e2[m]): W[ik,a,b,iw] = acc


# ------------------------------------------------------------ source kernel
#
# The weight kernel above redoes the divided differences of every piece at
# every frequency. The pieces themselves do not depend on the frequency:
# at T = 0 and at every node x of the shared finite-T quadrature the
# tiling is fixed, and a piece is fully given by the transition
# energies D_s on its nodes and, per node, the coefficient
#   c_s = coef sgn vol sum_j C_sj G_j,
# with G_j the matrix elements of the vertex j of the parent simplex,
# already contracted over the operators and averaged over the degenerate
# multiplets (that average is moved from the weights to G, which is exact
# for the contraction). The response at w is sum_s c_s r_s(w), with r_s
# the minus conjugate of the vertex averages of _vertex_averages.
#
# The pieces are the sources, but they are not merged by exact keys as in
# conductivitytk/kubosimplex.py: there the frequency-independent branch
# rule (exact partial fractions above a spread of delta) loses
# eps (|x|/s)^d on a simplex of spread s at distance |x| from the
# resonance, which is acceptable for d = 2 but not for the repeated-node
# tables here, d = 3 in 2D and 4 in 3D, and has no scale at all at delta
# = 0. Instead the sources are grouped in bins of energy, in the manner of
# a fast multipole sum on a line. A piece of spread s goes to the finest
# level whose bin width b >= s, so that its nodes lie within b of the
# center c of its bin. Far from the bin, |z - c| > _far b, the average of
# mu_s/(z - D) over the piece is the series
#   sum_n (p-1)! n!/(n+p)! h_n(y_0..y_(p-1),y_s) u^(n+1),
# y = D - c, u = 1/(z - c), h_n the complete homogeneous symmetric
# polynomial (the series of the Taylor branch of dostk.simplexdos), whose
# terms are bounded by (b/|z-c|)^n/p, so _nmom terms give 3^-40. The
# moments of every bin are summed once over its sources, and every
# frequency costs the bins far from it, by their series, plus the pieces
# of the bins near it, by the exact divided differences of the weight
# kernel with the same branches. The near bins are a few per level, so
# the exact work per frequency is a small fraction of the pieces.
#
# With delta < T the reference kernel adds to the knots in x the energies
# where a cut through an edge meets the resonance of each frequency, so
# its nodes depend on the frequency. Those knots only fall inside the
# pieces of x where the tiling moves, and only for the frequencies inside
# the range of D on the simplex. The source kernel keeps the shared nodes
# for every frequency and adds, for those simplices and frequencies, the
# quadrature on the refined pieces of x minus the one on the shared
# pieces, evaluated directly (_resonance_fix). This is the reference
# result to rounding, at the cost of a few pieces of x per simplex and
# frequency that crosses it.

_lev = 12 # the finest bins are 2^-_lev of the range of the transition
          # energies, the coarsest one is the whole range
_far = 3. # a bin of width b is summed by its moments where |z - c| > _far b
_nmom = 40 # terms of the series of a bin, (1/_far)^_nmom < 1e-18
_maxsrc = 2**21 # pieces held at once
_maxsim = 2**18 # simplices per pass of the piece kernel


@jit(nopython=True,cache=True)
def _workspace(p,ntb):
    """the work arrays of the tiling and of the divided differences"""
    I = np.zeros((p,p))
    for i in range(p): I[i,i] = 1.
    ws2 = (np.zeros(p),np.zeros(p),np.zeros(p,dtype=np.int64),
           np.zeros(p,dtype=np.complex128),np.zeros(p+1),
           np.zeros(p+1,dtype=np.complex128),
           np.zeros(p,dtype=np.complex128),np.zeros(_nterms+1),
           np.zeros(p+1,dtype=np.complex128))
    ws = (I,np.zeros((3,p,p)),np.zeros((_maxpieces,p,p)),
          np.zeros(_maxpieces),np.zeros(p),np.zeros(p),np.zeros(p),ws2,
          _R,_NR)
    kn = np.zeros(4+2*p+p*(p-1)+2*ntb)
    return ws,kn


@jit(nopython=True,cache=True)
def _bin_of(t,p,dmin,span,off):
    """flat index of the bin of a piece with node energies t"""
    tmin = t[0]; tmax = t[0]
    for i in range(1,p):
        tmin = min(tmin,t[i]); tmax = max(tmax,t[i])
    s = tmax-tmin
    b = span*2.**(-_lev)
    L = 0
    while L<_lev and s>b:
        b *= 2.
        L += 1
    j = int((0.5*(tmin+tmax)-dmin)/b)
    j = max(0,min(j,off[L+1]-off[L]-1))
    return off[L]+j


@jit(nopython=True,cache=True)
def _is_near(c,b,w,delta):
    """whether the bin of center c and width b is summed piece by piece
    at the frequency w"""
    return (w-c)*(w-c)+delta*delta<=(_far*b)*(_far*b)


@jit(nopython=True,cache=True)
def _emit0(En,Em,D,p,x,degenerate,coef,kv,G,n,m,ws,acc,i0,pt,pc,pb,
        dgrow,dmin,span,off):
    """the pieces of the T=0 response at chemical potential x, times coef,
    written from index i0 on. A degenerate pair adds its Fermi surface
    term to dgrow instead. Returns the new number of pieces"""
    I,P1,C,sgn,tmp,en,em,ws2,R,NR = ws
    for i in range(p):
        en[i] = En[i]-x
        em[i] = Em[i]-x
    nc = G.shape[3]
    if degenerate:
        for v in range(p): acc[v] = 0.
        _fermi_surface(en,p,-1.+0j,acc,ws2,_harm,_fact)
        for v in range(p):
            for c in range(nc): dgrow[c] += coef*acc[v]*G[kv[v],n,m,c]
        return i0
    npc = _regions(en,em,p,R,NR,I,P1,C,sgn,tmp)
    for ic in range(npc):
        vol = abs(_det(C[ic],p))
        if vol==0.: continue
        f = coef*sgn[ic]*vol
        for s in range(p):
            a = 0.
            for j in range(p): a += C[ic,s,j]*D[j]
            pt[i0,s] = a
            for c in range(nc):
                z = 0j
                for j in range(p): z += C[ic,s,j]*G[kv[j],n,m,c]
                pc[i0,s,c] = f*z
        pb[i0] = _bin_of(pt[i0],p,dmin,span,off)
        i0 += 1
    return i0


@jit(nopython=True,cache=True)
def _emitT(En,Em,D,p,T,degenerate,tbreaks,kv,G,n,m,ws,acc,kn,i0,pt,
        pc,pb,dgrow,dmin,span,off):
    """the pieces at temperature T on the nodes that every frequency
    shares, the path of _kernelT for delta >= T"""
    if _constant_window(En,Em,p,tbreaks[-1]*T):
        return _emit0(En,Em,D,p,0.,degenerate,1.,kv,G,n,m,ws,acc,i0,
                pt,pc,pb,dgrow,dmin,span,off)
    nkn = _knots(En,Em,D,p,0.,T,tbreaks,False,kn)
    wA = 0.; wB = 0.; xA = 0.; xB = 0.
    lo = np.empty(p); hi = np.empty(p) # a band below or above every node
    for v in range(p):
        lo[v] = -1.; hi[v] = 1.
    for i in range(nkn-1): # as in _quadrature
        a = kn[i]; b = kn[i+1]
        if b<=a: continue
        kind = _interval_kind(En,Em,p,a,b)
        if kind==0: continue
        sg,f1,sp = _interval(a,b,T)
        if sp<=0.: continue
        if kind==1:
            wA += sp; xA = 0.5*(a+b)
            continue
        if kind==2:
            wB += sp; xB = 0.5*(a+b)
            continue
        side = 0 if degenerate else _complement(En,Em,p,0.5*(a+b))
        if side!=0:
            emin = En[0]
            for v in range(p): emin = min(emin,En[v],Em[v])
            for v in range(p):
                lo[v] = min(a,emin)-1.; hi[v] = b+1.
        for q in range(_gx.shape[0]):
            ok,x,wq = _graded_node(q,sg,f1,sp,T)
            if not ok: continue
            if side==0:
                i0 = _emit0(En,Em,D,p,x,degenerate,wq,kv,G,n,m,ws,acc,i0,
                        pt,pc,pb,dgrow,dmin,span,off)
            elif side==1: # A = S - (the part E_n > x of S, in B)
                wA += wq
                i0 = _emit0(En,lo,D,p,x,False,wq,kv,G,n,m,ws,acc,i0,
                        pt,pc,pb,dgrow,dmin,span,off)
            elif side==2: # -B = -S + (the part E_n < x of S, in A)
                wB += wq
                i0 = _emit0(En,hi,D,p,x,False,wq,kv,G,n,m,ws,acc,i0,
                        pt,pc,pb,dgrow,dmin,span,off)
            elif side==3: # A = S - (the part E_m < x of S, in B)
                wA += wq
                i0 = _emit0(hi,Em,D,p,x,False,wq,kv,G,n,m,ws,acc,i0,
                        pt,pc,pb,dgrow,dmin,span,off)
            else: # -B = -S + (the part E_m > x of S, in A)
                wB += wq
                i0 = _emit0(lo,Em,D,p,x,False,wq,kv,G,n,m,ws,acc,i0,
                        pt,pc,pb,dgrow,dmin,span,off)
    if degenerate: # A and B of a degenerate pair carry no Fermi surface
        if wA>0.:
            i0 = _emit0(En,Em,D,p,xA,True,wA,kv,G,n,m,ws,acc,i0,pt,pc,pb,
                    dgrow,dmin,span,off)
        if wB>0.:
            i0 = _emit0(En,Em,D,p,xB,True,wB,kv,G,n,m,ws,acc,i0,pt,pc,pb,
                    dgrow,dmin,span,off)
        return i0
    for v in range(p):
        lo[v] = -1.; hi[v] = 1.
    if wA>0.: # the whole simplex in A
        i0 = _emit0(lo,hi,D,p,0.,False,wA,kv,G,n,m,ws,acc,i0,pt,pc,pb,
                dgrow,dmin,span,off)
    if wB>0.: # and in B
        i0 = _emit0(hi,lo,D,p,0.,False,wB,kv,G,n,m,ws,acc,i0,pt,pc,pb,
                dgrow,dmin,span,off)
    return i0


@jit(nopython=True,cache=True)
def _complement(En,Em,p,x):
    """whether the region of a piece of x where only one band moves is
    tiled with fewer pieces as the whole simplex minus the rest: 0 when
    not, and otherwise which of the four cases of _emitT applies. With
    the other band on one side of x, the region is the part E < x or E >
    x of the moving band E, whose tiling has _NR[p-2,k] pieces for k
    vertices inside it, and the whole simplex adds nothing, since it is
    summed into one piece per pair"""
    nn = 0; nm = 0
    for i in range(p):
        if En[i]<x: nn += 1
        if Em[i]<x: nm += 1
    if nm==0 and nn>0 and nn<p: # A = the part E_n < x
        if _NR[p-2,p-nn]<_NR[p-2,nn]: return 1
    elif nm==p and nn>0 and nn<p: # B = the part E_n > x
        if _NR[p-2,nn]<_NR[p-2,p-nn]: return 2
    elif nn==p and nm>0 and nm<p: # A = the part E_m > x
        if _NR[p-2,nm]<_NR[p-2,p-nm]: return 3
    elif nn==0 and nm>0 and nm<p: # B = the part E_m < x
        if _NR[p-2,p-nm]<_NR[p-2,nm]: return 4
    return 0


@jit(nopython=True,cache=True)
def _resonance_fix(En,Em,D,p,omegas,delta,T,degenerate,tbreaks,kv,G,n,m,ws,
        acc,kn,kr,res,fix):
    """fix[iw] += the T=0 weights contracted with G, integrated over x on
    the knots of the frequency iw minus the same on the shared knots, for
    every frequency whose resonance D = w crosses an edge of the simplex.
    With delta < T the reference kernel adds the resonance to the knots
    (_knots with resonance), and the two lists only differ in the pieces
    of x that hold a resonance knot, so the shared pieces plus this
    difference are the reference result. kr holds the resonance knots"""
    if _constant_window(En,Em,p,tbreaks[-1]*T): return
    nw = omegas.shape[0]; nc = G.shape[3]
    dlo = D[0]; dhi = D[0]
    for i in range(1,p):
        dlo = min(dlo,D[i]); dhi = max(dhi,D[i])
    nkn = -1
    for iw in range(nw):
        w = omegas[iw]
        if not (w>dlo and w<dhi): continue # no sign change on an edge
        if nkn<0: nkn = _knots(En,Em,D,p,0.,T,tbreaks,False,kn)
        nr = _knots(En,Em,D,p,w,T,tbreaks,True,kr)
        for v in range(p): res[v,iw] = 0.
        changed = False
        j = 0 # kr is the merge of kn and the resonance knots, both sorted
        for i in range(nkn-1):
            a = kn[i]; b = kn[i+1]
            if b<=a: continue
            while j<nr and kr[j]<=a: j += 1
            if j>=nr or kr[j]>=b: continue # no resonance knot inside
            if _interval_kind(En,Em,p,a,b)!=-1: continue
            changed = True
            _interval_nodes(En,Em,D,p,a,b,omegas,iw,iw+1,delta,T,
                    degenerate,-1.,res,acc,ws)
            c0 = a
            while j<nr and kr[j]<b:
                if kr[j]>c0:
                    _interval_nodes(En,Em,D,p,c0,kr[j],omegas,iw,iw+1,delta,
                            T,degenerate,1.,res,acc,ws)
                    c0 = kr[j]
                j += 1
            _interval_nodes(En,Em,D,p,c0,b,omegas,iw,iw+1,delta,T,
                    degenerate,1.,res,acc,ws)
        if not changed: continue
        for c in range(nc):
            z = 0j
            for v in range(p): z += res[v,iw]*G[kv[v],n,m,c]
            fix[iw,c] += z


_group = 64 # consecutive simplices per group of the piece kernels
_maxblk = 256 # blocks of groups of the piece kernels


@jit(nopython=True,cache=True)
def _max_pieces(En,Em,D,p,T,degenerate,tbreaks,kn):
    """an upper bound to the number of pieces _emitT or _emit0 writes for
    one pair: the nodes of the quadrature in x on the pieces of x where
    the tiling moves, times the largest number of tiles of the two
    regions, 2 (p-1)^2, plus the whole simplex in A and in B"""
    if degenerate: return 0
    tiles = 2*(p-1)*(p-1)
    if T==0. or _constant_window(En,Em,p,tbreaks[-1]*T): return tiles
    nkn = _knots(En,Em,D,p,0.,T,tbreaks,False,kn)
    nodes = 0
    for i in range(nkn-1):
        if kn[i+1]>kn[i] and _interval_kind(En,Em,p,kn[i],kn[i+1])==-1:
            nodes += _gx.shape[0]
    # plus the whole simplex in A and in B, one piece each
    return nodes*tiles+2


@jit(nopython=True,parallel=True,cache=True)
def _bounds_jit(sim,it0,it1,E1,E2,T,tol,tbreaks,bound):
    """bound[i] >= the number of pieces of the simplex it0+i, without
    tiling it"""
    p = sim.shape[1]; nb = E1.shape[1]
    nsim = it1-it0
    L = tbreaks[-1]*T
    ngr = (nsim+_group-1)//_group
    nblk = min(ngr,_maxblk)
    for blk in prange(nblk):
        kn = np.zeros(4+2*p+p*(p-1)+2*tbreaks.shape[0])
        En = np.zeros(p); Em = np.zeros(p); D = np.zeros(p)
        for g in range(blk,ngr,nblk):
            for i in range(g*_group,min(nsim,(g+1)*_group)):
                n0 = 0
                for n in range(nb):
                    for m in range(nb):
                        active,degenerate = _pair_fields(sim,E1,E2,it0+i,n,
                                m,tol,L,En,Em,D)
                        if active:
                            n0 += _max_pieces(En,Em,D,p,T,degenerate,
                                    tbreaks,kn)
                bound[i] = n0


@jit(nopython=True,parallel=True,cache=True)
def _pieces_jit(sim,it0,it1,E1,E2,G,T,tol,tbreaks,offs,pt,pc,pb,dg,
        dmin,span,off,omegas,delta,fix):
    """the pieces of the simplices it0..it1-1, written from offs[i] on for
    the simplex it0+i; the slots left over keep pb = -1. dg[i] takes the
    Fermi surface terms of its degenerate pairs. With T > delta, fix[blk]
    takes the resonance corrections (_resonance_fix) of the block blk, and
    there are at most len(fix) blocks.
    The groups of simplices are dealt to the blocks in turn, so that every
    block gets its share of the simplices near the Fermi surface (prange
    splits the blocks statically), and the work arrays are allocated per
    block"""
    p = sim.shape[1]; nb = E1.shape[1]
    nsim = it1-it0
    L = tbreaks[-1]*T
    ngr = (nsim+_group-1)//_group
    nblk = min(ngr,fix.shape[0])
    resonant = T>0. and delta<T
    nw = omegas.shape[0]
    for blk in prange(nblk):
        ws,kn = _workspace(p,tbreaks.shape[0])
        kr = np.zeros(kn.shape[0])
        res = np.zeros((p,nw),dtype=np.complex128)
        En = np.zeros(p); Em = np.zeros(p); D = np.zeros(p)
        acc = np.zeros(p,dtype=np.complex128)
        for g in range(blk,ngr,nblk):
            for i in range(g*_group,min(nsim,(g+1)*_group)):
                it = it0+i
                kv = sim[it]
                n0 = offs[i]
                for n in range(nb):
                    for m in range(nb):
                        active,degenerate = _pair_fields(sim,E1,E2,it,n,m,
                                tol,L,En,Em,D)
                        if not active: continue
                        if T==0.:
                            n0 = _emit0(En,Em,D,p,0.,degenerate,1.,kv,G,n,m,
                                    ws,acc,n0,pt,pc,pb,dg[i],dmin,span,off)
                        else:
                            n0 = _emitT(En,Em,D,p,T,degenerate,tbreaks,kv,G,
                                    n,m,ws,acc,kn,n0,pt,pc,pb,dg[i],dmin,
                                    span,off)
                            if resonant:
                                _resonance_fix(En,Em,D,p,omegas,delta,T,
                                        degenerate,tbreaks,kv,G,n,m,ws,acc,
                                        kn,kr,res,fix[blk])


@jit(nopython=True,cache=True)
def _sort_by_bin(pb,nbins):
    """counting sort of the pieces by bin, the slots with pb = -1 left
    out: the order, and the start of every bin in it"""
    starts = np.zeros(nbins+1,dtype=np.int64)
    for i in range(pb.shape[0]):
        if pb[i]>=0: starts[pb[i]+1] += 1
    for b in range(nbins): starts[b+1] += starts[b]
    pos = starts[:-1].copy()
    order = np.zeros(starts[nbins],dtype=np.int64)
    for i in range(pb.shape[0]):
        if pb[i]<0: continue
        order[pos[pb[i]]] = i
        pos[pb[i]] += 1
    return order,starts


@jit(nopython=True,parallel=True,cache=True)
def _moments_jit(order,starts,occ,tasks,pt,pc,binc,binw,fk,mom):
    """mom[b,n] += the moments of the pieces of every occupied bin b, in
    units of its width, mom[b,n] = sum (p-1)! n!/(n+p)! h_n(y,y_s) c_s with
    y = (D - c)/b. The task it owns the occupied bins
    occ[tasks[it]:tasks[it+1]], about the same number of pieces each"""
    p = pt.shape[1]; nc = pc.shape[2]
    for it in prange(tasks.shape[0]-1):
        H = np.zeros(_nmom)
        y = np.zeros(p)
        for ib in range(tasks[it],tasks[it+1]):
            b = occ[ib]
            _bin_moments(order,starts[b],starts[b+1],pt,pc,binc[b],binw[b],
                    fk,H,y,mom[b])


@jit(nopython=True,cache=True)
def _bin_moments(order,n0,n1,pt,pc,c0,w,fk,H,y,mb):
    """mb[n] += the moments of the pieces order[n0:n1] around c0, in units
    of w. H and y are work space"""
    p = pt.shape[1]; nc = pc.shape[2]
    for idx in range(n0,n1):
        i = order[idx]
        for s in range(p): y[s] = (pt[i,s]-c0)/w
        H[0] = 1.
        for k in range(1,_nmom): H[k] = 0.
        for v in range(p): # h_n(y_0..y_v)
            for k in range(1,_nmom): H[k] += y[v]*H[k-1]
        for s in range(p): # one more variable, y_s
            hs = 1.
            for k in range(_nmom):
                if k>0: hs = H[k]+y[s]*hs
                f = fk[k]*hs
                for c in range(nc): mb[k,c] += f*pc[i,s,c]


@jit(nopython=True,parallel=True,cache=True)
def _near_jit(order,starts,pt,pc,binc,binw,off,dmin,span,omegas,delta,
        nchunk,part):
    """part[iw,ic] = the exact sum over the pieces of the bins near the
    frequency iw, the share ic of every such bin"""
    p = pt.shape[1]; nc = pc.shape[2]
    nw = omegas.shape[0]
    nlev = off.shape[0]-1
    for task in prange(nw*nchunk):
        # share-major, so that the static split of prange gives every
        # thread all the frequencies
        ic = task//nw; iw = task%nw
        w = omegas[iw]
        ws,kn = _workspace(p,0)
        t,tb,ib,pbuf,tt,dcol,a,h,d = ws[7]
        acc = np.zeros(nc,dtype=np.complex128)
        for L in range(nlev):
            nbl = off[L+1]-off[L]
            b = span*2.**(L-_lev)
            jc = (w-dmin)/b
            if jc<-10. or jc>nbl+10.: continue
            j0 = max(0,int(jc)-int(_far)-2)
            j1 = min(nbl-1,int(jc)+int(_far)+2)
            for j in range(j0,j1+1):
                fb = off[L]+j
                if not _is_near(binc[fb],binw[fb],w,delta): continue
                n0 = starts[fb]; n1 = starts[fb+1]
                for idx in range(n0+(ic*(n1-n0))//nchunk,
                                 n0+((ic+1)*(n1-n0))//nchunk):
                    i = order[idx]
                    for s in range(p): t[s] = pt[i,s]-w
                    _vertex_averages(t,p,-delta,_harm,_fact,h,d,tb,ib,pbuf,
                            tt,dcol,a)
                    for s in range(p):
                        g = -np.conj(a[s])
                        for c in range(nc): acc[c] += g*pc[i,s,c]
        for c in range(nc): part[iw,ic,c] = acc[c]


@jit(nopython=True,parallel=True,cache=True)
def _far_jit(occ,mom,binc,binw,omegas,delta,out):
    """out[iw] += the series of every occupied bin far from the frequency
    iw, minus its conjugate as in _add_pieces"""
    nc = mom.shape[2]
    for iw in prange(omegas.shape[0]):
        w = omegas[iw]
        acc = np.zeros(nc,dtype=np.complex128)
        for ib in range(occ.shape[0]):
            b = occ[ib]
            c0 = binc[b]; bw = binw[b]
            if _is_near(c0,bw,w,delta): continue
            ub = 1./complex(w-c0,-delta) # conj(1/(z - c))
            r = bw*ub
            ar = abs(r)
            pw = ub
            q = 1.
            for k in range(_nmom):
                for c in range(nc): acc[c] -= mom[b,k,c]*pw
                q *= ar
                if q<1e-18: break
                pw *= r
        for c in range(nc): out[iw,c] += acc[c]


def _bins(dmin,span):
    """offsets of the levels in the flat list of bins, and the center and
    the width of every bin"""
    nbl = [2**(_lev-L)+1 for L in range(_lev+1)]
    off = np.concatenate([[0],np.cumsum(nbl)]).astype(np.int64)
    binc = np.zeros(off[-1]); binw = np.zeros(off[-1])
    for L in range(_lev+1):
        b = span*2.**(L-_lev)
        j = np.arange(nbl[L])
        binc[off[L]:off[L+1]] = dmin+(j+0.5)*b
        binw[off[L]:off[L+1]] = b
    return off,binc,binw


def _static_factors(omegas,delta):
    """_static_factor at every frequency"""
    if delta>0.: return 1j*delta/(1j*delta-omegas)
    return (omegas==0.).astype(np.complex128)


def response_sources(dim,nk,E1,E2,G,omegas,delta,T):
    """sum over k and the pairs (n at k, m at k+q) of the simplex
    integration of G[k,n,m,c] (f_n - f_m)/(E_n - E_m - w + i delta) at
    zero chemical potential, normalized to the Brillouin zone, as an
    (nw,nc) array. G holds the matrix elements already averaged over the
    degenerate multiplets. At T > 0 the frequencies share the nodes of
    the quadrature in x; with delta < T the pieces of x that hold the
    resonance of a frequency are corrected for that frequency
    (_resonance_fix)"""
    import numba
    from ..dostk.simplexdos import simplices, _degeneracy_tol
    sim,frac = simplices(dim,nk)
    E1 = np.ascontiguousarray(E1,dtype=np.float64)
    E2 = np.ascontiguousarray(E2,dtype=np.float64)
    G = np.ascontiguousarray(G,dtype=np.complex128)
    omegas = np.ascontiguousarray(omegas,dtype=np.float64)
    tol = _degeneracy_tol(np.concatenate([E1,E2]))
    ns,p = sim.shape
    nw = len(omegas); nc = G.shape[3]
    delta = float(delta); T = float(T)
    dmin = float(np.min(E1)-np.max(E2)); dmax = float(np.max(E1)-np.min(E2))
    span = max(dmax-dmin,1e-8*max(1.,abs(dmin),abs(dmax)))
    off,binc,binw = _bins(dmin,span)
    nbins = len(binc)
    fk = np.array([np.prod(np.arange(1.,p))*np.prod(np.arange(1.,k+1))
            /np.prod(np.arange(1.,k+p+1)) for k in range(_nmom)])
    mom = np.zeros((nbins,_nmom,nc),dtype=np.complex128)
    occupied = np.zeros(nbins,dtype=np.bool_)
    parallel.set_num_threads()
    out = np.zeros((nw,nc),dtype=np.complex128)
    dgsum = np.zeros(nc,dtype=np.complex128)
    resonant = T>0. and delta<T
    # the resonance corrections, one row per block of the piece kernel, at
    # most 256 blocks and 2e8 bytes as for the pieces (the number of blocks
    # does not depend on the threads, so neither does the sum)
    nfix = _maxblk
    if resonant: nfix = max(1,min(_maxblk,int(2e8)//(nw*nc*16)))
    fix = np.zeros((nfix,nw,nc) if resonant else (nfix,0,nc),
            dtype=np.complex128)
    maxsrc = max(1024,min(_maxsrc,int(2e8)//(p*(8+16*nc)+8)))
    for s0 in range(0,ns,_maxsim):
        s1 = min(ns,s0+_maxsim)
        bound = np.zeros(s1-s0,dtype=np.int64)
        _bounds_jit(sim,s0,s1,E1,E2,T,tol,_tbreaks,bound)
        cum = np.cumsum(bound)
        i0 = 0
        while i0<len(bound): # passes of at most maxsrc slots
            base = cum[i0-1] if i0>0 else 0
            i1 = int(np.searchsorted(cum,base+maxsrc,side="right"))
            i1 = max(i1,i0+1)
            cnt = bound[i0:i1]
            npc = int(np.sum(cnt))
            offs = np.concatenate([[0],np.cumsum(cnt)[:-1]]).astype(np.int64)
            pt = np.empty((npc,p)); pc = np.empty((npc,p,nc),dtype=np.complex128)
            pb = np.full(npc,-1,dtype=np.int64)
            dg = np.zeros((i1-i0,nc),dtype=np.complex128)
            _pieces_jit(sim,s0+i0,s0+i1,E1,E2,G,T,tol,_tbreaks,offs,pt,pc,pb,
                    dg,dmin,span,off,omegas,delta,fix)
            dgsum += np.sum(dg,axis=0)
            order,starts = _sort_by_bin(pb,nbins)
            npc = len(order)
            if npc>0:
                occ = np.nonzero(starts[1:]>starts[:-1])[0].astype(np.int64)
                occupied[occ] = True
                cnt = np.cumsum(starts[occ+1]-starts[occ])
                ntask = min(len(occ),16*numba.get_num_threads())
                tasks = np.searchsorted(cnt,np.arange(ntask+1)*(cnt[-1]/ntask),
                        side="left")
                tasks[0] = 0; tasks[-1] = len(occ)
                tasks = np.maximum.accumulate(tasks).astype(np.int64)
                _moments_jit(order,starts,occ,tasks,pt,pc,binc,binw,fk,mom)
                nchunk = max(1,min(npc//256,4*numba.get_num_threads()))
                part = np.zeros((nw,nchunk,nc),dtype=np.complex128)
                _near_jit(order,starts,pt,pc,binc,binw,off,dmin,span,omegas,
                        delta,nchunk,part)
                out += np.sum(part,axis=1)
            i0 = i1
    _far_jit(np.nonzero(occupied)[0].astype(np.int64),mom,binc,binw,omegas,
            delta,out)
    out += dgsum[None,:]*_static_factors(omegas,delta)[:,None]
    if resonant: out += np.sum(fix,axis=0)
    return out*frac


# ------------------------------------------------------------ driver

_chunk = 2**15 # simplices per call of the weight kernel


@jit(nopython=True,cache=True)
def _scatter_add(sim,out,W):
    """W[k,n,m,iw] += out[it,n,m,v,iw] for k the vertex v of simplex it"""
    nb = W.shape[1]; nw = W.shape[3]; p = sim.shape[1]
    for it in range(sim.shape[0]):
        for v in range(p):
            k = sim[it,v]
            for n in range(nb):
                for m in range(nb):
                    for iw in range(nw): W[k,n,m,iw] += out[it,n,m,v,iw]


def pair_weights(dim,nk,E1,E2,omegas,delta,T):
    """Weights W[k,n,m,iw] of every pair (n at k, m at k+q) at every vertex
    of the uniform nk^dim mesh, from the simplex integration of
    (f_n - f_m)/(E_n - E_m - w + i delta) at zero chemical potential.
    Summed against the matrix elements M_nm(k) they give the bare
    response, normalized to the Brillouin zone"""
    from ..dostk.simplexdos import simplices, _degeneracy_tol
    sim,frac = simplices(dim,nk)
    E1 = np.ascontiguousarray(E1,dtype=np.float64)
    E2 = np.ascontiguousarray(E2,dtype=np.float64)
    omegas = np.ascontiguousarray(omegas,dtype=np.float64)
    nkk,nb = E1.shape
    nw = len(omegas)
    tol = _degeneracy_tol(np.concatenate([E1,E2]))
    parallel.set_num_threads()
    W = np.zeros((nkk,nb,nb,nw),dtype=np.complex128)
    p = sim.shape[1]
    chunk = max(1,min(_chunk,2**23//(nb*nb*p*nw+1)))
    for i0 in range(0,len(sim),chunk):
        s = sim[i0:i0+chunk]
        out = np.zeros((len(s),nb,nb,p,nw),dtype=np.complex128)
        _weights_jit(s,E1,E2,omegas,float(delta),float(T),tol,_tbreaks,
                out)
        # a serial scatter costs the size of the chunk. A bincount over
        # the whole W per chunk (the code before) cost the size of W at
        # every chunk; with W kept below max_memory_gb by blocks of
        # frequencies, the number of those passes grew as nk^(2 dim), and
        # it took most of the time at nk = 1024 and 2048 in 2D
        _scatter_add(s,out,W)
    W = W*frac
    _multiplet_average_pairs(E1,E2,W,tol)
    return W


def check_hamiltonian(h):
    if h.dimensionality not in (1,2,3):
        raise ValueError("integration='triangle' of the response function "
                "needs a periodic Hamiltonian of dimensionality 1, 2 or 3, "
                "got "+str(h.dimensionality))
    if getattr(h,"non_hermitian",False):
        raise NotImplementedError("integration='triangle' of the response "
                "function needs a Hermitian Hamiltonian")


max_memory_gb = 2. # eigenvectors of both meshes kept above this are
                   # recomputed instead, at the cost of a second pass


def _eigen(hkgen,ks,batch,vectors):
    from ..dostk.simplexdos import _hk_batch
    from ..htk.eigenvectors import parallel_diagonalization, peigvalsh
    parallel.set_num_threads()
    for i0 in range(0,len(ks),batch):
        hks = _hk_batch(hkgen,ks[i0:i0+batch])
        if vectors: yield i0,parallel_diagonalization(hks)
        else: yield i0,(peigvalsh(hks),None)


mesh_q_tol = 1e-12 # q*nk within this of integers, relative to the size of
# q*nk, is a q on the mesh. The rounding of a q of klist.kmesh, i/nk from a
# linspace, is a few machine epsilons of q*nk, so 1e-12 takes every q of
# the mesh and leaves the snap of a q off the mesh by eps (a change of
# chi of order eps) at the level of the rounding of the sum itself


def mesh_shift(q,dim,nk):
    """The index map of a q on the uniform nk^dim mesh, or None.

    For q with every component times nk an integer to mesh_q_tol (relative
    to the largest component of q*nk, one at least), k+q is
    the mesh point (k+q) mod 1, whose Bloch Hamiltonian is the one at k+q
    up to the phases exp(2 pi i G.R) = 1, so the bands at k+q are those of
    the mesh, reindexed. Returns perm with ks[perm] = (ks+q) mod 1 for ks
    the row-major uniform_mesh(dim,nk), or None for a q off the mesh"""
    qv = np.zeros(3)
    if q is not None:
        qa = np.array(q,dtype=np.float64).ravel()
        qv[:len(qa)] = qa[:3]
    if np.any(np.abs(qv[dim:])>mesh_q_tol): return None
    s = qv[:dim]*nk
    si = np.rint(s)
    tol = mesh_q_tol*max(1.,np.max(np.abs(s)))
    if np.max(np.abs(s-si))>tol: return None
    idx = np.arange(nk**dim).reshape((nk,)*dim)
    shift = tuple(-int(x) for x in si) # np.roll(a,-s)[i] = a[i+s]
    return np.roll(idx,shift,axis=tuple(range(dim))).ravel()


def _hoppings(h):
    """the Bloch hoppings of h, {lattice vector: dense copy}"""
    from .. import algebra
    return {d: np.array(algebra.todense(m),dtype=np.complex128)
            for d,m in h.get_multihopping().get_dict().items()}


def _same_hoppings(a,b):
    """whether two _hoppings dicts are the same, entry by entry"""
    return a.keys()==b.keys() and all(np.array_equal(a[d],b[d]) for d in a)


class MeshEigen:
    """The eigenpairs of h on the uniform nk^dim mesh, diagonalized once on
    first use and shared by every q of the mesh: the band energies always,
    the eigenvectors when one copy of them fits max_memory_gb. chiAB with
    q=None and integration="triangle" passes one to every chi_triangle of
    its q-mesh, and chi_triangle builds its own for a single q on the
    mesh. ndiag counts the diagonalizations it has done.

    It belongs to one Hamiltonian object: check(h, nk) refuses another
    object, even one with the same shape, and the same object once its
    hoppings changed after the diagonalization"""
    def __init__(self,h,nk):
        self.h = h
        self.nk = nk
        self.E = None # (nk^dim, n) band energies
        self.U = None # (nk^dim, n, n) eigenvectors as columns, or None
        self.hoppings = None # the hoppings of h when diagonalized
        self.ndiag = 0
    def check(self,h,nk):
        """Raise ValueError unless this is the mesh of h on nk^dim"""
        if h is not self.h:
            raise ValueError("eigen must be a MeshEigen of the Hamiltonian "
                    "of this call, h, built as MeshEigen(h,nk); got one of "
                    "another Hamiltonian object")
        if nk!=self.nk:
            raise ValueError("eigen must hold the mesh of this call, nk="
                    +str(nk)+", got nk="+str(self.nk))
        if self.hoppings is not None and \
                not _same_hoppings(self.hoppings,_hoppings(h)):
            raise ValueError("eigen must hold the bands of h as it is now; "
                    "h was changed in place after its MeshEigen "
                    "diagonalized the mesh, so build a new MeshEigen(h,nk)")
    def get(self,batch):
        """(E, U), diagonalizing the mesh on the first call"""
        if self.E is None:
            from ..dostk.simplexdos import uniform_mesh
            self.hoppings = _hoppings(self.h)
            h = self.h
            if h.is_sparse: h = h.get_dense()
            ks = uniform_mesh(h.dimensionality,self.nk)
            n = h.intra.shape[0]
            keep = len(ks)*n*n*16 <= max_memory_gb*1e9 # one copy
            E = np.zeros((len(ks),n))
            U = np.zeros((len(ks),n,n),dtype=np.complex128) if keep else None
            for i0,(e,u) in _eigen(h.get_hk_gen(),ks,batch,keep):
                E[i0:i0+len(e)] = e
                if keep: U[i0:i0+len(e)] = u
            self.E,self.U = E,U
            self.ndiag += len(ks)
        return self.E,self.U


def _nonzero_rows(ops,nz,idx):
    """The operators ops[idx] restricted to their nonzero rows, nz[i] the
    mask of the nonzero rows of ops[i]: rows[c] holds the row indices of
    ops[idx[c]] and sub[c] = ops[idx[c]][rows[c]], padded with zero rows
    to a common count, one at least"""
    r = max(1,int(np.max(nz[idx].sum(axis=1),initial=0)))
    rows = np.zeros((len(idx),r),dtype=np.int64)
    sub = np.zeros((len(idx),r,ops.shape[2]),dtype=np.complex128)
    for c,i in enumerate(idx):
        x = np.flatnonzero(nz[i])
        rows[c,:len(x)] = x
        sub[c,:len(x)] = ops[i][x]
    return rows,sub


def _sandwich(U,rows,sub,V):
    """U^dag A_c V as [k,c,n,m], for U and V of shape (k,n,n) with the
    eigenvectors as columns and the operators A_c given by their nonzero
    rows (_nonzero_rows). A site projector times a local operator has only
    the rows of its site, r of them, so the product costs 2 r n^2 per k
    and operator instead of 2 n^3, and 2 n^3 summed over the sites. The
    padding rows are zero and add nothing"""
    Ur = U[:,rows,:] # (k,c,r,n), the rows of every operator
    return np.conj(np.transpose(Ur,(0,1,3,2))) @ (sub[None] @ V[:,None])


def _chi_simplices(dim,nk,E1,E2,energies,delta,T,elements,diag,ni,nj):
    """the weights of every vertex at every frequency (pair_weights), then
    contracted with the matrix elements. Returns the response and the
    number of passes over the matrix elements"""
    nkk,n = E1.shape
    nw = len(energies)
    out = np.zeros((nw,ni) if diag else (nw,ni,nj),dtype=np.complex128)
    wblock = max(1,int(max_memory_gb*1e9/4)//(nkk*n*n*16)) # frequencies
    npass = 0
    for w0 in range(0,nw,wblock):
        ew = energies[w0:w0+wblock]
        W = pair_weights(dim,nk,E1,E2,ew,delta,T) # (k,n,m,w)
        npass += 1
        for i0,MA,MB in elements():
            kb = len(MA)
            Wk = W[i0:i0+kb].reshape(kb,n*n,len(ew))
            if diag:
                out[w0:w0+wblock] += np.einsum("kpw,kip,kip->wi",Wk,MA,MB,
                        optimize=True)
            else:
                out[w0:w0+wblock] += np.einsum("kpw,kip,kjp->wij",Wk,MA,MB,
                        optimize=True)
    return out,npass


def _chi_sources(dim,nk,E1,E2,energies,delta,T,elements,diag,ni,nj):
    """the matrix elements contracted first, G[k,n,m,c] for the components
    c = i (diag) or (i,j), then the source kernel (response_sources), by
    blocks of components that keep G, and the moments of the bins, below
    max_memory_gb/4. The pieces are rebuilt for every block, and the
    matrix elements of the operators of the block only. Returns the
    response and the number of passes over the matrix elements"""
    from ..dostk.simplexdos import _degeneracy_tol
    nkk,n = E1.shape
    nw = len(energies)
    comps = [(i,i) for i in range(ni)] if diag else [(i,j)
            for i in range(ni) for j in range(nj)]
    tol = _degeneracy_tol(np.concatenate([E1,E2]))
    nbins = sum(2**(_lev-L)+1 for L in range(_lev+1)) # as in _bins
    cblock = max(1,int(max_memory_gb*1e9/4)//max(nkk*n*n*16,
            nbins*_nmom*16))
    out = np.zeros((nw,len(comps)),dtype=np.complex128)
    npass = 0
    for c0 in range(0,len(comps),cblock):
        cs = comps[c0:c0+cblock]
        ia = np.array([c[0] for c in cs]); ib = np.array([c[1] for c in cs])
        ua,ja = np.unique(ia,return_inverse=True) # the operators of the block
        ub,jb = np.unique(ib,return_inverse=True)
        G = np.zeros((nkk,n*n,len(cs)),dtype=np.complex128)
        for i0,MA,MB in elements(ua,ub):
            G[i0:i0+len(MA)] = np.transpose(MA[:,ja]*MB[:,jb],(0,2,1))
        npass += 1
        G = G.reshape(nkk,n,n,len(cs))
        # the multiplet average of the weights, moved to G
        _multiplet_average_pairs(E1,E2,G,tol)
        out[:,c0:c0+len(cs)] = response_sources(dim,nk,E1,E2,G,energies,
                delta,T)
        del G
    if not diag: out = out.reshape(nw,ni,nj)
    return out,npass


_kernels = {
    "sources": _chi_sources, # the pieces built once, every frequency a sum
    "simplices": _chi_simplices, # every piece at every frequency
}
default_kernel = "sources"


def chi_triangle(h,q=None,energies=np.linspace(-3.,3.,100),nk=60,
        pAs=None,pBs=None,delta=0.1,T=0.,mode="matrix",counted=False,
        kernel=None,eigen=None,reuse_mesh=True):
    """Bare response of chitk.chiAB.chiAB_q with the linear simplex
    integration of the k-sum on the uniform nk^dim mesh, for the operators
    pAs[i] and pBs[j] (the site projectors times A and B). Returns
    (energies, out) in the layout of chiAB_q for mode="matrix", "trace"
    and "diagonal". delta = 0 and T = 0 are allowed and are the exact
    limits. With counted=True also returns the number of
    diagonalizations done by this call. kernel picks the evaluation of the
    simplex sum, one of _kernels, default_kernel when None; the two agree
    to rounding.

    A q on the mesh (q = None, q = 0, every component times nk an integer)
    takes the bands at k+q from those at k, an index map of the mesh
    (mesh_shift), so the mesh is diagonalized once, nk^dim times instead
    of 2 nk^dim. eigen, a MeshEigen of the same h object and nk, shares
    that diagonalization between calls, as chiAB does over the q of the
    local response; a MeshEigen of another Hamiltonian, or of this one
    changed in place since, raises ValueError (MeshEigen.check). reuse_mesh=False diagonalizes k+q anyway, the route of a
    generic q, kept as the reference of the tests"""
    check_hamiltonian(h)
    if not delta>=0.:
        raise ValueError("integration='triangle' of the response function "
                "needs a broadening delta >= 0 (delta=0 is the exact limit), "
                "got delta="+str(delta))
    modes = ("matrix","trace","diagonal")
    if mode not in modes:
        raise ValueError("unknown mode '"+str(mode)+"' of the response "
                "function; integration='triangle' accepts "+str(list(modes)))
    T = 0. if T is None else abs(float(T))
    energies = np.atleast_1d(np.asarray(energies,dtype=np.float64))
    if eigen is not None: eigen.check(h,nk) # the mesh of this h only
    if h.is_sparse: h = h.get_dense()
    dim = h.dimensionality
    from ..dostk.simplexdos import uniform_mesh
    ks = uniform_mesh(dim,nk)
    qv = np.zeros(3)
    if q is not None:
        qa = np.array(q,dtype=np.float64).ravel()
        qv[:len(qa)] = qa[:3]
    kq = ks + qv[None,:]
    hkgen = h.get_hk_gen()
    pAs = np.array([np.asarray(a,dtype=np.complex128) for a in pAs])
    pBs = np.array([np.asarray(b,dtype=np.complex128) for b in pBs])
    n = h.intra.shape[0]
    ni,nj = len(pAs),len(pBs)
    nw = len(energies)
    keep = 2*len(ks)*n*n*16 <= max_memory_gb*1e9
    batch = max(1,min(2**14,2**22//(n*n*max(ni,nj)+n*n*nw+1)))
    diag = mode!="matrix"
    if diag and ni!=nj:
        raise ValueError("mode='"+mode+"' needs as many A as B operators, "
                "got "+str(ni)+" and "+str(nj))
    perm = mesh_shift(q,dim,nk) if reuse_mesh else None
    if perm is not None: # a q on the mesh: k+q is a mesh point
        if eigen is None: eigen = MeshEigen(h,nk)
        ndiag0 = eigen.ndiag
        E1,U1 = eigen.get(batch)
        E2 = E1[perm]
        ndiag = eigen.ndiag - ndiag0
        keep = U1 is not None
        identity = np.array_equal(perm,np.arange(len(ks)))
        def vectors(): # (i0, U at k, U at k+q) per batch
            if keep:
                for i0 in range(0,len(ks),batch):
                    yield i0,U1[i0:i0+batch],U1[perm[i0:i0+batch]]
            elif identity:
                for i0,(_,u1) in _eigen(hkgen,ks,batch,True):
                    yield i0,u1,u1
            else:
                for (i0,(_,u1)),(_,(_,u2)) in zip(
                        _eigen(hkgen,ks,batch,True),
                        _eigen(hkgen,ks[perm],batch,True)):
                    yield i0,u1,u2
        npair = 1 if identity else 2 # diagonalizations per k and pass
    else:
        # band energies (and eigenvectors) at k and k+q
        E1 = np.zeros((len(ks),n)); E2 = np.zeros((len(ks),n))
        U1s,U2s = [],[]
        for (i0,(e1,u1)),(_,(e2,u2)) in zip(_eigen(hkgen,ks,batch,keep),
                                            _eigen(hkgen,kq,batch,keep)):
            E1[i0:i0+len(e1)] = e1; E2[i0:i0+len(e2)] = e2
            if keep: U1s.append(u1); U2s.append(u2)
        ndiag = 2*len(ks)
        def vectors(): # (i0, U at k, U at k+q) per batch
            if keep:
                for i0,u1,u2 in zip(range(0,len(ks),batch),U1s,U2s):
                    yield i0,u1,u2
            else:
                for (i0,(e1,u1)),(_,(e2,u2)) in zip(
                        _eigen(hkgen,ks,batch,True),
                        _eigen(hkgen,kq,batch,True)):
                    yield i0,u1,u2
        npair = 2
    nzA = np.any(pAs!=0.,axis=2) # the nonzero rows of every operator
    nzB = np.any(pBs!=0.,axis=2)
    def elements(ia=None,ib=None):
        """<n,k|A_i|m,k+q> as [k,i,nm] and <m,k+q|B_j|n,k> as [k,j,nm] per
        batch of k, for the operators ia and ib, all of them when None"""
        ia = np.arange(ni) if ia is None else ia
        ib = np.arange(nj) if ib is None else ib
        rA,sA = _nonzero_rows(pAs,nzA,ia)
        rB,sB = _nonzero_rows(pBs,nzB,ib)
        for i0,U1,U2 in vectors():
            kb = len(U1)
            MA = _sandwich(U1,rA,sA,U2).reshape(kb,len(ia),n*n)
            MB = _sandwich(U2,rB,sB,U1) # [k,j,m,n]
            MB = np.transpose(MB,(0,1,3,2)).reshape(kb,len(ib),n*n)
            yield i0,MA,MB
    def traced(ia=None,ib=None):
        """the mean over the sites i of the products M^ii_nm as [k,0,nm],
        the one component of mode="trace", against B = 1, in the layout of
        elements. A kernel can only ask for that component, so ia and ib
        are not read"""
        for i0,MA,MB in elements():
            P = np.mean(MA*MB,axis=1,keepdims=True)
            yield i0,P,np.ones_like(P)
    if kernel is None: kernel = default_kernel
    if kernel not in _kernels:
        raise ValueError("unknown kernel "+repr(kernel)+" of the triangle "
                "response, accepted values are "+", ".join(_kernels))
    if mode=="trace":
        # the kernels are linear in the products of the matrix elements,
        # so the mean over the sites goes before them: one component in
        # place of one per site, and in the source kernel one pass over
        # the matrix elements in place of one per block of sites
        out,npass = _kernels[kernel](dim,nk,E1,E2,energies,delta,T,traced,
                True,1,1)
        out = out[:,0]
    else:
        out,npass = _kernels[kernel](dim,nk,E1,E2,energies,delta,T,
                elements,diag,ni,nj)
    if not keep: ndiag += npair*len(ks)*npass
    if counted: return energies,out,ndiag
    return energies,out
