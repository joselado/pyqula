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
# piece is done by Gauss-Legendre quadrature in the variable f(x). A pair
# whose two bands keep the same side of the window on every vertex has an
# x-independent T=0 response there, which is evaluated once.
#
# The weights are summed over the simplices that share each vertex, and
# averaged over the degenerate multiplets of n at k and of m at k+q, which
# makes the result independent of the basis eigh picks inside a multiplet.
# The bands are ordered by energy at each k, so two bands that cross are
# interpolated with a kink, as in the simplex DOS.
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
def _quadrature(en,em,D,p,omegas,iw0,iw1,delta,T,degenerate,tbreaks,
        resonance,out,acc,ws,kn):
    """the convolution of the T=0 weights with -f'(x) for the frequencies
    omegas[iw0:iw1], on the breakpoints of the frequency omegas[iw0]"""
    m = _knots(en,em,D,p,omegas[iw0],T,tbreaks,resonance,kn)
    for i in range(m-1):
        a = kn[i]; b = kn[i+1]
        if b<=a: continue
        if b<=0.: # mirror: x = -u, u in [-b,-a]
            sg = -1.; u0 = -b; u1 = -a
        else:
            sg = 1.; u0 = a; u1 = b
        f0 = _fermi(u0,T); f1 = _fermi(u1,T)
        span = f0-f1
        if span<=0.: continue
        for q in range(_gx.shape[0]):
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
            if phi<=0. or phi>=1.: continue
            x = sg*T*(np.log1p(-phi)-np.log(phi))
            _kernel0(en,em,D,p,x,omegas,iw0,iw1,delta,degenerate,
                    0.5*_gw[q]*span*dg,out,acc,ws)


@jit(nopython=True,cache=True)
def _kernelT(en,em,D,p,omegas,delta,T,degenerate,tbreaks,out,acc,ws,kn):
    """the weights at temperature T, every frequency, added to out[:,iw].
    When delta < T the breakpoints include the resonance of each frequency
    and the quadrature is done frequency by frequency; otherwise the
    features in x are at least T wide and the frequencies share the
    nodes"""
    nw = omegas.shape[0]
    L = tbreaks[-1]*T
    const = True # does every band keep one side of the window?
    for b in range(2):
        e = en if b==0 else em
        below = 0; above = 0
        for i in range(p):
            if e[i]<=-L: below += 1
            elif e[i]>=L: above += 1
        if not (below==p or above==p): const = False
    if const:
        _kernel0(en,em,D,p,0.,omegas,0,nw,delta,degenerate,1.,out,acc,ws)
        return
    if delta<T:
        for iw in range(nw):
            _quadrature(en,em,D,p,omegas,iw,iw+1,delta,T,degenerate,tbreaks,
                    True,out,acc,ws,kn)
    else:
        _quadrature(en,em,D,p,omegas,0,nw,delta,T,degenerate,tbreaks,
                False,out,acc,ws,kn)


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
            nmin = 1e300; nmax = -1e300
            for v in range(p):
                En[v] = E1[sim[it,v],n]
                nmin = min(nmin,En[v]); nmax = max(nmax,En[v])
            for m in range(nb):
                mmin = 1e300; mmax = -1e300
                dmax = 0.
                for v in range(p):
                    Em[v] = E2[sim[it,v],m]
                    D[v] = En[v]-Em[v]
                    mmin = min(mmin,Em[v]); mmax = max(mmax,Em[v])
                    dmax = max(dmax,abs(D[v]))
                degenerate = dmax<tol
                # skip a pair with an empty A and B at every chemical
                # potential of the window [-L,L]
                if degenerate:
                    if nmin>=L or nmax<=-L: continue
                else: # A needs nmin < x < mmax, B mmin < x < nmax
                    a = nmin<L and mmax>-L and nmin<mmax
                    b = mmin<L and nmax>-L and mmin<nmax
                    if not (a or b): continue
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


# ------------------------------------------------------------ driver

_chunk = 2**15 # simplices per call of the weight kernel


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
    W = np.zeros(nkk*nb*nb*nw,dtype=np.complex128)
    p = sim.shape[1]
    chunk = max(1,min(_chunk,2**23//(nb*nb*p*nw+1)))
    for i0 in range(0,len(sim),chunk):
        s = sim[i0:i0+chunk]
        out = np.zeros((len(s),nb,nb,p,nw),dtype=np.complex128)
        _weights_jit(s,E1,E2,omegas,float(delta),float(T),tol,_tbreaks,
                out)
        # flat index of (vertex k, n, m, iw)
        idx = (((s[:,None,None,:,None]*nb + np.arange(nb)[None,:,None,None,None])
                *nb + np.arange(nb)[None,None,:,None,None])*nw
                + np.arange(nw)[None,None,None,None,:])
        idx = np.broadcast_to(idx,out.shape).ravel()
        W += np.bincount(idx,weights=out.real.ravel(),minlength=len(W))
        W += 1j*np.bincount(idx,weights=out.imag.ravel(),minlength=len(W))
    W = W.reshape((nkk,nb,nb,nw))*frac
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


def chi_triangle(h,q=None,energies=np.linspace(-3.,3.,100),nk=60,
        pAs=None,pBs=None,delta=0.1,T=0.,mode="matrix",counted=False):
    """Bare response of chitk.chiAB.chiAB_q with the linear simplex
    integration of the k-sum on the uniform nk^dim mesh, for the operators
    pAs[i] and pBs[j] (the site projectors times A and B). Returns
    (energies, out) in the layout of chiAB_q for mode="matrix", "trace"
    and "diagonal". delta = 0 and T = 0 are allowed and are the exact
    limits. With counted=True also returns the number of
    diagonalizations"""
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
                    _eigen(hkgen,ks,batch,True),_eigen(hkgen,kq,batch,True)):
                yield i0,u1,u2
    out = np.zeros((nw,ni) if diag else (nw,ni,nj),dtype=np.complex128)
    wblock = max(1,int(max_memory_gb*1e9/4)//(len(ks)*n*n*16)) # frequencies
    for w0 in range(0,nw,wblock):
        ew = energies[w0:w0+wblock]
        W = pair_weights(dim,nk,E1,E2,ew,delta,T) # (k,n,m,w)
        if not keep: ndiag += 2*len(ks)
        for i0,U1,U2 in vectors():
            kb = len(U1)
            Wk = W[i0:i0+kb].reshape(kb,n*n,len(ew))
            # <n,k|A_i|m,k+q> as [k,i,n,m], <m,k+q|B_j|n,k> as [k,j,n,m]
            MA = np.conj(np.transpose(U1,(0,2,1)))[:,None] @ (
                    pAs[None] @ U2[:,None])
            MB = np.conj(np.transpose(U2,(0,2,1)))[:,None] @ (
                    pBs[None] @ U1[:,None])
            MA = MA.reshape(kb,ni,n*n)
            MB = np.transpose(MB,(0,1,3,2)).reshape(kb,nj,n*n)
            if diag:
                out[w0:w0+wblock] += np.einsum("kpw,kip,kip->wi",Wk,MA,MB,
                        optimize=True)
            else:
                out[w0:w0+wblock] += np.einsum("kpw,kip,kjp->wij",Wk,MA,MB,
                        optimize=True)
    if mode=="trace": out = np.mean(out,axis=1)
    if counted: return energies,out,ndiag
    return energies,out
