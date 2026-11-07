# Density matrix of a crystal from the linear simplex (triangle in 2D,
# tetrahedron in 3D, segment in 1D) integration of the occupations, for the
# self-consistent mean field (scftk.densitydensity.get_dm, integration=
# "triangle").
#
# The density matrix the mean field reads at a lattice vector d is
#   dm[d]_ij = int dk sum_n f(E_n(k)) e^{2 pi i k.d} conj(u_n,i(k)) u_n,j(k)
# over the Brillouin zone normalized to one. On a uniform mesh this is the
# mean over the k-points, which converges exponentially for a smooth
# integrand (an insulator), but in a metal at low temperature the Fermi
# step needs a mesh with nk of order v_F/T. Here the band energies E_n(k)
# and the field X_n(k) = e^{2 pi i k.d} conj(u_n,i) u_n,j are interpolated
# linearly inside each simplex of the uniform mesh, and the step is
# integrated exactly. Since X is linear inside the simplex, the result is a
# set of scalar vertex weights,
#   int_T theta(mu - E_L(k)) X_L(k) dk = sum_j w_j X_j,
#   w_j = int_T theta(mu - E_L(k)) lambda_j(k) dk,
# with lambda_j the barycentric coordinates, so the density matrix is the
# mean-field kernel of exact diagonalization with the Fermi occupation of
# each state replaced by w_n(k) summed over the simplices that share k. This
# is the linear tetrahedron method (P. E. Bloechl, O. Jepsen and O. K.
# Andersen, PRB 49, 16223 (1994)). The weights come from a decomposition of
# the occupied part of each simplex into sub-simplices whose vertices are
# the simplex vertices and the points where E_L = mu on its edges. Every
# denominator there is e_j - e_i with e_i <= mu < e_j, so it never
# vanishes, also for degenerate vertex energies.
#
# Four points decide how this is used.
#
# 1. The Bloch phase. e^{2 pi i k.d} is not linear in k. We keep it inside
# the interpolated field (the phase is evaluated at the vertices and
# interpolated with the projector). The alternative is to interpolate the
# phase-free projector and integrate the phase exactly. In a simplex that
# is fully occupied, the weights of the first choice are 1/nk^dim per
# vertex once summed over the simplices that share it, so the density
# matrix of a fully occupied band is exactly the uniform-mesh mean, which
# is exponentially accurate. The exact phase instead integrates the
# linearly interpolated projector times the exact phase, which has an
# error of second order in 1/nk in every simplex, occupied or not. So the
# phase in the field is exact in the Fermi sea, and both are of second
# order on the simplices cut by the Fermi surface. We measured both for
# the first-neighbor vectors of a honeycomb metal at T=0, the exact phase
# being emulated by sampling the linear fields on an m times finer mesh
# (future_development/tetrahedron_density_matrix.md). The largest error
# over the entries was the same within 7% at every nk from 24 to 192,
# 1.35e-5 with the phase in the field and 1.40e-5 with m = 8 at nk = 192.
# So the exact phase buys nothing, and only the phase in the field is
# built. As a result an insulator at T=0 gives exactly the density matrix
# of exact diagonalization on the same mesh.
#
# 2. Finite temperature. The Fermi function is integrated over the linear
# simplex through its density of states: the weight at temperature T is
# the T=0 weight convolved with -f'(x),
#   w_j(mu,T) = int dx (-f'(x)) w_j(mu + x, T=0).
# w_j(mu,0) is a polynomial in mu between two vertex energies, so the
# integral is split at the vertex energies, at mu and at a few multiples
# of T, and each piece is done by Gauss-Legendre quadrature in the variable
# f(x), in which -f'(x) dx is flat. Above the highest vertex energy the
# weight is constant and that piece is exact, f(e_max - mu)/(dim+1). The
# quadrature error, against an adaptive quadrature, is below 3e-9 in
# units of the simplex volume for T from 1e-4 to 0.3 of the energy spread
# of the simplex, with 8 nodes per piece. T=0 is the step itself.
# This is not the uniform mesh at temperature T: the mesh only has to
# resolve the bands, not the Fermi function, so the same mesh serves every
# T, T=0 included.
#
# 3. The Fermi level. The number of electrons N(mu) = sum_T C_T(mu), with
# C_T the occupied volume of each simplex, is continuous and monotonous in
# mu also at T=0 (a piecewise polynomial), unlike the eigenvalue count of
# a uniform mesh, which is a staircase. So the Fermi level for a filling
# is located on the same simplices as the density matrix (get_fermi4filling
# below) and the density matrix holds the requested charge to 1e-12, with
# no partly occupied level. When the requested charge is held on a whole
# interval of mu (a gap at T=0, or at T>0 a gap wider than 2 _tcut T, since
# the Fermi function is cut there), the middle of the gap is returned. The
# search is a safeguarded Newton iteration with the simplex DOS as the
# derivative (fermi_from_energies), three or four charge evaluations in a
# metal (future_development/simplex_fermi_search.md).
# The band energy is the sum of w_n(k) E_n(k) with the same weights at T=0
# (the exact band energy of the linear bands, plus the Bloechl term of
# point 4), so the total energy of the mean field is summed with the same
# construction.
#
# 4. The Bloechl correction. Bloechl, Jepsen and Andersen add to the weights
#   dw_j = sum_{T containing j} D_T(mu) c_dim sum_{i in T} (e_i - e_j),
# with D_T the density of states of the simplex at mu and c_3 = 1/40. The
# derivation gives c_dim = 1/(2 (dim+1) (dim+2)): the mean over a simplex
# of the error E_L - E of the linear interpolation of a quadratic band is
# sum_{i<j} e_ij.A.e_ij/(2 (dim+1) (dim+2)), with e_ij the edges and A the
# Hessian, and the sum over the simplices that share a vertex turns the
# edge terms into e_i - e_j, because the star of a vertex is
# centrosymmetric on this mesh (in 2D and in 3D). So c_2 = 1/24 and
# c_1 = 1/12. The correction is a scalar weight per band, vertex and
# simplex, so it applies unchanged to a matrix-valued field such as the
# projector with its Bloch phase, entry by entry. What it can correct
# follows from sum_j dw_j = 0 in every simplex: it does not change the
# charge of any band, and so it does not move the Fermi level either. It
# acts only through the variation of the field along the band, as
# -c_dim D_T sum_{ij} (X_i - X_j)(e_i - e_j)/2. In particular, on a model
# where each band carries one fixed orbital and spin (a one-orbital
# Hubbard ferromagnet that reads only the onsite density matrix), it is
# exactly zero on the mean field and only changes the band energy. For
# X = E, the band energy, it removes the leading 1/nk^2 error: on a curved
# square-lattice band at nk = 64 the error goes from 6e-4 to 2e-7. It is
# not variational, there is no total energy whose stationary point is the
# corrected mean field, so we checked that the loop reaches the same state:
# the measurements in future_development/tetrahedron_density_matrix.md
# show the corrected and uncorrected mean fields converging to the same
# limit, the bond order of a V1 metal to 2e-9 at nk = 128. It improved the
# bond order by a factor 300 at nk = 32 there, the total energy by a
# factor 100 to 4000, and left the magnetization of a ferromagnet
# unchanged, as it has to. So it is on by default (bloechl_default).
#
# The weights are averaged over degenerate multiplets at each k before the
# contraction, which is the same as averaging the projector over the
# multiplet, as in dostk/simplexdos.py, so the result does not depend on
# the basis that eigh picks inside a multiplet. As in the simplex DOS, the
# bands are ordered by energy at each k, so two bands that cross are
# interpolated with a kink, which costs an error of order one in a window
# of width of order v/nk around the crossing.
#
# For a Bogoliubov-de Gennes (Nambu) Hamiltonian the triangle reduces to
# the step density matrix of exact diagonalization on the same mesh. The
# reason is that the spectrum is symmetric, E and -E, and the bands are
# sorted at each k, so the lower half of the bands is below zero at every
# k and the upper half above: no simplex is ever partly occupied. In a
# gapped superconductor this is the exact answer on the mesh. In a state
# without a gap it is the staircase of exact diagonalization, and the
# continuous charge of the triangles is lost. So the Fermi level of a BdG
# Hamiltonian is counted as exact diagonalization counts it, on the normal
# state (the electron block) of the same mesh (spectrum.get_fermi4filling
# at T=0, since a vertex on a level is occupied by half, as a step and not
# as a Fermi-Dirac weight), and not with the triangles: the triangle count of the normal state is a
# different number, and the density matrix at that level held another
# charge (0.3604 instead of 0.35 per state with no pairing at nk=32). A
# mean field then corrects the level on the charge of the paired state
# (scftk.superscf.BdGFermiLevel). The band energy is the electronic one,
# (sum_{E<0} E_BdG + Tr h_e)/2, as in spectrum.total_energy.
# qtcitk/densitymatrix_qtci.py does the same on its reconstructed mesh.
import numpy as np
from numba import jit, prange

from .. import parallel


_ngl = 8 # Gauss-Legendre nodes per piece of the finite-T integral
_gx,_gw = np.polynomial.legendre.leggauss(_ngl)
_tcut = 40. # beyond |x| = 40 T the Fermi function is 1 or 0 to 4e-18
# the extra breakpoints of the finite-T integral, in units of T
_tbreaks = np.array([1.,2.,3.,4.,6.,8.,10.,12.,14.,18.,24.,32.,_tcut])


default_nk = 20 # mesh of get_dm when the caller gives none
bloechl_default = True # whether the mean field adds the Bloechl term,
                       # see point 4 of the module comment


def bloechl_coefficient(dim):
    """c_dim of the Bloechl correction, 1/(2 (dim+1) (dim+2))"""
    return 1./(2.*(dim+1)*(dim+2))


# ------------------------------------------------------------ T=0 kernel

@jit(nopython=True,cache=True)
def _sub(out,c,b0,b1,b2,b3,p):
    """add vol*mean(barycentric rows) of the sub-simplex with barycentric
    vertices b0..b3 (the first p used) to out, return vol"""
    if p==2:
        vol = abs(b0[0]*b1[1]-b0[1]*b1[0])
    elif p==3:
        vol = abs(b0[0]*(b1[1]*b2[2]-b1[2]*b2[1])
                 -b0[1]*(b1[0]*b2[2]-b1[2]*b2[0])
                 +b0[2]*(b1[0]*b2[1]-b1[1]*b2[0]))
    else:
        m = np.empty((4,4))
        for j in range(4):
            m[0,j] = b0[j]; m[1,j] = b1[j]; m[2,j] = b2[j]; m[3,j] = b3[j]
        vol = abs(np.linalg.det(m))
    for j in range(p):
        s = b0[j]+b1[j]
        if p>2: s += b2[j]
        if p>3: s += b3[j]
        out[j] += c*vol*s/p
    return vol


@jit(nopython=True,cache=True)
def _cut(i,j,ei,ej,mu,p):
    """barycentric coordinates of the point E_L = mu on the edge (i,j),
    with ei <= mu < ej"""
    b = np.zeros(p)
    a = (mu-ei)/(ej-ei)
    b[i] = 1.-a
    b[j] = a
    return b


@jit(nopython=True,cache=True)
def _vert(i,p):
    b = np.zeros(p)
    b[i] = 1.
    return b


@jit(nopython=True,cache=True)
def _step_weights(e,p,mu,out):
    """T=0 weights int_T theta(mu - E_L) lambda_j / vol(T) for sorted vertex
    energies e[0..p-1], written to out[0..p-1] (sorted order). Returns the
    occupied fraction of the simplex, sum(out)"""
    for j in range(p): out[j] = 0.
    if mu<=e[0]: return 0.
    if mu>=e[p-1]:
        for j in range(p): out[j] = 1./p
        return 1.
    z = np.zeros(p)
    if p==2: # e0 < mu < e1
        _sub(out,1.,_vert(0,p),_cut(0,1,e[0],e[1],mu,p),z,z,p)
    elif p==3:
        if mu<e[1]: # the corner of vertex 0
            _sub(out,1.,_vert(0,p),_cut(0,1,e[0],e[1],mu,p),
                    _cut(0,2,e[0],e[2],mu,p),z,p)
        else: # everything but the corner of vertex 2
            for j in range(p): out[j] = 1./p
            _sub(out,-1.,_vert(2,p),_cut(0,2,e[0],e[2],mu,p),
                    _cut(1,2,e[1],e[2],mu,p),z,p)
    else:
        if mu<e[1]: # the corner of vertex 0
            _sub(out,1.,_vert(0,p),_cut(0,1,e[0],e[1],mu,p),
                    _cut(0,2,e[0],e[2],mu,p),_cut(0,3,e[0],e[3],mu,p),p)
        elif mu<e[2]: # a prism (v0,p02,p03)-(v1,p12,p13), three tetrahedra
            v0 = _vert(0,p); v1 = _vert(1,p)
            p02 = _cut(0,2,e[0],e[2],mu,p); p03 = _cut(0,3,e[0],e[3],mu,p)
            p12 = _cut(1,2,e[1],e[2],mu,p); p13 = _cut(1,3,e[1],e[3],mu,p)
            _sub(out,1.,v0,p02,p03,v1,p)
            _sub(out,1.,p02,p03,v1,p12,p)
            _sub(out,1.,p03,v1,p12,p13,p)
        else: # everything but the corner of vertex 3
            for j in range(p): out[j] = 1./p
            _sub(out,-1.,_vert(3,p),_cut(0,3,e[0],e[3],mu,p),
                    _cut(1,3,e[1],e[3],mu,p),_cut(2,3,e[2],e[3],mu,p),p)
    s = 0.
    for j in range(p): s += out[j]
    return s


@jit(nopython=True,cache=True)
def _step_dos(e,p,mu):
    """T=0 density of states of the simplex at mu, d(occupied fraction)/dmu,
    for sorted vertex energies"""
    if mu<=e[0] or mu>=e[p-1]: return 0.
    if p==2: return 1./(e[1]-e[0])
    if p==3:
        if mu<e[1]: return 2.*(mu-e[0])/((e[1]-e[0])*(e[2]-e[0]))
        return 2.*(e[2]-mu)/((e[2]-e[0])*(e[2]-e[1]))
    if mu<e[1]:
        return 3.*(mu-e[0])**2/((e[1]-e[0])*(e[2]-e[0])*(e[3]-e[0]))
    if mu<e[2]:
        return 3./((e[2]-e[0])*(e[3]-e[0]))*((e[1]-e[0])+2.*(mu-e[1])
                -(e[2]+e[3]-e[0]-e[1])*(mu-e[1])**2/((e[2]-e[1])*(e[3]-e[1])))
    return 3.*(e[3]-mu)**2/((e[3]-e[0])*(e[3]-e[1])*(e[3]-e[2]))


# ------------------------------------------------------------ finite T

@jit(nopython=True,cache=True)
def _fermi(x,T):
    if x>=0.:
        ex = np.exp(-x/T)
        return ex/(1.+ex)
    return 1./(1.+np.exp(x/T))


@jit(nopython=True,cache=True)
def _piece(e,p,a,b,T,gx,gw,out,tmp,dos):
    """add int_a^b (-f'(x)) g(x) dx to out, with g the T=0 weights (or the
    T=0 DOS when dos, into out[0]) at mu = x, for a<b on one side of 0. The
    variable is phi = f(|x|), in which -f'(x) dx = |dphi|"""
    if b<=0.: # mirror: x = -u, u in [-b,-a]
        sgn = -1.
        u0 = -b; u1 = -a
    else:
        sgn = 1.
        u0 = a; u1 = b
    f0 = _fermi(u0,T); f1 = _fermi(u1,T) # f0 > f1
    h = (f0-f1)/2.; c = (f0+f1)/2.
    if h<=0.: return
    for q in range(gx.shape[0]):
        phi = c+h*gx[q]
        x = sgn*T*(np.log1p(-phi)-np.log(phi))
        if dos: out[0] += gw[q]*h*_step_dos(e,p,x)
        else:
            _step_weights(e,p,x,tmp)
            for j in range(p): out[j] += gw[q]*h*tmp[j]


@jit(nopython=True,cache=True)
def _knots(e,p,T,tbreaks,kn):
    """sorted breakpoints of the finite-T integral inside [e0, e_(p-1)]
    clipped to |x| <= tcut T: the vertex energies, 0 and +-tbreaks*T.
    Returns their number"""
    lo = max(e[0],-tbreaks[-1]*T); hi = min(e[p-1],tbreaks[-1]*T)
    m = 0
    kn[m] = lo; m += 1
    kn[m] = hi; m += 1
    for j in range(p):
        if e[j]>lo and e[j]<hi: kn[m] = e[j]; m += 1
    if 0.>lo and 0.<hi: kn[m] = 0.; m += 1
    for t in tbreaks:
        for s in (-t*T,t*T):
            if s>lo and s<hi: kn[m] = s; m += 1
    kn[:m].sort()
    return m


@jit(nopython=True,cache=True)
def _weights_T(e,p,T,gx,gw,tbreaks,out,tmp,kn):
    """weights at temperature T for sorted vertex energies e (measured from
    mu), into out; returns the occupied fraction"""
    for j in range(p): out[j] = 0.
    L = tbreaks[-1]*T
    if e[0]>=L: return 0.
    if e[p-1]<=-L:
        for j in range(p): out[j] = 1./p
        return 1.
    ftop = _fermi(e[p-1],T) # the constant piece above the top vertex
    for j in range(p): out[j] = ftop/p
    if e[p-1]>e[0]:
        m = _knots(e,p,T,tbreaks,kn)
        for i in range(m-1):
            if kn[i+1]>kn[i]:
                _piece(e,p,kn[i],kn[i+1],T,gx,gw,out,tmp,False)
    s = 0.
    for j in range(p): s += out[j]
    return s


@jit(nopython=True,cache=True)
def _dos_T(e,p,T,gx,gw,tbreaks,tmp,kn):
    """density of states of the simplex at temperature T"""
    L = tbreaks[-1]*T
    if e[0]>=L or e[p-1]<=-L or e[p-1]<=e[0]: return 0.
    acc = np.zeros(1)
    m = _knots(e,p,T,tbreaks,kn)
    for i in range(m-1):
        if kn[i+1]>kn[i]: _piece(e,p,kn[i],kn[i+1],T,gx,gw,acc,tmp,True)
    return acc[0]


@jit(nopython=True,cache=True)
def _sorted_vertices(es,sim,it,n,mu,e,perm,p):
    for v in range(p):
        e[v] = es[sim[it,v],n]-mu
        perm[v] = v
    for i in range(1,p): # insertion sort, carrying the permutation
        x = e[i]; ix = perm[i]
        j = i-1
        while j>=0 and e[j]>x:
            e[j+1] = e[j]; perm[j+1] = perm[j]
            j -= 1
        e[j+1] = x; perm[j+1] = ix


@jit(nopython=True,parallel=True,cache=True)
def _simplex_weights_jit(sim,es,mu,T,bcoef,gx,gw,tbreaks,out):
    """per simplex, band and vertex (in the vertex order of sim): the
    weight in units of the simplex volume, with the Bloechl term when
    bcoef != 0"""
    ns = sim.shape[0]; p = sim.shape[1]; nb = es.shape[1]
    for it in prange(ns):
        e = np.zeros(p); perm = np.zeros(p,dtype=np.int64)
        w = np.zeros(p); tmp = np.zeros(p)
        kn = np.zeros(p+3+2*tbreaks.shape[0])
        for n in range(nb):
            _sorted_vertices(es,sim,it,n,mu,e,perm,p)
            if T>0.: _weights_T(e,p,T,gx,gw,tbreaks,w,tmp,kn)
            else: _step_weights(e,p,0.,w)
            if bcoef!=0.:
                if T>0.: D = _dos_T(e,p,T,gx,gw,tbreaks,tmp,kn)
                else: D = _step_dos(e,p,0.)
                if D!=0.:
                    es_sum = 0.
                    for i in range(p): es_sum += e[i]
                    for j in range(p): w[j] += bcoef*D*(es_sum-p*e[j])
            for j in range(p): out[it,n,perm[j]] = w[j]


@jit(nopython=True,parallel=True,cache=True)
def _multiplet_average_real(es,w,tol):
    """replace w[k,n] by its mean over the degenerate multiplet of n"""
    for ik in prange(es.shape[0]):
        nb = es.shape[1]
        i = 0
        while i<nb:
            j = i+1
            while j<nb and es[ik,j]-es[ik,j-1]<tol: j += 1
            if j-i>1:
                acc = 0.
                for m in range(i,j): acc += w[ik,m]
                for m in range(i,j): w[ik,m] = acc/(j-i)
            i = j


# ------------------------------------------------------------ charge and DOS
#
# The Fermi search needs the charge N(mu) and its derivative, the density
# of states, at each trial level. The kernel below gives both in one pass
# over the simplices. The occupied fraction of a simplex is the closed form
# of sum_j w_j, so no weight array is formed, and at T>0 the fraction and
# the DOS are read at the same quadrature nodes and knots as _weights_T and
# _dos_T.

@jit(nopython=True,cache=True)
def _step_fraction(e,p,mu):
    """T=0 occupied fraction of the simplex, sum of _step_weights, in closed
    form for sorted vertex energies"""
    if mu<=e[0]: return 0.
    if mu>=e[p-1]: return 1.
    if p==2: return (mu-e[0])/(e[1]-e[0])
    if p==3:
        if mu<e[1]: return (mu-e[0])**2/((e[1]-e[0])*(e[2]-e[0]))
        return 1.-(e[2]-mu)**2/((e[2]-e[0])*(e[2]-e[1]))
    if mu<e[1]:
        return (mu-e[0])**3/((e[1]-e[0])*(e[2]-e[0])*(e[3]-e[0]))
    if mu<e[2]:
        x = mu-e[1]; e10 = e[1]-e[0]
        return (e10*e10+3.*e10*x+3.*x*x-(e[2]+e[3]-e[0]-e[1])*x**3
                /((e[2]-e[1])*(e[3]-e[1])))/((e[2]-e[0])*(e[3]-e[0]))
    return 1.-(e[3]-mu)**3/((e[3]-e[0])*(e[3]-e[1])*(e[3]-e[2]))


@jit(nopython=True,cache=True)
def _charge_dos_T(e,p,T,gx,gw,tbreaks,kn):
    """occupied fraction and DOS of the simplex at temperature T, for sorted
    vertex energies measured from mu, with the knots and nodes of _weights_T
    and _dos_T. A flat simplex is a step, f(e - mu), whose derivative
    f(1-f)/T is its DOS"""
    L = tbreaks[-1]*T
    if e[0]>=L: return 0.,0.
    if e[p-1]<=-L: return 1.,0.
    ftop = _fermi(e[p-1],T)
    if e[p-1]<=e[0]: return ftop,ftop*(1.-ftop)/T
    c = ftop; d = 0.
    m = _knots(e,p,T,tbreaks,kn)
    for i in range(m-1):
        a = kn[i]; b = kn[i+1]
        if b<=a: continue
        if b<=0.: # mirror, as in _piece
            sgn = -1.; u0 = -b; u1 = -a
        else:
            sgn = 1.; u0 = a; u1 = b
        f0 = _fermi(u0,T); f1 = _fermi(u1,T)
        h = (f0-f1)/2.; cc = (f0+f1)/2.
        if h<=0.: continue
        for q in range(gx.shape[0]):
            phi = cc+h*gx[q]
            x = sgn*T*(np.log1p(-phi)-np.log(phi))
            c += gw[q]*h*_step_fraction(e,p,x)
            d += gw[q]*h*_step_dos(e,p,x)
    return c,d


_block = 1024 # simplices per task of the charge kernel


@jit(nopython=True,parallel=True,cache=True)
def _charge_dos_jit(sim,es,mu,T,gx,gw,tbreaks):
    """sum over simplices and bands of the occupied fraction and of the
    DOS, in blocks so that the sum does not depend on the threads"""
    ns = sim.shape[0]; p = sim.shape[1]; nb = es.shape[1]
    L = tbreaks[-1]*T
    nblk = (ns+_block-1)//_block
    cs = np.zeros(nblk); ds = np.zeros(nblk)
    for ib in prange(nblk):
        e = np.zeros(p); perm = np.zeros(p,dtype=np.int64)
        kn = np.zeros(p+3+2*tbreaks.shape[0])
        c = 0.; d = 0.
        for it in range(ib*_block,min(ns,(ib+1)*_block)):
            for n in range(nb):
                lo = es[sim[it,0],n]; hi = lo
                for v in range(1,p):
                    x = es[sim[it,v],n]
                    if x<lo: lo = x
                    if x>hi: hi = x
                if lo-mu>=L: continue # empty, also at T=0
                if hi-mu<=-L: # full
                    c += 1.
                    continue
                _sorted_vertices(es,sim,it,n,mu,e,perm,p)
                if T>0.:
                    cc,dd = _charge_dos_T(e,p,T,gx,gw,tbreaks,kn)
                    c += cc; d += dd
                else:
                    c += _step_fraction(e,p,0.); d += _step_dos(e,p,0.)
        cs[ib] = c; ds[ib] = d
    return np.sum(cs),np.sum(ds)


# ------------------------------------------------------------ weights

_chunk = 2**18 # simplices per call of the weight kernel


def simplex_weights(dim,nk,es,mu=0.,T=0.,bloechl=False):
    """Occupation weights w[k,n] of every state of the uniform nk^dim mesh
    (row-major, as klist.kmesh), from the linear simplex integration of the
    Fermi function at chemical potential mu and temperature T. Summed over
    k and n they give the number of electrons per unit cell. A fully
    occupied state has w = 1/nk^dim, the weight of the uniform mesh."""
    from ..dostk.simplexdos import simplices
    sim,frac = simplices(dim,nk)
    es = np.ascontiguousarray(es,dtype=np.float64)
    nb = es.shape[1]
    T = 0. if T is None else float(abs(T))
    bcoef = bloechl_coefficient(dim) if bloechl else 0.
    parallel.set_num_threads()
    w = np.zeros(es.shape[0]*nb)
    for i0 in range(0,len(sim),_chunk):
        s = sim[i0:i0+_chunk]
        out = np.zeros((len(s),nb,s.shape[1]))
        _simplex_weights_jit(s,es,float(mu),T,bcoef,_gx,_gw,_tbreaks,out)
        idx = (s[:,None,:]*nb+np.arange(nb)[None,:,None]).ravel()
        w += np.bincount(idx,weights=out.ravel(),minlength=len(w))
    return w.reshape(es.shape)*frac


def simplex_charge(dim,nk,es,mu=0.,T=0.):
    """Electrons per unit cell at chemical potential mu and temperature T,
    the sum of simplex_weights without forming them"""
    return simplex_charge_dos(dim,nk,es,mu=mu,T=T)[0]


def simplex_charge_dos(dim,nk,es,mu=0.,T=0.):
    """Electrons per unit cell and density of states per unit cell, dN/dmu,
    at chemical potential mu and temperature T, in one pass"""
    from ..dostk.simplexdos import simplices
    sim,frac = simplices(dim,nk)
    es = np.ascontiguousarray(es,dtype=np.float64)
    T = 0. if T is None else float(abs(T))
    return _charge_and_dos(sim,frac,es,float(mu),T)


def _charge_and_dos(sim,frac,es,mu,T):
    """(N, dN/dmu) per unit cell on the simplices sim of volume frac"""
    parallel.set_num_threads()
    c,d = _charge_dos_jit(sim,es,mu,T,_gx,_gw,_tbreaks)
    return c*frac,d*frac


def _staircase_guess(es,target):
    """Fermi level of the uniform mesh, each state counted as a step that is
    half occupied at its energy, interpolated linearly between two states.
    target is the charge per unit cell"""
    x = es.ravel()
    if len(x)<2: return float(x[0])
    t = target*es.shape[0]-0.5 # fractional index into the sorted energies
    i = int(np.clip(np.floor(t),0,len(x)-2))
    s = np.partition(x,[i,i+1])
    a = min(max(t-i,0.),1.)
    return float(s[i]+a*(s[i+1]-s[i]))


def fermi_from_energies(dim,nk,es,filling,T=0.):
    """Chemical potential at which the simplex integration holds the
    fraction `filling` of the states. N(mu) is continuous, so the requested
    charge is reached exactly; when it is reached on a whole interval (a
    gap), the middle of the interval is returned.

    The root is found by a safeguarded Newton iteration, with the DOS of the
    same simplices as the derivative, inside a bracket that bisection takes
    over when a Newton step leaves it or does not halve. It starts from the
    level of the uniform mesh, so that a metal takes three or four charge
    evaluations. N is exactly flat on an interval where no simplex is
    within the cutoff of the Fermi function (tcut T, zero at T=0), and the
    DOS is exactly zero there. A level on such an interval that holds the
    charge returns its middle, which is the middle of the gap between the
    vertex energies on each side, at every T"""
    from ..filling import check_filling
    from ..dostk.simplexdos import simplices
    check_filling(filling)
    es = np.ascontiguousarray(es,dtype=np.float64)
    nb = es.shape[1]
    target = filling*nb
    emin,emax = float(np.min(es)),float(np.max(es))
    T = 0. if T is None else float(abs(T))
    pad = 1e-6+50.*T
    if target<=0.: return emin-pad
    if target>=nb: return emax+pad
    sim,frac = simplices(dim,nk)
    tol = 1e-12*nb
    L = _tbreaks[-1]*T # beyond L the Fermi function is a step
    lo,hi = emin-pad,emax+pad # N(lo) = 0 and N(hi) = nb, exactly
    xtol = 1e-14*max(1.,abs(emax)+abs(emin)) # bracket width that stops
    xstop = 1e-13*max(1.,abs(emax)+abs(emin)) # Newton step that stops
    mu = _staircase_guess(es,target)
    if not lo<mu<hi: mu = (lo+hi)/2.
    dxold = hi-lo
    for it in range(200): # bisection alone needs about 60
        c,d = _charge_and_dos(sim,frac,es,mu,T)
        r = c-target
        if r<0.: lo = mu
        elif r>0.: hi = mu
        if d==0.: # N is flat around mu, between two vertex energies
            below = es[es<=mu-L]; above = es[es>=mu+L]
            if abs(r)<=tol and len(below)>0 and len(above)>0:
                return (float(np.max(below))+float(np.min(above)))/2.
            # not at the charge: skip the flat interval
            if r<0. and len(above)>0: lo = max(lo,float(np.min(above))-L)
            if r>0. and len(below)>0: hi = min(hi,float(np.max(below))+L)
        elif r==0.: return mu
        step = r/d if d>0. else 0.
        new = mu-step
        if d>0. and lo<new<hi and abs(step)<=0.5*abs(dxold): # Newton
            dxold = step
            if abs(step)<=xstop: return new
        else: # bisection
            dxold = 0.5*(hi-lo)
            new = lo+dxold
            if hi-lo<=xtol: return new
        mu = new
    return mu


# ------------------------------------------------------------ the mesh

def check_hamiltonian(h,what):
    """Refuse what the simplex density matrix does not cover"""
    if h.dimensionality not in (1,2,3):
        raise ValueError(what+" needs a periodic Hamiltonian of "
                "dimensionality 1, 2 or 3, got "+str(h.dimensionality))
    if getattr(h,"non_hermitian",False):
        raise NotImplementedError(what+" needs a Hermitian Hamiltonian; a "
                "non-Hermitian one is solved with its biorthogonal density "
                "matrix, which only integration=\"ed\" builds")


max_memory_gb = 2. # eigenvectors of the whole mesh kept above this are
                   # recomputed instead, at the cost of a second diagonalization


class _Mesh():
    """Band energies (and eigenvectors) of h on the uniform mesh, computed
    in batches; ndiag counts the diagonalizations"""
    def __init__(self,h,nk,batch=2**14):
        h = h.get_dense()
        self.hkgen = h.get_hk_gen()
        self.dim = h.dimensionality
        self.nk = nk
        from ..dostk.simplexdos import uniform_mesh
        self.ks = uniform_mesh(self.dim,nk)
        self.n = h.intra.shape[0]
        self.batch = batch
        self.ndiag = 0
    def _hk(self,ks):
        from ..dostk.simplexdos import _hk_batch
        return _hk_batch(self.hkgen,ks)
    def energies(self):
        from ..htk.eigenvectors import peigvalsh
        parallel.set_num_threads()
        out = []
        for i0 in range(0,len(self.ks),self.batch):
            out.append(peigvalsh(self._hk(self.ks[i0:i0+self.batch])))
        self.ndiag += len(self.ks)
        return np.concatenate(out)
    def eigenpairs(self):
        """yields (i0, es, U) per batch"""
        from ..htk.eigenvectors import parallel_diagonalization
        parallel.set_num_threads()
        for i0 in range(0,len(self.ks),self.batch):
            es,U = parallel_diagonalization(self._hk(self.ks[i0:i0+self.batch]))
            self.ndiag += len(es)
            yield i0,es,U


def get_fermi4filling(h,filling,nk=20,T=0.):
    """Fermi energy that holds `filling` (a fraction of the states) under
    the simplex integration of the density matrix, on the uniform nk^dim
    mesh. See the module comment, point 3. For a Bogoliubov-de Gennes
    Hamiltonian the triangle density matrix is the step one of exact
    diagonalization on the same mesh (see the module comment), so the
    count is the one of exact diagonalization on the normal state,
    spectrum.get_fermi4filling; a mean field corrects it on the charge of
    the paired state (scftk.superscf.BdGFermiLevel)"""
    if h.has_eh:
        from ..spectrum import get_fermi4filling as get_fermi4filling_ed
        return get_fermi4filling_ed(h,filling,nk=nk,T=0.) # the step
    check_hamiltonian(h,"the simplex Fermi level")
    mesh = _Mesh(h,nk)
    return fermi_from_energies(mesh.dim,nk,mesh.energies(),filling,T=T)


def get_dm(h,ds,nk=20,T=0.,bloechl=None,counted=False):
    """Density matrix {d: (n,n) matrix} of h at zero chemical potential for
    the lattice vectors ds, from the simplex integration of the
    occupations on the uniform nk^dim mesh at temperature T, in the
    convention of densitymatrix.full_dm (dm[d][i,j] = <c^dag_i c_j> between
    the cell 0 and the cell d, up to that convention's transpose).
    bloechl=None takes bloechl_default. With counted=True also returns the
    number of diagonalizations"""
    check_hamiltonian(h,"the simplex density matrix")
    if bloechl is None: bloechl = bloechl_default
    mesh = _Mesh(h,nk)
    ds = [tuple(int(x) for x in d) for d in ds]
    n = mesh.n
    keep = len(mesh.ks)*n*n*16 <= max_memory_gb*1e9
    stored = []
    es_all = []
    for i0,es,U in mesh.eigenpairs():
        es_all.append(es)
        if keep: stored.append(U)
    es = np.concatenate(es_all)
    w = simplex_weights(mesh.dim,nk,es,mu=0.,T=T,bloechl=bloechl)
    from ..dostk.simplexdos import _degeneracy_tol
    _multiplet_average_real(es,w,_degeneracy_tol(es))
    dm = {d: np.zeros((n,n),dtype=np.complex128) for d in ds}
    dvecs = np.array(ds,dtype=np.float64).reshape(len(ds),3)
    def add(i0,U):
        wk = w[i0:i0+len(U)]
        ks = mesh.ks[i0:i0+len(U)]
        active = np.nonzero(np.any(wk!=0.,axis=0))[0] # bands with weight
        if len(active)==0: return
        Ua = U[:,:,active]
        wa = wk[:,active]
        ph = np.exp(2j*np.pi*(ks@dvecs.T)) # (k,d)
        for idd,d in enumerate(ds):
            A = Ua.conj()*(wa*ph[:,idd][:,None])[:,None,:]
            dm[d] += np.einsum("kin,kjn->ij",A,Ua,optimize=True)
    if keep:
        for i0,U in zip(range(0,len(mesh.ks),mesh.batch),stored): add(i0,U)
    else: # second pass
        for i0,es2,U in mesh.eigenpairs(): add(i0,U)
    if counted: return dm,mesh.ndiag
    return dm


def band_energy(h,nk=20,bloechl=None):
    """Sum of the band energies below zero per unit cell, int theta(-E) E,
    with the simplex weights at T=0 on the uniform nk^dim mesh: the exact
    band energy of the linearly interpolated bands, plus the Bloechl term
    when bloechl (None takes bloechl_default). For a Bogoliubov-de Gennes
    Hamiltonian it is the electronic energy,
    E = (sum_{E<0} E_BdG + Tr h_e)/2, with Tr h_e the onsite trace of the
    electron block, which is the mean of Tr h_e(k) over the mesh, as in
    spectrum.total_energy"""
    check_hamiltonian(h,"the simplex band energy")
    if bloechl is None: bloechl = bloechl_default
    mesh = _Mesh(h,nk)
    e = band_energy_from_energies(mesh.dim,nk,mesh.energies(),
            bloechl=bloechl)
    if h.has_eh:
        from .. import operators, algebra
        pe = np.array(algebra.todense(operators.get_electron(h)))
        intra = np.array(algebra.todense(h.intra))
        e = (e+np.sum(pe.diagonal().real*intra.diagonal().real))/2.
    return e


def band_energy_from_energies(dim,nk,es,bloechl=True):
    """band_energy from the band energies es[k,n] of the uniform nk^dim
    mesh (row-major, as klist.kmesh), measured from the Fermi level"""
    w = simplex_weights(dim,nk,es,mu=0.,T=0.,bloechl=bloechl)
    return float(np.sum(w*es))
