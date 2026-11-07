# Second-harmonic (second-order interband) optical conductivity of a
# periodic tight-binding Hamiltonian. The physics, the units and the sign
# conventions are documented in the public entry point, conductivity.py
# (shg_conductivity) -- read that first; this module only holds the
# machinery, plus the length-gauge second-order conductivity at two
# frequencies (sigma2_length), the independent check of the photocurrents.
import numpy as np
from numba import jit, prange

from .. import parallel
from ..htk.eigenvectors import hk_matrix_batch, parallel_diagonalization
from . import kubo


_kbatch = kubo._kbatch # k-points held in memory at once (see kubo.py)

_charge = -1. # charge q of the carriers, in units of e: electrons. The
              # second-order response is odd in q (it goes as q^3), so
              # unlike the linear conductivity its sign depends on this


def _derivative_tensors(h):
    """Everything needed to build every Cartesian k-derivative of the
    Peierls-coupled Bloch Hamiltonian on a batch of k-points, up to third
    order.

    In the Peierls (atomic-gauge) tight-binding model every hopping enters
    as t_ij(R) exp(i K.d_ij(R)), with the full bond vector d_ij(R) = R +
    r_j - r_i (see kubo._velocities), so a derivative with respect to the
    Cartesian K_a just multiplies the hopping by i d_a. Rotated back to
    pyqula's lattice gauge (the diagonal unitary exp(-i K.r_j) that relates
    the two Hamiltonians, under which the eigenbasis matrix elements are
    unchanged), this is

      d^m H/dK_a1...dK_am = sum_R (i d_a1)...(i d_am) o t(R) exp(2 pi i k.R)

    with o the elementwise product and k the reduced momentum. The first
    and second derivatives are therefore exactly kubo's velocity and
    inverse-mass operators (checked in tests/conductivity), and the third
    one, needed by the velocity-gauge second-order response, comes out of
    the same rule at no extra cost.

    Returns the stacked hoppings tms (nhop,n,n), including the intracell
    block as R = 0, their reduced lattice vectors dirs (nhop,3), the bond
    vectors d (nhop,3,n,n), the Hamiltonian generator and the cell volume
    and energy scale of kubo._setup."""
    hm,orders,hkgen,jac,dr,cellvol,scale = kubo._setup(h)
    tms,dirs = kubo._hopping_arrays(hm)
    n = hm.intra.shape[0]
    tms = np.concatenate([np.array([hm.intra],dtype=np.complex128),tms])
    dirs = np.concatenate([np.zeros((1,3)),dirs])
    avecs = 2.*np.pi*jac # the lattice vectors, one row per periodic direction
    dim = len(orders)
    R = dirs[:,:dim]@avecs # Cartesian lattice vectors, (nhop,3)
    d = R[:,:,None,None] + dr[None,:,:,:] # full bond vectors
    return tms,dirs,d,hkgen,cellvol,scale


def _cartesian_derivative(tms,d,ph,axes):
    """d^m H/dK_axes[0]...dK_axes[m-1] on a batch of k-points (the Bloch
    phases ph come from kubo._bloch_phases), as a (nk,n,n) array, see
    _derivative_tensors"""
    w = tms.copy()
    for a in axes: w = w*(1j*d[:,a]) # one factor i d_a per derivative
    return np.einsum("kh,hnm->knm",ph,w,optimize=True)


def _eigenbasis_derivatives(tms,dirs,d,hkgen,kb):
    """Band energies es[k,n] and the eigenbasis matrix elements of the
    first, second and third Cartesian derivatives of H on a batch of
    k-points: h1[k,a,n,m], h2[k,a,b,n,m] and h3[k,a,b,c,n,m]. The second
    and third are symmetric in their Cartesian indices, so only the sorted
    combinations are built and the rest are copies."""
    hks = hk_matrix_batch(hkgen,kb) # H(k) for the batch, densified
    es,wb = parallel_diagonalization(hks) # batched numba eigh
    wc = np.conjugate(wb).transpose(0,2,1) # the bras <n|
    ph = kubo._bloch_phases(dirs,kb)
    nk,n = es.shape
    h1 = np.zeros((nk,3,n,n),dtype=np.complex128)
    h2 = np.zeros((nk,3,3,n,n),dtype=np.complex128)
    h3 = np.zeros((nk,3,3,3,n,n),dtype=np.complex128)
    def rot(m): return wc@m@wb # into the eigenbasis
    for a in range(3):
        h1[:,a] = rot(_cartesian_derivative(tms,d,ph,[a]))
        for b in range(a,3):
            m2 = rot(_cartesian_derivative(tms,d,ph,[a,b]))
            h2[:,a,b] = m2 ; h2[:,b,a] = m2
            for c in range(b,3):
                m3 = rot(_cartesian_derivative(tms,d,ph,[a,b,c]))
                for (i,j,l) in set([(a,b,c),(a,c,b),(b,a,c),(b,c,a),
                                    (c,a,b),(c,b,a)]):
                    h3[:,i,j,l] = m3
    return es,h1,h2,h3


@jit(nopython=True,parallel=True,cache=True)
def _sigma2_velocity_jit(es,f,h1,h2,h3,omegas,delta,triplets,pairs,ipair):
    """Velocity-gauge second-order conductivity sigma_abc(w,w), summed (not
    normalized) over the k-points of a batch, for every frequency and every
    requested Cartesian triplet (a,b,c) (unsymmetrized in b,c; the caller
    symmetrizes). pairs lists the ordered (b,c) whose second-order density
    matrix is needed, and ipair[t] points triplet t to its pair.

    The recursion is Passos et al., PRB 97, 235446 (2018), Eqs. (rho) and
    (responsefunction), with hbar = 1 and every frequency carrying the
    same broadening, w -> w + i delta:

      rho1^c_nm    = h^c_nm (f_m - f_n)/(w - (E_n - E_m))
      rho2^bc_nm   = ([h^b,rho1^c] + (1/2)[h^bc,rho0])_nm/(2w - (E_n - E_m))
      sigma_abc    = (i^2 q^3/w^2) sum_nm [h^a_nm rho2^bc_mn
                       + h^ab_nm rho1^c_mn + (1/2) h^abc_nn f_n delta_nm]

    As in kubo._sigma_jit the parallel loop runs over frequencies, so each
    thread owns its output slice."""
    nk = es.shape[0]
    n = es.shape[1]
    nw = omegas.shape[0]
    nt = triplets.shape[0]
    npair = pairs.shape[0]
    out = np.zeros((nw,nt),dtype=np.complex128)
    for iw in prange(nw): # parallel loop over frequencies
        w = omegas[iw] + 1j*delta
        rho1 = np.zeros((3,n,n),dtype=np.complex128)
        rho2 = np.zeros((npair,n,n),dtype=np.complex128)
        for ik in range(nk): # loop over k-points
            e = es[ik]
            for c in range(3):
                for i in range(n):
                    for j in range(n):
                        rho1[c,i,j] = h1[ik,c,i,j]*(f[ik,j]-f[ik,i])/(
                                w - (e[i]-e[j]))
            for ip in range(npair):
                b = pairs[ip,0] ; c = pairs[ip,1]
                comm = h1[ik,b]@rho1[c] - rho1[c]@h1[ik,b]
                for i in range(n):
                    for j in range(n):
                        rho2[ip,i,j] = (comm[i,j] + 0.5*h2[ik,b,c,i,j]*(
                                f[ik,j]-f[ik,i]))/(2.*w - (e[i]-e[j]))
            for t in range(nt):
                a = triplets[t,0] ; b = triplets[t,1] ; c = triplets[t,2]
                acc = 0j
                for i in range(n):
                    acc += 0.5*h3[ik,a,b,c,i,i]*f[ik,i]
                    for j in range(n):
                        acc += h1[ik,a,i,j]*rho2[ipair[t],j,i]
                        acc += h2[ik,a,b,i,j]*rho1[c,j,i]
                out[iw,t] += acc
        out[iw] = out[iw]*(-_charge**3)/(w*w) # i^2 q^3/w^2
    return out


def _sigma2_length(es,f,h1,h2,omegas,delta,triplets,degeneracy_tol):
    """Length-gauge second-order conductivity sigma_abc(w,w) summed over a
    batch of k-points, for a cold insulator: the independent code path the
    tests check the velocity gauge against. Same output layout as
    _sigma2_velocity_jit. It is the two-frequency kernel _sigma2_length2
    at w1 = w2 = w, see there for the formula"""
    return _sigma2_length2(es,f,h1,h2,omegas,omegas,delta,triplets,
            degeneracy_tol)[0]


def _sigma2_length2(es,f,h1,h2,w1s,w2s,delta,triplets,degeneracy_tol):
    """Length-gauge second-order conductivity sigma_abc(w1,w2) summed over
    a batch of k-points, for a cold insulator, at the frequency pairs
    (w1s[i],w2s[i]) and for every ordered triplet (a,b,c), unsymmetrized:
    the field along b is at w1 and the one along c at w2, and the caller
    averages with the partner (a,c,b) at (w2,w1).

    Passos et al., PRB 97, 235446 (2018), Eq. (secondgamma), with the
    covariant derivatives written out in the eigenbasis (Sipe and
    Shkrebtii, PRB 61, 5337 (2000)), every frequency carrying the same
    broadening w -> w + i delta:

      [D^c,rho0]_nm  = i f_nm r^c_nm
      Y^c_nm         = i f_nm r^c_nm/(w2 - (E_n - E_m))
      (Y^c_nm);b     = i f_nm [r^c_nm;b/(w2 - (E_n - E_m))
                       + r^c_nm Delta^b_nm/(w2 - (E_n - E_m))^2]
      [D^b,Y^c]_mn   = (Y^c_mn);b - i sum_p (r^b_mp Y^c_pn - Y^c_mp r^b_pn)
      sigma_abc      = -q^3 sum_nm h^a_nm [D^b,Y^c]_mn/(w1 + w2 - (E_m - E_n))

    with Delta^b_nm = h^b_nn - h^b_mm, r^a_nm = h^a_nm/(i(E_n - E_m)) the
    interband Berry connection and its generalized derivative r^c_nm;b
    obtained from the exact identity [D^b,h^c] = h^bc of the tight-binding
    model (the sum rule of Ibanez-Azpiroz, Tsirkin and Souza, PRB 97,
    245143 (2018), Eq. (32), here including the second-derivative term).
    The terms with a k-derivative of the occupations are left out, which
    is exact only for an insulator at T=0, and every band pair must be
    nondegenerate, since r and r;b divide by the energy differences.

    On a mesh the two gauges agree to rounding only where a symmetry
    cancels the mesh sums of total k-derivatives (the aaa components, or
    C3 with a mirror); otherwise they agree once nk has converged. The
    double pole of the Delta term is the slow part of the length gauge: on
    a strained honeycomb lattice at delta = 0.02 and w = 1.5 it still
    differs from the velocity gauge by more than the value itself at
    nk = 96, while its static limit has converged at nk = 48.

    Returns the total and, separately, its n = m part
    sum_n h^a_nn [D^b,Y^c]_nn/(w1 + w2), which at (w,-w) is the pole of
    the injection current (see sigma2_length_diagonal), both of shape
    (len(w1s),len(triplets))."""
    nk,n = es.shape
    w1s = np.asarray(w1s,dtype=np.float64)
    w2s = np.asarray(w2s,dtype=np.float64)
    out = np.zeros((len(w1s),len(triplets)),dtype=np.complex128)
    outd = np.zeros((len(w1s),len(triplets)),dtype=np.complex128)
    dE = es[:,:,None] - es[:,None,:] # dE[k,n,m] = E_n - E_m
    dET = np.swapaxes(dE,1,2) # E_m - E_n at [k,n,m]
    off = ~np.eye(n,dtype=bool)
    if n>1 and np.min(np.abs(dE[:,off]))<degeneracy_tol: raise ValueError(
            "the length-gauge path needs a nondegenerate spectrum at every "
            "k-point")
    inv = np.where(off,1./np.where(off,dE,1.),0.)
    r = h1*inv[:,None]/1j # r^a_nm, (nk,3,n,n)
    vd = np.real(np.diagonal(h1,axis1=2,axis2=3)) # h^a_nn, (nk,3,n)
    Delta = vd[:,:,:,None] - vd[:,:,None,:] # Delta^a_nm = h^a_nn - h^a_mm
    rgds = {} # generalized derivatives r^c_nm;b, built once per (c,b)
    def rgd(c,b):
        if (c,b) not in rgds:
            gd = h2[:,b,c] + 1j*(r[:,b]@h1[:,c] - h1[:,c]@r[:,b])
            rgds[(c,b)] = (gd - 1j*Delta[:,b]*r[:,c])*inv/1j
        return rgds[(c,b)]
    fnm = f[:,:,None] - f[:,None,:] # f_nm = f_n - f_m
    for iw in range(len(w1s)):
        z1 = w1s[iw] + 1j*delta
        z2 = w2s[iw] + 1j*delta
        g1 = 1./(z2 - dE) # 1/(w2 - (E_n - E_m))
        g2 = 1./(z1 + z2 - dET) # 1/(w1 + w2 - (E_m - E_n)) at [n,m]
        for t,(a,b,c) in enumerate(triplets):
            Y = 1j*fnm*r[:,c]*g1
            Yb = 1j*fnm*(rgd(c,b)*g1 + r[:,c]*Delta[:,b]*g1**2)
            Z = Yb - 1j*(r[:,b]@Y - Y@r[:,b]) # [D^b,Y^c]
            out[iw,t] = -_charge**3*np.sum(h1[:,a]*np.swapaxes(Z,1,2)*g2)
            Zd = np.diagonal(Z,axis1=1,axis2=2) # [D^b,Y^c]_nn, (nk,n)
            outd[iw,t] = -_charge**3*np.sum(vd[:,a]*Zd)/(z1 + z2)
    return out,outd


def _triplets(components):
    """Cartesian triplets (a,b,c) from labels such as "xyy", together with
    the distinct ordered pairs (b,c) whose second-order density matrix
    they need, both orders (b,c) and (c,b), since the response is
    symmetrized in its last two indices"""
    trip = []
    for comp in components:
        if not isinstance(comp,str) or len(comp)!=3: raise ValueError(
            "a component is three Cartesian labels such as 'xyy', got "
            +repr(comp))
        trip.append(tuple(kubo._index(s) for s in comp))
    full = [] # every ordered triplet, including the b<->c partners
    for (a,b,c) in trip:
        for tt in [(a,b,c),(a,c,b)]:
            if tt not in full: full.append(tt)
    pairs = []
    for (a,b,c) in full:
        if (b,c) not in pairs: pairs.append((b,c))
    ipair = [pairs.index((b,c)) for (a,b,c) in full]
    return trip,full,np.array(full,dtype=np.int64),\
            np.array(pairs,dtype=np.int64),np.array(ipair,dtype=np.int64)


def _velocity(es,f,h1,h2,h3,energies,delta,triplets,pairs,ipair,full,tol):
    return _sigma2_velocity_jit(es,f,h1,h2,h3,energies,delta,triplets,
            pairs,ipair)


def _length(es,f,h1,h2,h3,energies,delta,triplets,pairs,ipair,full,tol):
    return _sigma2_length(es,f,h1,h2,energies,delta,full,tol)


_gauges = {"velocity":_velocity,"length":_length} # gauge= registry


_all_components = [x+y+z for x in "xyz" for y in "xyz" for z in "xyz"]


def _sigma2_grid(h,energies,nk,T,delta,trip,gauge,degeneracy_tol,
        tolerance=None):
    """Second-harmonic conductivity of the Cartesian triplets trip as a
    plain sum over the uniform mesh, symmetrized in the last two indices,
    as an array (len(energies),len(trip))"""
    trip,full,triplets,pairs,ipair = _triplets(
            ["xyz"[a]+"xyz"[b]+"xyz"[c] for (a,b,c) in trip])
    tms,dirs,d,hkgen,cellvol,scale = _derivative_tensors(h)
    ks = kubo._kmesh(h,nk)
    parallel.set_num_threads() # set the number of threads for numba
    acc = np.zeros((len(energies),len(full)),dtype=np.complex128)
    for i0 in range(0,len(ks),_kbatch): # loop over batches of k-points
        kb = ks[i0:i0+_kbatch]
        es,h1,h2,h3 = _eigenbasis_derivatives(tms,dirs,d,hkgen,kb)
        f = kubo._fermi(es,T)
        acc += _gauges[gauge](es,f,h1,h2,h3,energies,delta,triplets,
                pairs,ipair,full,degeneracy_tol*scale)
    acc = acc/(len(ks)*cellvol) # (1/V) sum_k, with V = N_k * V_cell
    return np.array([(acc[:,full.index((a,b,c))]+acc[:,full.index((a,c,b))])/2.
            for (a,b,c) in trip]).T # symmetrize in the last two indices


def _sigma2_triangle(h,energies,nk,T,delta,trip,gauge,degeneracy_tol,
        tolerance=None):
    if h.dimensionality==3: raise NotImplementedError("integration="
            "'triangle' of the second-harmonic conductivity is only "
            "implemented in 1D and 2D, not in 3D; use integration='grid'")
    from . import shgsimplex
    return shgsimplex.sigma_triangle(h,energies,nk,T,delta,trip,
            degeneracy_tol)


def _sigma2_qtci(h,energies,nk,T,delta,trip,gauge,degeneracy_tol,
        tolerance=1e-6):
    if h.dimensionality==3: raise NotImplementedError("integration="
            "'qtci' of the second-harmonic conductivity is only implemented "
            "in 2D, not in 3D; use integration='grid'")
    from . import shgsimplex
    return shgsimplex.sigma_qtci(h,energies,nk,T,delta,trip,degeneracy_tol,
            tolerance=tolerance)


_integrations = {
    "grid": _sigma2_grid, # plain sum over the mesh
    "triangle": _sigma2_triangle, # exact resonances over linear simplices
    "qtci": _sigma2_qtci, # the same on a quantics-interpolated fine mesh
    }


def get_integration_names():
    """Accepted values of integration= in shg_conductivity"""
    return list(_integrations)


def _check_hamiltonian(h,delta):
    """Guards shared by shg_conductivity and sigma2_length"""
    if h.dimensionality not in (1,2,3): raise NotImplementedError(
            "the second-order conductivity is only implemented for "
            "dimensionality 1, 2 or 3, got "+str(h.dimensionality))
    if h.has_eh: raise NotImplementedError("the second-harmonic "
            "conductivity is not implemented for Nambu (superconducting) "
            "Hamiltonians: the charge current operator there is not i[H,r]")
    if not delta>0.: raise ValueError("the broadening delta must be "
            "positive, the response has a 1/(w + i delta)^2 prefactor, got "
            "delta="+str(delta))


def shg_conductivity(h,energies=None,nk=20,T=None,delta=0.1,
        components=None,gauge="velocity",degeneracy_tol=1e-6,
        integration="grid",tolerance=1e-6):
    """Second-harmonic conductivity sigma_abc(2w;w,w), see
    conductivity.shg_conductivity for the documentation"""
    if energies is None: energies = np.linspace(0.,4.,100)
    energies = np.array(energies,dtype=np.float64)
    if T is None: T = delta # same default as kubo.optical_conductivity
    single = isinstance(components,str)
    if components is None: components = _all_components
    if single: components = [components]
    _check_hamiltonian(h,delta)
    if gauge not in _gauges: raise ValueError("unknown gauge "+repr(gauge)
            +", accepted values are "+", ".join(sorted(_gauges)))
    if gauge=="length" and T>0.: raise ValueError("the length-gauge path "
            "leaves out the Fermi-surface terms and is exact only for a cold "
            "insulator: it needs T=0, got T="+str(T))
    if integration not in _integrations: raise ValueError("unknown "
            "integration "+repr(integration)+" for the second-harmonic "
            "conductivity, accepted values are "
            +", ".join(get_integration_names()))
    if integration!="grid" and gauge!="velocity": raise NotImplementedError(
            "integration="+repr(integration)+" of the second-harmonic "
            "conductivity is only implemented in the velocity gauge, got "
            "gauge="+repr(gauge))
    trip = _triplets(components)[0]
    s = _integrations[integration](h,energies,nk,T,delta,trip,gauge,
            degeneracy_tol,tolerance=tolerance)
    out = {comp:s[:,i] for (i,comp) in enumerate(components)}
    if single: return energies,out[components[0]]
    return energies,out


def _sigma2_length_grid(h,w1s,w2s,nk,delta,components,degeneracy_tol):
    """Total and diagonal part of the length-gauge sigma_abc(w1,w2) on the
    uniform mesh, averaged with the (b,w1) <-> (c,w2) partner, as two
    arrays (len(w1s),len(components)), see sigma2_length"""
    w1s = np.atleast_1d(np.array(w1s,dtype=np.float64))
    w2s = np.atleast_1d(np.array(w2s,dtype=np.float64))
    if w1s.shape!=w2s.shape: raise ValueError("w1s and w2s are the two "
            "frequencies of each pair and need the same length, got "
            +str(len(w1s))+" and "+str(len(w2s)))
    _check_hamiltonian(h,delta)
    trip,full,triplets,pairs,ipair = _triplets(components)
    tms,dirs,d,hkgen,cellvol,scale = _derivative_tensors(h)
    ks = kubo._kmesh(h,nk)
    parallel.set_num_threads() # set the number of threads for numba
    acc = np.zeros((2,2,len(w1s),len(full)),dtype=np.complex128)
    for i0 in range(0,len(ks),_kbatch): # loop over batches of k-points
        kb = ks[i0:i0+_kbatch]
        es,h1,h2,h3 = _eigenbasis_derivatives(tms,dirs,d,hkgen,kb)
        f = kubo._fermi(es,0.)
        tol = degeneracy_tol*scale
        acc[0] += np.array(_sigma2_length2(es,f,h1,h2,w1s,w2s,delta,full,
                tol)) # (b,w1),(c,w2)
        acc[1] += np.array(_sigma2_length2(es,f,h1,h2,w2s,w1s,delta,full,
                tol)) # (b,w2),(c,w1), the partners
    acc = acc/(len(ks)*cellvol) # (1/V) sum_k, with V = N_k * V_cell
    def sym(p): # average (a,b,c) at (w1,w2) with (a,c,b) at (w2,w1)
        return np.array([(acc[0,p][:,full.index((a,b,c))]
                + acc[1,p][:,full.index((a,c,b))])/2.
                for (a,b,c) in trip]).T
    return sym(0),sym(1)


def _components_out(s,components):
    """dict of arrays, or one array for a single component string"""
    if isinstance(components,str): return s[:,0]
    return {comp:s[:,i] for (i,comp) in enumerate(components)}


def sigma2_length(h,w1s,w2s,nk=20,delta=0.1,components=None,
        degeneracy_tol=1e-6):
    """Second-order conductivity sigma_abc(w1+w2;w1,w2) of a cold
    insulator in the length gauge, at the frequency pairs (w1s[i],w2s[i]),
    J_a(w1+w2) = sigma_abc(w1,w2) E_b(w1) E_c(w2), see _sigma2_length2 for
    the formula. Every frequency carries the broadening w -> w + i delta,
    so the pole at w1 + w2 carries 2 i delta; the occupations are those at
    T=0 and the spectrum must be nondegenerate.

    sigma is averaged over the two orderings of the fields, (b,w1) with
    (c,w2), so sigma_abc(w1,w2) = sigma_acb(w2,w1), and it is not symmetric
    in b and c at w1 != w2. At w1 = w2 it is the length-gauge
    shg_conductivity. At (w,-w) it is the rectified current of the real
    field E(t) = Re E0 exp(-iwt), J_a = (1/2) sigma_abc(w,-w) E0_b E0_c^*,
    whose diagonal part (sigma2_length_diagonal) is eta_abc/delta, the
    injection rate over the broadening, and whose remainder is the shift
    current.

    Units and conventions are those of shg_conductivity. Returns a dict
    {component: array of len(w1s)}, or one array if components is a single
    string."""
    if components is None: components = _all_components
    comps = [components] if isinstance(components,str) else components
    s = _sigma2_length_grid(h,w1s,w2s,nk,delta,comps,degeneracy_tol)[0]
    return _components_out(s,components)


def sigma2_length_diagonal(h,w1s,w2s,nk=20,delta=0.1,components=None,
        degeneracy_tol=1e-6):
    """The n = m part of sigma2_length,

      -q^3 (1/(N_k V_cell)) sum_k sum_n h^a_nn [D^b,Y^c]_nn/(w1 + w2 + 2 i delta)

    averaged with the same partner. Since Y^c_nn = 0 it involves only the
    interband pairs, and at (w,-w) delta times it is the injection rate

      eta_abc = (pi q^3/2) (1/(N_k V_cell)) sum_k sum_{n!=m} f_nm
                (h^a_mm - h^a_nn) r^b_mn r^c_nm L_delta(w - (E_m - E_n))

    with L_delta(x) = (delta/pi)/(x^2 + delta^2), exactly at any delta and
    nk: the principal parts of the two orderings cancel and only the
    Lorentzian survives. eta_acb = eta_abc^*, so the bc-symmetric part is
    the linear injection, odd under time reversal, and the bc-antisymmetric
    part the circular one. Same arguments and return as sigma2_length."""
    if components is None: components = _all_components
    comps = [components] if isinstance(components,str) else components
    s = _sigma2_length_grid(h,w1s,w2s,nk,delta,comps,degeneracy_tol)[1]
    return _components_out(s,components)


_e_over_eps0 = 1.602176634e-19/8.8541878128e-12 # e/epsilon_0, in V m


def shg_susceptibility(h,length_unit=1.0,energy_unit=1.0,**kwargs):
    """Second-harmonic susceptibility chi_abc(2w;w,w) in SI units, see
    conductivity.shg_susceptibility for the documentation.

    With J = dP/dt and the same broadening w -> w + i delta that every
    frequency carries in shg_conductivity, chi = sigma/(-i 2 (w + i delta)
    epsilon_0), finite at w = 0 for an insulator. In the units of
    shg_conductivity (e = hbar = 1, energies in units of t and lengths in
    units of a) this is

      chi[SI] = i sigma/(2 (w + i delta)) * e a^(3-d)/(epsilon_0 t^2)

    with t in eV and a in m, returned here in nm^(4-d)/V (nm^2/V for the
    sheet susceptibility of a 2D system)."""
    delta = kwargs.get("delta",0.1)
    energies,sigma = shg_conductivity(h,**kwargs)
    return to_susceptibility(h,energies,sigma,delta,length_unit,energy_unit)


def to_susceptibility(h,energies,sigma,delta,length_unit,energy_unit):
    """Convert a second-harmonic conductivity (an array, or a dict of
    arrays) to the SI susceptibility, see shg_susceptibility"""
    d = h.dimensionality
    a = length_unit*1e-10 # in m
    fac = _e_over_eps0*a**(3-d)/energy_unit**2/(1e-9)**(4-d)
    def conv(s): return 1j*s/(2.*(energies + 1j*delta))*fac
    if isinstance(sigma,dict): return energies,{c:conv(s) for (c,s) in sigma.items()}
    return energies,conv(sigma)
