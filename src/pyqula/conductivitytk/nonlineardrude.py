"""l-th order nonlinear Drude (spin and charge) conductivity.

See conductivity.nonlinear_drude_conductivity for the user-facing
documentation; this module holds the implementation.

Formula
-------
Ezawa, "Third-order and fifth-order nonlinear spin-current generation in
g-wave and i-wave altermagnets, and perfectly nonreciprocal spin-current in
f-wave magnets", arXiv:2411.16036 (Phys. Rev. B 111, 125420 (2025)), Eq.
(Drude). For a Hamiltonian in which spin is a good quantum number, so that
each spin channel s = +-1 is an independent single-band problem, solving the
semiclassical Boltzmann equation recursively in the electric field gives,
for the current expanded as j_b = sum sigma^{x^l1 y^l2 ; b} E_x^l1 E_y^l2,

  sigma_s^{x^l1 y^l2 ; b} =  (-e/hbar)^(l1+l2+1) / (i w + 1/tau)^(l1+l2)
                             * Int d^D k f_s^(0) d^(l1+l2+1) eps_s
                                         / (dk_x^l1 dk_y^l2 dk_b)

and the spin and charge responses are

  sigma_spin   = (sigma_up - sigma_dn)/2,
  sigma_charge =  sigma_up + sigma_dn.

The whole content of the X-wave selection rule sits in the derivative in
that integrand. An X-wave form factor is a harmonic of order l+1 in k, so
its (l+1)-th derivative is the first one that is a nonzero constant and
every lower one integrates to zero over the zone: d-wave responds at first
order, f-wave at second, g-wave at third and i-wave at fifth. Reading off
the lowest nonvanishing order therefore measures the wave index.

Units and normalization
-----------------------
e = hbar = 1, as elsewhere in pyqula. The Brillouin-zone integral is taken
in the density normalization Int d^D k/(2 pi)^D = (1/(N_k V_cell)) sum_k,
the same convention as conductivitytk/kubo.py, so these numbers are Ezawa's
divided by (2 pi)^D. Ratios against fermi_volume() below, which uses the
same normalization, are free of that factor -- which is how the
absolute-scale tests are written.

Gauge
-----
Nothing here needs the atomic gauge that kubo.py's velocity operators do.
Band energies are gauge invariant (the lattice-gauge and atomic-gauge Bloch
Hamiltonians differ by a k-dependent diagonal unitary), so eps_n(k) and all
of its k-derivatives are identical in both and no intracell-bond correction
arises.

Cartesian derivatives of a Bloch series
---------------------------------------
H(k) = sum_R m_R exp(i k.R) over real-space lattice vectors R (pyqula stores
the integer direction and multiplies the reduced momentum by 2 pi, see
geometrytk/bloch.py). Differentiating with respect to the *Cartesian*
momentum K_a is then just multiplication of each term by i R_a,

  d^n H/dK_{a1}...dK_{an} = sum_R m_R prod_j (i R_{a_j}) exp(i k.R),

exact at any order, with no reduced-to-Cartesian chain rule to get wrong and
no finite differences. Since the phases exp(i k.R) do not depend on which
derivative is taken, they are built once per k-mesh and reused for every
tensor component -- which is what makes a full selection-rule sweep over
orders 0..6 cheap. tests/nonlinearconductivity/ pins this against
current.hk_derivative, the repo's shared (reduced-coordinate) k-derivative,
so that the two routes are known to agree.
"""
import math
import warnings

# memory budget per k-block of the multiband Taylor recursion (see _chunks)
_CHUNK_BYTES = 1e9

import numpy as np

from .. import algebra
from .. import klist
from .kubo import _fermi
from ..check import require_spin


def _axis(c):
    """Cartesian index of a component label"""
    d = {"x":0,"y":1,"z":2}
    if c in d: return d[c]
    if c in (0,1,2): return int(c)
    raise ValueError("Unknown Cartesian component "+str(c))


def _check_spin_conserving(h,tol=1e-8):
    """Check that every hopping matrix is block diagonal in spin.

    Without this, remove_spin() would silently discard the spin off-diagonal
    blocks. The single-band Boltzmann derivation implemented here would not
    apply to such a Hamiltonian anyway: once spin-orbit coupling mixes the
    channels, the nonlinear response picks up quantum-metric and
    Berry-curvature-dipole contributions as well (Ezawa, arXiv:2409.09241)."""
    hm = h.get_multicell()
    ms = [np.asarray(algebra.todense(hm.intra))]
    ms += [np.asarray(algebra.todense(t.m)) for t in hm.hopping]
    for m in ms:
        off = np.abs(m[0::2,1::2]).max()+np.abs(m[1::2,0::2]).max()
        if off>tol: raise ValueError(
            "spin is not a good quantum number in this Hamiltonian (spin "
            "off-diagonal terms of size "+str(off)+"); the nonlinear Drude "
            "formula of arXiv:2411.16036 assumes a spin-diagonal Hamiltonian")


def _spin_channels(h):
    """Return [(s, hs)] with s = +-1 and hs the spinless Hamiltonian of that
    spin channel"""
    require_spin(h,"the nonlinear Drude conductivity, whose two spin "
        "channels have to be resolved,")
    if h.has_eh: raise NotImplementedError(
        "not implemented for Nambu (superconducting) Hamiltonians")
    _check_spin_conserving(h)
    out = []
    for (s,channel) in [(1,"up"),(-1,"dn")]:
        hs = h.copy()
        hs.remove_spin(channel=channel)
        out.append((s,hs.get_multicell().copy()))
    return out


def _bloch_series(hm):
    """Decompose a multicell Hamiltonian into (integer directions, Cartesian
    lattice vectors, matrices): the terms of H(k) = sum_R m_R e^{i k.R}"""
    g = hm.geometry
    avecs = np.array([g.a1,g.a2,np.zeros(3)])
    dirs = [(0,0,0)]
    ms = [np.asarray(algebra.todense(hm.intra))]
    for t in hm.hopping:
        m = np.asarray(algebra.todense(t.m))
        if np.max(np.abs(m))<1e-12: continue
        dirs.append(tuple(np.array(t.dir)[0:3]))
        ms.append(m)
    dirs = np.array(dirs,dtype=np.float64)
    return dirs,dirs@avecs,np.array(ms)


class _Channel():
    """One spin channel, with its Bloch series and the Bloch phases on a
    fixed k-mesh precomputed. Band energies come for free, and Cartesian
    derivatives of them are then obtained exactly -- see derivative()."""

    def __init__(self,hm,ks,degeneracy_tol=1e-8):
        self.hm = hm
        self.ks = np.array(ks,dtype=np.float64)
        self.dirs,self.rcart,self.ms = _bloch_series(hm)
        self.nbands = self.ms.shape[1]
        self.degeneracy_tol = degeneracy_tol
        self.scale = max(float(np.max(np.abs(self.ms))),1e-12)
        self.ph = np.exp(2.j*np.pi*(self.ks[:,0:3]@self.dirs.T)) # (nk,nhop)
        # cached eigenvalue Taylor coefficients. An expansion to order N
        # contains every lower order, so one expansion to the highest order
        # asked for serves every call below it.
        self._taylor = None
        self._taylor_order = -1
        if self.nbands==1: # eps(k) is H(k) itself
            self.coeffs = self.ms[:,0,0]
            self.es = np.real(self.ph@self.coeffs)[:,None]
        else: # multiorbital: diagonalize on the mesh, in chunks
            self.es = np.zeros((len(self.ks),self.nbands))
            for (lo,hi) in self._chunks():
                self.es[lo:hi] = np.linalg.eigvalsh(self._hk(lo,hi))

    def _chunks(self):
        """Split the k-mesh into blocks small enough that the Taylor
        coefficient matrices of one block fit comfortably in memory.

        The perturbation theory below holds ~(N+1)(N+2)/2 Hamiltonian
        Taylor matrices and as many eigenvector ones, each (nk, nb, nb)
        complex -- ~56 arrays at N = 6, so ~900 bytes per k-point per nb^2.
        At the ~100 orbitals of a real superlattice cell that is gigabytes
        for a full mesh, so k is processed in blocks and only the (much
        smaller) eigenvalue coefficients are kept."""
        nb = self.nbands
        per = max(1,int(_CHUNK_BYTES/max(1.,1000.*nb*nb)))
        return [(lo,min(lo+per,len(self.ks)))
                for lo in range(0,len(self.ks),per)]

    def _hk(self,lo,hi):
        """Bloch Hamiltonian on a block of the mesh, explicitly Hermitized
        (the Bloch sum is Hermitian only up to rounding)"""
        hk = np.einsum("kh,hij->kij",self.ph[lo:hi],self.ms)
        return (hk+np.conjugate(np.swapaxes(hk,1,2)))/2.

    def prepare(self,N):
        """Precompute the band-energy expansion to total order N, so that a
        sweep over several orders pays for it once"""
        if self.nbands>1: self._eigenvalue_taylor(N)

    def derivative(self,axes):
        """d^n eps_n/dK_{a1}...dK_{an} on the mesh, shape (nk, nbands).

        One orbital per cell is the easy case: eps(k) = H(k), and since
        H(k) = sum_R m_R e^{i k.R} the Cartesian derivative is just a
        multiplication by i R_a per order. For a multiorbital block eps_n(k)
        is an eigenvalue, and _eigenvalue_taylor propagates its Taylor
        coefficients instead. Both are exact to machine precision at any
        order -- which matters, because the X-wave selection rule is a
        statement that certain derivatives vanish *identically*, and any
        finite-difference scheme manufactures a spurious nonzero value there
        and so destroys the very thing being measured."""
        counts = [list(axes).count(a) for a in range(3)]
        if counts[2]!=0: return np.zeros(self.es.shape) # 2D: no k_z
        if self.nbands==1:
            w = np.ones(len(self.rcart),dtype=np.complex128)
            for a in axes: w = w*(1.j*self.rcart[:,a])
            return np.real(self.ph@(self.coeffs*w))[:,None]
        p,q = counts[0],counts[1]
        c = self._eigenvalue_taylor(p+q)[(p,q)]
        # the Taylor coefficient is the derivative divided by p! q!
        return c*float(math.factorial(p)*math.factorial(q))

    def _hamiltonian_taylor(self,N,lo,hi):
        """Hc[(p,q)] = (1/(p! q!)) d^(p+q)H/dKx^p dKy^q on the mesh.

        Straight from the Bloch series: each term m_R e^{i k.R} contributes
        m_R (i R_x)^p (i R_y)^q/(p! q!) e^{i k.R}. Exact, and vectorized over
        the whole k-mesh in one matrix product per (p,q)."""
        out = dict()
        for n in range(N+1):
            for p in range(n+1):
                q = n-p
                w = ((1.j*self.rcart[:,0])**p)*((1.j*self.rcart[:,1])**q)
                w = w/float(math.factorial(p)*math.factorial(q))
                out[(p,q)] = np.einsum("kh,hij->kij",self.ph[lo:hi]*w,self.ms)
        return out

    def _eigenvalue_taylor(self,N):
        """Taylor coefficients c[(p,q)][k,n] of the band energies,

          eps_n(k+d) = sum_{p,q} c[(p,q)] dx^p dy^q,

        by Rayleigh-Schroedinger perturbation theory carried out in the ring
        of truncated two-variable Taylor series. Writing
        H(k+d) = sum_a Hc[a] d^a and expanding the eigenpair as
        eps = sum_a e[a] d^a, v = sum_a v[a] d^a with the intermediate
        normalization u^dag v[a] = 0 for a != 0, the eigenvalue equation
        gives the standard pair of recursions

          e[a] = sum_{a' != 0} u^dag Hc[a'] v[a-a'],
          v[a] = S ( sum_{a' != 0} e[a'] v[a-a'] - sum_{a' != 0} Hc[a'] v[a-a'] )

        with S = sum_{m != n} |m><m|/(eps_m - eps_n) the reduced resolvent.
        Every mixed partial up to total order N comes out of a single pass,
        which is what the selection-rule sweep wants, and everything is
        vectorized over the k-mesh.

        This needs the band being expanded to be non-degenerate, since the
        reduced resolvent divides by eps_m - eps_n. A Dirac point sitting
        exactly on the k-mesh therefore raises rather than returning
        nonsense -- see the error message for the way out."""
        if N<=self._taylor_order: return self._taylor
        nk,nb = self.es.shape
        alphas = [(p,n-p) for n in range(N+1) for p in range(n+1)]
        e = {a:np.zeros((nk,nb)) for a in alphas}
        e[(0,0)] = self.es.copy()
        for (lo,hi) in self._chunks():
            for (a,c) in self._taylor_chunk(N,lo,hi,alphas).items():
                e[a][lo:hi] = c
        self._taylor = e
        self._taylor_order = N
        return e

    def _taylor_chunk(self,N,lo,hi,alphas):
        """The recursion above, on one block of the k-mesh, carried out in
        the *block* (Kato) form so that degenerate bands are handled.

        Individual band energies are not analytic where two bands touch, and
        their reduced resolvent diverges. Degeneracies are not exotic here:
        any cell with a C3 or C6 axis -- which is to say any of the
        superlattices these responses are interesting for -- has
        two-dimensional irreducible representations, and hence exactly
        degenerate bands, at its high-symmetry points.

        What is always analytic is the *sum* of the eigenvalues of a
        degenerate multiplet, since that is the trace of the effective
        Hamiltonian of a subspace separated from the rest of the spectrum
        (Kato). And the sum is all this module needs: the Fermi factor
        f(eps_n) is common to a degenerate multiplet, so

          sum_{n in block} f(eps_n) D^a eps_n = f(eps) D^a Tr H_eff.

        So bands closer than the degeneracy tolerance are grouped, the
        reduced resolvent excludes each group (S has no component inside
        it), and the eigenvalue coefficient matrix E keeps its off-diagonal
        elements within a group. Both are expressed as masks on (nb, nb)
        arrays, which keeps the whole thing vectorized over the k-mesh
        rather than needing a per-k-point list of blocks. Where no bands are
        degenerate the mask is the identity and this reduces exactly to
        ordinary non-degenerate Rayleigh-Schroedinger."""
        hc = self._hamiltonian_taylor(N,lo,hi)
        es,ws = np.linalg.eigh(self._hk(lo,hi))
        nb = self.nbands
        de = es[:,:,None]-es[:,None,:] # de[k,m,n] = eps_m - eps_n
        near = np.abs(de)<self.degeneracy_tol*self.scale # same block
        inv = np.where(near,0.,1./np.where(near,1.,de)) # reduced resolvent
        out = {a:np.zeros((hi-lo,nb)) for a in alphas}
        out[(0,0)] = es
        # V[a][k,i,n]: i the orbital index, n the band being expanded.
        # E[a][k,m,n]: the effective-Hamiltonian coefficients, nonzero only
        # within a degenerate group (the identity for a simple spectrum).
        v = {a:np.zeros((hi-lo,nb,nb),dtype=np.complex128) for a in alphas}
        E = {a:np.zeros((hi-lo,nb,nb),dtype=np.complex128) for a in alphas}
        v[(0,0)] = ws
        wsc = np.conjugate(ws)
        for n in range(1,N+1):
            for p in range(n+1):
                a = (p,n-p)
                hv = np.zeros((hi-lo,nb,nb),dtype=np.complex128)
                ve = np.zeros((hi-lo,nb,nb),dtype=np.complex128)
                for (p2,q2) in alphas:
                    if (p2,q2)==(0,0): continue
                    b = (a[0]-p2,a[1]-q2)
                    if b[0]<0 or b[1]<0: continue
                    hv = hv+np.einsum("kij,kjn->kin",hc[(p2,q2)],v[b])
                    ve = ve+np.einsum("kim,kmn->kin",v[b],E[(p2,q2)])
                Ea = np.einsum("kim,kin->kmn",wsc,hv) # <m|.|n>
                E[a] = np.where(near,Ea,0.) # keep only within a group
                out[a] = np.real(np.einsum("kmm->km",Ea))
                r = ve-hv # no component inside the group, by construction
                proj = np.einsum("kim,kin->kmn",wsc,r)
                v[a] = np.einsum("kim,kmn->kin",ws,proj*inv)
        return out


def _kmesh(h,nk):
    """Uniform Gamma-centered k-mesh, nk points per periodic direction"""
    return klist.kmesh(h.dimensionality,nk=nk)


def _cell_volume(h):
    """Unit cell volume (area in 2D, length in 1D)"""
    g = h.geometry
    if h.dimensionality==1: return np.sqrt(g.a1.dot(g.a1))
    return np.linalg.norm(np.cross(g.a1,g.a2))


def _channels(h,nk,degeneracy_tol=1e-8):
    """The two spin channels, set up on a shared k-mesh, plus the 1/(N_k
    V_cell) normalization of the Brillouin-zone integral"""
    if h.dimensionality not in (1,2): raise NotImplementedError(
        "the nonlinear Drude conductivity is implemented for dimensionality "
        "1 and 2 only")
    ks = _kmesh(h,nk)
    chans = [(s,_Channel(hm,ks,degeneracy_tol=degeneracy_tol))
             for (s,hm) in _spin_channels(h)]
    return chans,1./(len(ks)*_cell_volume(h))


def _spin_splitting_guard(chans):
    """Warn if the two spin channels are the same state to within rounding.

    A spin response only means something if the two channels actually have
    different bands. When they do not -- a compensated Neel state on a
    bipartite lattice is PT symmetric and exactly spin degenerate, an
    antiferromagnet rather than an altermagnet -- sigma_spin is built out of
    rounding, and it is *not* caught by looking at the answer: the two
    channels are diagonalized independently, so near a degeneracy their
    eigenvector gauges differ and the high-order derivatives drift apart far
    more than the band energies do. Measured on a 44-site honeycomb antidot
    Neel state: bands agreeing to 1.2e-14 gave sigma_up and sigma_dn
    differing by a factor of 8, and a fifth-order "spin response" of 1.3e-3
    that looks exactly like a signal.

    So the diagnostic is the band splitting itself, which is unambiguous and
    already computed. A genuine altermagnet is nowhere near this floor."""
    es = [c.es for (s,c) in chans]
    if es[0].shape!=es[1].shape: return
    split = float(np.max(np.abs(es[0]-es[1])))
    scale = float(np.max(es[0])-np.min(es[0]))
    if scale>0. and split<_SPIN_SPLITTING_TOL*scale:
        warnings.warn(
            "the two spin channels are degenerate to within rounding "
            "(max spin splitting %.2e against a bandwidth of %.2e), so this "
            "state carries no spin response and the number returned is "
            "rounding amplified by the k-derivatives, not a signal. Either "
            "the system is not magnetic at all, or its order is spin "
            "degenerate -- a compensated Neel state on a bipartite lattice "
            "is PT symmetric and is an antiferromagnet, not an altermagnet."
            %(split,scale),RuntimeWarning,stacklevel=3)


# relative spin splitting below which the two channels are numerically the
# same state and no spin response is meaningful (see _spin_splitting_guard)
_SPIN_SPLITTING_TOL = 1e-10


def _combine(out,channel):
    """Spin/charge combination of the two spin-resolved responses"""
    if channel=="spin": return (out[1]-out[-1])/2.
    elif channel=="charge": return out[1]+out[-1]
    elif channel=="up": return out[1]
    elif channel=="dn": return out[-1]
    else: raise ValueError("Unknown channel "+str(channel))


def _sigma(chans,norm,axes,b,channel,T,mu,tau,omega):
    """One tensor component, from already-built channels"""
    l = len(axes)
    pref = (-1.)**(l+1)/(1j*omega+1./tau)**l # e = hbar = 1
    out = dict()
    for (s,c) in chans:
        f = _fermi(c.es-mu,T)
        out[s] = pref*np.sum(f*c.derivative(list(axes)+[b]))*norm
    return _combine(out,channel)


def fermi_volume(h,nk=100,T=0.01,mu=0.,channel="charge"):
    """Fermi volume V^F = Int d^Dk/(2 pi)^D f^(0), in the same normalization
    as the conductivities. channel="charge" sums the two spin channels,
    "spin" takes half their difference, "up"/"dn" pick one."""
    chans,norm = _channels(h,nk)
    out = {s:np.sum(_fermi(c.es-mu,T))*norm for (s,c) in chans}
    return _combine(out,channel)


def nonlinear_drude_conductivity(h,field="x",current="x",channel="spin",
        nk=100,T=0.01,mu=0.,tau=1.,omega=0.,degeneracy_tol=1e-8):
    """l-th order nonlinear Drude conductivity, see
    conductivity.nonlinear_drude_conductivity for the documentation."""
    chans,norm = _channels(h,nk,degeneracy_tol=degeneracy_tol)
    if channel=="spin": _spin_splitting_guard(chans)
    axes = [_axis(c) for c in field] # one entry per power of the field
    return _sigma(chans,norm,axes,_axis(current),channel,T,mu,tau,omega)


def nonlinear_drude_components(h,l,channel="spin",nk=100,T=0.01,mu=0.,
        tau=1.,omega=0.,degeneracy_tol=1e-8):
    """Every component of the l-th order conductivity, as
    {"x^l1 y^l2;b": value}: the l field powers split between x and y, times
    the two current directions. They share one k-mesh, one set of Bloch
    phases and one perturbation expansion, so this is much cheaper than the
    equivalent individual calls."""
    chans,norm = _channels(h,nk,degeneracy_tol=degeneracy_tol)
    if channel=="spin": _spin_splitting_guard(chans)
    return _components(chans,norm,l,channel,T,mu,tau,omega)


def _components(chans,norm,l,channel,T,mu,tau,omega):
    """Components of one order, from already-built channels"""
    out = dict()
    for l1 in range(l+1):
        field = "x"*l1+"y"*(l-l1)
        axes = [_axis(c) for c in field]
        for b in ["x","y"]:
            out[field+";"+b] = _sigma(chans,norm,axes,_axis(b),channel,T,mu,
                    tau,omega)
    return out


def nonlinear_drude_orders(h,lmax=6,channel="spin",nk=100,T=0.01,mu=0.,
        tau=1.,omega=0.,degeneracy_tol=1e-8):
    """Selection-rule sweep: for each order l = 0..lmax, the largest
    |sigma^{x^l1 y^l2 ; b}| over all of that order's components.

    The lowest l at which this is nonzero is the order of the magnet, and so
    a direct readout of the X-wave index -- d gives 1, f gives 2, g gives 3
    and i gives 5 (Ezawa, arXiv:2411.16036). Taking the maximum over
    components matters: a d-wave altermagnet responds in sigma^{x;y} but not
    in sigma^{x;x}, so a sweep of one fixed component would miss it.

    Every order is evaluated from a single set of channels. That is not just
    tidiness: on a multiorbital cell the cost is dominated by the
    perturbation expansion of the band energies, which is computed once to
    the highest order needed and then cached, so the whole sweep costs about
    what its top order alone would."""
    chans,norm = _channels(h,nk,degeneracy_tol=degeneracy_tol)
    if channel=="spin": _spin_splitting_guard(chans)
    for (s,c) in chans: c.prepare(lmax+1) # one expansion, to the top order
    return [max(abs(v) for v in _components(chans,norm,l,channel,T,mu,tau,
        omega).values()) for l in range(lmax+1)]
