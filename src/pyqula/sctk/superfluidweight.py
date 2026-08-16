# Superfluid weight (superfluid stiffness) of a Bogoliubov-de Gennes
# mean-field Hamiltonian, with its conventional/quantum-geometric
# decomposition and a Berezinskii-Kosterlitz-Thouless temperature estimate.
#
# ---------------------------------------------------------------------------
# DEFINITION
# ---------------------------------------------------------------------------
# The superfluid weight is the rigidity of the grand potential against a
# phase twist of the superconducting order parameter,
#
#     D_s^{ab} = (1/V) d^2 Omega / dQ_a dQ_b   |_{Q=0} ,
#
# taken with the pairing amplitude |Delta_ij| *frozen* at its
# self-consistent value while its phase is wound as
# Delta_ij -> |Delta_ij| exp(i Q.(r_i+r_j)) (Cooper pairs of momentum 2Q).
# V is the unit-cell volume (area in 2d, length in 1d) and Q is a Cartesian
# wavevector, so in 2d D_s has units of energy and the Nelson-Kosterlitz
# criterion reads T_BKT = (pi/8) D_s.  This is the thermodynamic
# definition; it is equivalent to the Q->0, omega=0 limit of the
# current-current (Kubo) response -- see Liang et al. below.
#
# ---------------------------------------------------------------------------
# THE TWIST, AND WHICH GAUGE IT LIVES IN
# ---------------------------------------------------------------------------
# Gauging the wound phase into the fermions, c_i -> exp(i Q.r_i) c_i, leaves
# |Delta_ij| untouched and puts a Peierls phase on every hopping,
#
#     t_ij -> t_ij exp(i Q.d_ij) ,     d_ij = R + r_j - r_i ,
#
# with d_ij the *full bond vector*: the lattice vector R connecting the two
# unit cells plus the difference of the two orbital positions inside them.
# In the Nambu Hamiltonian that is a tau_z-graded phase on the stored
# hopping matrices,
#
#     electron-electron entry (i,j) of T_R  ->  x exp(+i Q.d_ij)
#     hole-hole entry (i,j) of T_R          ->  x exp(-i Q.d_ij)
#     anomalous (electron-hole) entries     ->  unchanged
#
# and, on a Bloch matrix, the electron block becomes h(k+Q) and the hole
# block -Theta h(-k+Q) Theta^-1 (in the atomic/periodic gauge).  Leaving
# the anomalous *hoppings* alone as well makes the twist equally correct
# for non-local pairing (extended s-wave, p-wave, d-wave): what is held
# fixed is the whole real-space matrix |Delta_ij|, only the centre-of-mass
# phase of the pairs is wound.
#
# THE INTRACELL-BOND TRAP.  pyqula builds Bloch matrices in the *lattice*
# (cell) gauge, H(k) = sum_R T_R exp(2 pi i k.R), whose phase carries only
# R and not the orbital positions -- so current.hk_derivative, the shared
# dH/dk wrapper, silently drops every intracell bond.  Twisting with
# exp(2 pi i q.R) instead of exp(i Q.d_ij) is therefore *not* the physical
# Peierls substitution whenever a unit cell holds more than one orbital.
# It is wrong in a way that is easy to see: on the honeycomb lattice, where
# one of the three nearest-neighbour bonds is intracell, it produces a D_s
# with D_xx != D_yy, which C3 symmetry forbids (0.3169 vs 0.2392 for the
# model in tests/superfluid/test_gauge_and_bond_vectors.py), and it makes
# D_s change when the same crystal is described with a supercell.  The
# operators built here therefore use the full bond vector, which is also
# why they are assembled explicitly below instead of going through
# current.hk_derivative -- that function cannot express a per-matrix-element
# bond vector.  (In the lattice gauge the two agree, and the test module
# above checks the explicit construction against current.hk_derivative
# there, so the shared operator still pins the normalisation.)
#
# Because the twist is written directly in terms of Cartesian bond vectors,
# Q is Cartesian throughout and no reduced->Cartesian conversion of the
# result is needed.  The twist directions are the Cartesian axes for a 2d
# or 3d lattice, and the direction of a1 for a chain.
#
# GAUGE OPTION.  gauge="atomic" (the default) is the physical prescription
# above.  gauge="lattice" reproduces the convention of Peotta & Toermae and
# Liang et al., who write H(k) without the orbital positions and twist it
# as k -> k+q; it is the convention in which pyqula's own quantum metric
# (topologytk/qgt.py) is computed, and is kept so that the flat-band
# identity below can be checked against it in a single consistent gauge.
# The two differ only for multi-orbital cells, and the difference is
# exactly the orbital-embedding dependence of the fixed-|Delta| superfluid
# weight analysed by Huhtinen, Herzog-Arbeitman, Chew, Bernevig & Toermae,
# "Revisiting flat band superconductivity: dependence on minimal quantum
# metric and band touchings", PRB 106, 014518 (2022).  Use the default
# unless you are deliberately reproducing lattice-gauge literature numbers.
#
# ---------------------------------------------------------------------------
# THE GRAND POTENTIAL, AND WHY THE FACTOR 1/2
# ---------------------------------------------------------------------------
# pyqula's "spinful_nambu" mode is the *doubled* Nambu basis (4 components
# per site: c_up, c_dn, c^dag_dn, -c^dag_up), so each physical quasiparticle
# appears twice in the 4N-dimensional spectrum {E_i(k,Q)}.  The grand
# potential is therefore
#
#     Omega(Q) = -(1/(2 beta)) sum_{k,i} ln(1 + exp(-beta E_i(k,Q))) + const
#
# where const collects (1/2) sum_k Tr h(k) and |Delta|^2/U, both independent
# of Q when the sum runs over a full uniform BZ mesh.  At T=0 this reduces
# to (1/2) sum_{k,i} E_i theta(-E_i), i.e. the standard BCS sum_k (xi_k-E_k)
# for one orbital.  This normalisation (no extra spin factor) is the one for
# which the T=0 weak-coupling limit gives the full Drude weight n/m and the
# single-band result reduces to Liang et al. Eq. (21); both are pinned by
# tests.  A "spinless_nambu" Hamiltonian describes a single spin species
# and its weight comes out exactly half of the spinful one.
#
# ---------------------------------------------------------------------------
# THE ANALYTIC (KUBO) FORMULA -- the primary result
# ---------------------------------------------------------------------------
# Writing A_a = dH/dQ_a and B_ab = d^2 H/dQ_a dQ_b (both from the twisted
# hoppings above), second-order perturbation theory in Q gives the exact
# derivative of Omega,
#
#   D_s^{ab} = (1/(2 V N_k)) sum_k { sum_{ij} W_ij <i|A_a|j><j|A_b|i>
#                                  + sum_i n_F(E_i) <i|B_ab|i> }
#
#   W_ij = [n_F(E_i) - n_F(E_j)] / (E_i - E_j)   ( -> n_F'(E_i) if E_i=E_j )
#
# with |i> the BdG eigenvectors at k.  W is a divided difference, so exact
# and near degeneracies (e.g. the ever-present spin degeneracy) need no
# special treatment beyond the numerical switch to n_F'.  This is
# equivalent to Eq. (14) of Liang et al.; there the diamagnetic term is
# folded into the paramagnetic one using the fact that a *uniform* shift
# H(k) -> H(k+delta) leaves the BZ sum invariant.
#
# ---------------------------------------------------------------------------
# CONVENTIONAL / GEOMETRIC DECOMPOSITION -- and its validity limits
# ---------------------------------------------------------------------------
# Under the assumptions
#   (i)  uniform on-site pairing, anomalous block = Delta * identity,
#   (ii) time-reversal symmetry, Theta h(-k) Theta^-1 = h(k),
# the BdG matrix block-diagonalises in the normal-state band basis {|m(k)>}
# into 2x2 blocks [[eps_m, Delta],[Delta*, -eps_m]] with E_m = sqrt(eps_m^2
# + |Delta|^2), and A_a = 1 (x) v_a, B_ab = tau_z (x) w_ab, where v_a is the
# (bond-vector) velocity operator of the normal state and w_ab its second
# derivative.  The current matrix J_a = W^dag v_a W then splits into its
# band-diagonal part [J_a]_mm = d eps_m/d K_a -- gauge independent, this is
# the band velocity -- which is the conventional contribution, and its
# interband part, which is the geometric one, exactly as in Liang et al.
# Eq. (17).  Splitting the Kubo formula accordingly gives
# D_s = D_conv + D_geom *identically on any k-mesh*.  The closed form of the
# conventional part,
#
#   D_conv^{ab} = (1/(2 V N_k)) sum_{k,m} (|Delta|^2/E_m^2)
#                 [ tanh(beta E_m/2)/E_m - (beta/2) sech^2(beta E_m/2) ]
#                 d_a eps_m d_b eps_m                                (*)
#
# is Liang et al. Eq. (21) and follows from the implemented form by a BZ
# integration by parts; (*) is what superfluid_weight_conventional_closed
# returns, and the tests check it against the implemented split.
#
# For an isolated flat band at the Fermi level the conventional part
# vanishes and
#
#   D_geom^{ab} = (2 |Delta|^2/(V N_k)) sum_k [tanh(beta E/2)/E] g_{ab}(k)
#
# with E = sqrt(xi^2+|Delta|^2) of the flat band and g_{ab} its quantum
# metric *in the same gauge as the twist*, in pyqula's normalisation --
# i.e. with gauge="lattice", exactly what h.get_quantum_metric(k=k,
# occ_idxs=[...the flat band...]) returns, traced over the subspace so that
# it already contains both spin copies.  This is Liang et al. Eq. (23);
# their Eq. (24) normalises the metric as ds^2 = (1/2) g dk dk whereas
# pyqula/topologytk/qgt.py (and the usual Provost-Vallee convention) uses
# ds^2 = g dk dk, so their g is twice the one used here -- and their band
# index is single-spin, so the two factors of two cancel and the prefactor
# above is literally theirs.
#
# VALIDITY LIMITS, enforced by the code:
#   * assumptions (i) and (ii) are *checked numerically* at every k-point;
#     superfluid_weight(...,decompose=True) raises if either fails, rather
#     than reporting a meaningless split.  The general Kubo result needs
#     neither and is always available.
#   * the split needs a resolvable normal-state band gap, because the
#     interband pieces carry 1/(eps_m - eps_n).  Exactly degenerate bands
#     with a vanishing interband current (the generic spin degeneracy) are
#     handled by dropping those terms; a degeneracy with a *finite*
#     interband current (a Kramers pair mixed by spin-orbit coupling, or a
#     band touching sitting on the mesh) raises.
#   * even the total D_s at fixed |Delta| is not a property of the band
#     structure alone: it depends on where the orbitals sit inside the cell
#     (that is what the gauge option above is about) and, for non-uniform
#     pairing, on which orbitals the interaction acts on.  See Huhtinen et
#     al., PRB 106, 014518 (2022).
#
# ---------------------------------------------------------------------------
# REFERENCES
# ---------------------------------------------------------------------------
#   Peotta & Toermae, "Superfluidity in topologically nontrivial flat
#     bands", Nat. Commun. 6, 8944 (2015), arXiv:1506.02815
#   Liang, Vanhala, Peotta, Siro, Harju & Toermae, "Band geometry, Berry
#     curvature and superfluid weight", PRB 95, 024515 (2017),
#     arXiv:1610.01803  -- Eqs. (14), (17), (21), (23) above
#   Toermae, Peotta & Bernevig, "Quantum geometry in superfluidity and
#     superconductivity", arXiv:2308.08248 (review)
#   Huhtinen, Herzog-Arbeitman, Chew, Bernevig & Toermae, PRB 106, 014518
#     (2022) -- orbital-embedding dependence, minimal quantum metric
#   Nelson & Kosterlitz, PRL 39, 1201 (1977) -- T_BKT = (pi/8) D_s
#
# WHAT THE TESTS PIN DOWN (tests/superfluid/):
#   * the tau_z mask and the bond vectors, block by block, against an
#     independently built h(k+Q) / -Theta h(-k+Q) Theta^-1;
#   * C3 isotropy of D_s on the honeycomb lattice and invariance under
#     redescribing the crystal with a supercell -- both fail in the lattice
#     gauge and hold in the atomic one;
#   * the analytic formula against a brute-force central finite difference
#     of Omega(Q) on the same mesh (agreement is mesh-independent, so this
#     is sharp already at small nk);
#   * the absolute normalisation against the single-band closed form
#     D_s^{ab} = (1/(V N_k)) sum_k |Delta|^2 d_a eps d_b eps / E^3, and its
#     large-Delta limit 2 t^2/|Delta| for the square lattice;
#   * D_s = 0 at Delta=0 and finite T, D_s > 0 at finite Delta;
#   * a flat-band model where D_conv = 0 and D_geom matches the quantum
#     metric of topologytk/qgt.py;
#   * symmetry and positive semi-definiteness of the tensor;
#   * the BKT self-consistency T = (pi/8) D_s(T).
import numpy as np

from .. import algebra
from ..kpointstk.kmesh import kmesh


# ---------------------------------------------------------------------------
# Nambu bookkeeping
# ---------------------------------------------------------------------------

def electron_hole_signs(h):
    """Return a vector s with s[i]=+1 for electron-like components of the
    Nambu basis and s[i]=-1 for hole-like ones.

    pyqula's spinful_nambu spinor is site-interleaved with four components
    per site, (c_up, c_dn, c^dag_dn, -c^dag_up), so the electrons are the
    indices with i%4 < 2 (see superconductivity.py and sctk/reorder.py);
    the spinless_nambu spinor is (c, c^dag) per site."""
    n = h.intra.shape[0]
    if h.check_mode("spinful_nambu"): return np.where(np.arange(n)%4<2,1.,-1.)
    elif h.check_mode("spinless_nambu"):
        return np.where(np.arange(n)%2<1,1.,-1.)
    else: raise ValueError("the superfluid weight requires a Nambu (BdG) "
            "Hamiltonian; call h.turn_nambu() or h.add_swave(...) first")


def twist_masks(h):
    """Return the two matrix masks that grade the twist in Nambu space.

    tau  = +1 on the electron-electron block, -1 on the hole-hole block,
           0 on the anomalous blocks -- the sign of the twist phase,
    diag = +1 on both diagonal blocks, 0 on the anomalous blocks -- tau^2,
           i.e. the mask that selects the entries the twist touches."""
    s = electron_hole_signs(h)
    same = np.equal.outer(s,s) # both electron or both hole
    tau = np.where(same,s[:,None],0.) # +1 ee, -1 hh, 0 anomalous
    diag = np.where(same,1.,0.) # +1 ee, +1 hh, 0 anomalous
    return tau,diag


def component_positions(h):
    """Cartesian position of every component of the Nambu basis, i.e. the
    position of the site it belongs to (4 components per site in
    spinful_nambu, 2 in spinless_nambu)."""
    n = h.intra.shape[0]
    r = np.array(h.geometry.r)
    if n%len(r)!=0: raise ValueError("cannot map the Hamiltonian basis onto "
            "the geometry: %d components for %d sites"%(n,len(r)))
    return np.repeat(r,n//len(r),axis=0)


# ---------------------------------------------------------------------------
# twist directions, cell volume
# ---------------------------------------------------------------------------

def twist_directions(g):
    """Cartesian unit vectors along which the twist tensor is reported: the
    direction of a1 for a chain, the x/y axes for a 2d lattice (which
    pyqula puts in the xy plane) and x/y/z for a 3d one."""
    dim = g.dimensionality
    if dim==1: return [g.a1/np.sqrt(g.a1.dot(g.a1))]
    elif dim==2: return [np.array([1.,0.,0.]),np.array([0.,1.,0.])]
    elif dim==3: return list(np.identity(3))
    else: raise NotImplementedError("the superfluid weight needs a periodic "
            "system (dimensionality 1, 2 or 3)")


def _cell_volume(g,dim=None):
    """Unit-cell volume: length in 1d, area in 2d, volume in 3d"""
    if dim is None: dim = g.dimensionality
    if dim==1: return np.sqrt(g.a1.dot(g.a1))
    elif dim==2: return np.abs(np.cross(g.a1,g.a2)[2])
    elif dim==3: return np.abs(np.dot(g.a1,np.cross(g.a2,g.a3)))
    else: raise NotImplementedError


# ---------------------------------------------------------------------------
# the twisted Hamiltonian and its Q-derivatives
# ---------------------------------------------------------------------------

class TwistOperators():
    """Everything the twist needs, precomputed once for a Hamiltonian:
    the stored hoppings, their Cartesian bond-vector matrices d_ij, and the
    matrices entering H(k,Q), A_a = dH/dQ_a and B_ab = d^2H/dQ_a dQ_b.

    Each stored hopping matrix T_R contributes, at Bloch phase
    exp(2 pi i k.R):
        H   :  T_R * exp(i tau_ij Q.d_ij)
        A_a :  T_R * (i tau_ij d_ij.e_a)
        B_ab:  T_R * (-diag_ij (d_ij.e_a)(d_ij.e_b))
    with tau/diag the Nambu masks and e_a the twist directions."""

    def __init__(self,h,gauge="atomic"):
        if gauge not in ("atomic","lattice"): raise ValueError(
                "unknown gauge "+str(gauge)+" (use 'atomic' or 'lattice')")
        hm = h.get_multicell().copy() # own copy: get_multicell may alias h
        hm.intra = np.asarray(hm.intra)
        for t in hm.hopping: t.m = np.asarray(t.m)
        self.h = hm
        self.geometry = hm.geometry
        self.dim = hm.dimensionality
        self.gauge = gauge
        self.tau,self.diag = twist_masks(hm)
        self.dirs = twist_directions(hm.geometry)
        self.volume = _cell_volume(hm.geometry,self.dim)
        avecs = np.array([hm.geometry.a1,hm.geometry.a2,hm.geometry.a3])
        rs = component_positions(hm)
        n = hm.intra.shape[0]
        # intra plus every stored hopping, with its Bloch cell index
        self.ms = [hm.intra] + [t.m for t in hm.hopping]
        self.ds = [np.zeros(3)] + [np.array(t.dir,dtype=np.float64)
                                    for t in hm.hopping]
        # Cartesian bond vector of every matrix element of every hopping
        self.bonds = []
        for d in self.ds:
            rv = d@avecs # lattice vector joining the two cells
            if gauge=="atomic": # full bond vector R + r_j - r_i
                self.bonds.append(rv[None,None,:]+rs[None,:,:]-rs[:,None,:])
            else: # lattice gauge: only the cell vector
                self.bonds.append(np.broadcast_to(rv,(n,n,3)).copy())
        # precomputed first and second derivative matrices
        nd = len(self.dirs)
        self.mA = [[1j*self.tau*(b@e)*m for (b,m) in zip(self.bonds,self.ms)]
                    for e in self.dirs]
        self.mB = dict()
        for a in range(nd):
            for b in range(a,nd):
                mm = [-self.diag*(bo@self.dirs[a])*(bo@self.dirs[b])*m
                        for (bo,m) in zip(self.bonds,self.ms)]
                self.mB[(a,b)] = mm ; self.mB[(b,a)] = mm

    def _bloch(self,ms,k):
        """sum_R ms[R] exp(2 pi i k.R) -- pyqula's Bloch convention"""
        k = _pad3(k)
        out = np.zeros(ms[0].shape,dtype=np.complex128)
        for (m,d) in zip(ms,self.ds):
            out = out + m*np.exp(1j*2.*np.pi*k.dot(d))
        return out

    def hk(self,k,Q=None):
        """Twisted Bloch BdG matrix H(k,Q); Q Cartesian, k reduced"""
        if Q is None: return self._bloch(self.ms,k)
        Q = _pad3(Q)
        ms = [m*np.where(self.diag!=0.,np.exp(1j*self.tau*(b@Q)),1.)
                for (b,m) in zip(self.bonds,self.ms)]
        return self._bloch(ms,k)

    def A(self,k):
        """[dH/dQ_a](k) for every twist direction a"""
        return [self._bloch(ms,k) for ms in self.mA]

    def B(self,k):
        """{(a,b): [d^2H/dQ_a dQ_b](k)}"""
        return {key: self._bloch(self.mB[key],k) for key in self.mB}


def _pad3(v):
    """Pad a k-point (or twist) to three components"""
    v = np.array(v,dtype=np.float64)
    return np.concatenate([v,np.zeros(3-len(v))]) if len(v)<3 else v[0:3]


def get_twisted_hk_gen(h,gauge="atomic"):
    """Return f(k,Q) giving the twisted Bloch BdG matrix, with k reduced and
    Q a Cartesian twist wavevector.  f(k,0) reproduces h.get_hk_gen()(k)."""
    ops = TwistOperators(h,gauge=gauge)
    return lambda k,Q: ops.hk(k,Q)


# ---------------------------------------------------------------------------
# thermal factors
# ---------------------------------------------------------------------------

def _fermi(e,T):
    """Fermi function, overflow safe.  T=0 gives the step function."""
    e = np.asarray(e,dtype=np.float64)
    if T<=0.: return np.where(e<0.,1.,np.where(e>0.,0.,0.5))
    x = e/T
    xp = np.clip(x,0.,None) ; xm = np.clip(x,None,0.)
    return np.where(x>0.,np.exp(-xp)/(1.+np.exp(-xp)),1./(1.+np.exp(xm)))


def _dfermi(e,T):
    """Derivative of the Fermi function (negative), overflow safe"""
    e = np.asarray(e,dtype=np.float64)
    if T<=0.: return np.zeros(e.shape)
    x = np.clip(np.abs(e)/(2.*T),None,300.)
    return -1./(4.*T*np.cosh(x)**2)


def _divided_difference(ei,ej,T,tol=1e-9):
    """W_ij = [n_F(E_i)-n_F(E_j)]/(E_i-E_j), continued to n_F'(E_i) when the
    two energies coincide.  Degeneracies (the spin degeneracy is always
    there) are thus handled exactly, with no tolerance on the physics --
    tol only decides which branch is numerically better conditioned."""
    fi = _fermi(ei,T) ; fj = _fermi(ej,T)
    de = ei[:,None] - ej[None,:]
    close = np.abs(de)<tol
    safe = np.where(close,1.,de)
    dfi = _dfermi(ei,T)[:,None]*np.ones(len(ej))[None,:]
    return np.where(close,dfi,(fi[:,None]-fj[None,:])/safe)


# ---------------------------------------------------------------------------
# the grand potential and the finite-difference route
# ---------------------------------------------------------------------------

def grand_potential(h,Q=None,nk=20,T=0.0,ks=None,gauge="atomic",ops=None):
    """Twist-dependent part of the grand potential per unit cell,

        omega(Q) = -(1/(2 beta N_k)) sum_{k,i} ln(1+exp(-beta E_i(k,Q)))

    (at T=0, (1/(2 N_k)) sum_{k,i} E_i theta(-E_i)).  The Q-independent
    constant of the BdG grand potential is dropped, so only differences in
    Q are meaningful -- which is all the superfluid weight needs.  Q is a
    Cartesian twist wavevector."""
    if ops is None: ops = TwistOperators(h,gauge=gauge)
    if ks is None: ks = kmesh(ops.dim,nk=nk)
    tot = 0.
    for k in ks:
        es = algebra.eigvalsh(ops.hk(k,Q))
        if T<=0.: tot += 0.5*np.sum(es[es<0.])
        else: tot += -0.5*T*np.sum(np.logaddexp(0.,-es/T))
    return tot/len(ks)


def superfluid_weight_finite_difference(h,nk=20,T=0.0,dQ=1e-3,ks=None,
        gauge="atomic"):
    """Superfluid weight from a brute-force central finite difference of the
    grand potential with respect to the twist.  Slow but conceptually
    unimpeachable: it is the definition, with no perturbation theory in
    between.  Used as the correctness oracle for superfluid_weight."""
    ops = TwistOperators(h,gauge=gauge)
    if ks is None: ks = kmesh(ops.dim,nk=nk) # one fixed mesh for every Q
    def om(Q): return grand_potential(h,Q=Q,T=T,ks=ks,ops=ops)
    nd = len(ops.dirs)
    e = np.array(ops.dirs) # Cartesian twist directions
    D = np.zeros((nd,nd))
    o0 = om(np.zeros(3))
    for a in range(nd): # diagonal, three-point stencil
        D[a,a] = (om(dQ*e[a]) - 2.*o0 + om(-dQ*e[a]))/dQ**2
    for a in range(nd): # off diagonal, four-point stencil
        for b in range(a+1,nd):
            v = (om(dQ*(e[a]+e[b])) - om(dQ*(e[a]-e[b]))
                 - om(dQ*(e[b]-e[a])) + om(-dQ*(e[a]+e[b])))/(4.*dQ**2)
            D[a,b] = v ; D[b,a] = v
    return D/ops.volume


# ---------------------------------------------------------------------------
# the analytic (Kubo) route -- primary result
# ---------------------------------------------------------------------------

def _superfluid_weight_at(es,ws,A,B,T,nd):
    """Per-k-point Kubo contribution, given the BdG spectrum/eigenvectors at
    that k and the twist operators A_a, B_ab.  Returns the paramagnetic and
    diamagnetic pieces separately (their sum, halved, is the contribution to
    D_s -- see the module docstring for the 1/2)."""
    W = _divided_difference(es,es,T)
    wsc = np.conjugate(ws)
    Arot = [wsc.T@a@ws for a in A]
    para = np.zeros((nd,nd))
    for a in range(nd):
        for b in range(a,nd):
            v = np.sum(W*Arot[a]*Arot[b].T).real
            para[a,b] = v ; para[b,a] = v
    nf = _fermi(es,T)
    dia = np.zeros((nd,nd))
    for (a,b) in B:
        if a>b: continue
        v = np.sum(nf*np.einsum("ij,jk,ki->i",wsc.T,B[(a,b)],ws)).real
        dia[a,b] = v ; dia[b,a] = v
    return para,dia


def superfluid_weight(h,nk=20,T=0.0,ks=None,gauge="atomic"):
    """Superfluid weight tensor from the analytic Kubo formula (see the
    module docstring).  This is the general multiband mean-field result: it
    assumes neither uniform pairing, nor an isolated band, nor
    time-reversal symmetry.

    Parameters
    ----------
    h : Hamiltonian in a Nambu (BdG) mode, dimensionality 1, 2 or 3
    nk : linear number of k-points of the BZ mesh
    T : temperature (same units as the Hamiltonian; 0 is exact, not a limit)
    ks : optional explicit list of k-points, overriding nk
    gauge : "atomic" (default, the physical full-bond-vector twist) or
        "lattice" (the cell-gauge convention of the Peotta/Toermae papers,
        which ignores the orbital positions inside the cell)

    Returns
    -------
    D : real ndarray, shape (dim,dim), in Cartesian twist coordinates (see
        twist_directions for what the axes are)
    """
    ops = TwistOperators(h,gauge=gauge)
    if ks is None: ks = kmesh(ops.dim,nk=nk)
    nd = len(ops.dirs)
    D = np.zeros((nd,nd))
    for k in ks:
        (es,ws) = algebra.eigh(ops.hk(k))
        para,dia = _superfluid_weight_at(es,ws,ops.A(k),ops.B(k),T,nd)
        D = D + 0.5*(para+dia)
    return D/(len(ks)*ops.volume)


# ---------------------------------------------------------------------------
# conventional / geometric decomposition
# ---------------------------------------------------------------------------

def _nambu2block_permutation(h):
    """Index permutation implementing sctk.reorder.nambu2block, i.e. the map
    from the block basis (all electrons, then all holes) to pyqula's
    site-interleaved Nambu basis, so that a matrix can be reordered with
    m[np.ix_(p,p)] instead of two sparse matrix products per call."""
    n = h.intra.shape[0]
    if h.check_mode("spinful_nambu"):
        ns = n//4 # sites
        p = np.zeros(n,dtype=int)
        for i in range(ns):
            p[2*i] = 4*i ; p[2*i+1] = 4*i+1          # electrons, up/down
            p[2*i+2*ns] = 4*i+2 ; p[2*i+1+2*ns] = 4*i+3  # holes
        return p
    elif h.check_mode("spinless_nambu"):
        ns = n//2
        p = np.zeros(n,dtype=int)
        for i in range(ns):
            p[i] = 2*i ; p[i+ns] = 2*i+1
        return p
    else: raise ValueError("not a Nambu Hamiltonian")


def _check_decomposition_assumptions(hb,ab,bb,tol):
    """Verify, at one k-point and in the block basis, the two assumptions
    that make the conventional/geometric split meaningful: (i) the anomalous
    block is Delta times the identity (uniform on-site pairing) and (ii) the
    hole block is minus the electron block (time-reversal symmetry,
    Theta h(-k) Theta^-1 = h(k)).  Returns Delta."""
    n = hb.shape[0]//2
    he = hb[0:n,0:n] ; hh = hb[n:,n:] ; d = hb[0:n,n:]
    scale = max(np.max(np.abs(hb)),1e-12)
    delta = np.trace(d)/n
    if np.max(np.abs(d-delta*np.identity(n)))>tol*scale:
        raise ValueError("the conventional/geometric decomposition assumes "
            "uniform on-site pairing (anomalous block = Delta*identity); "
            "this Hamiltonian has a non-uniform or non-local pairing. Use "
            "h.get_superfluid_weight() without decompose=True")
    if np.max(np.abs(hh+he))>tol*scale:
        raise ValueError("the conventional/geometric decomposition assumes "
            "time-reversal symmetry of the normal state (the hole block "
            "must be minus the electron block). Use "
            "h.get_superfluid_weight() without decompose=True")
    for m in list(ab)+list(bb): # the twist operators must be block diagonal
        if np.max(np.abs(m[0:n,n:]))>tol*scale:
            raise ValueError("the twist operators have anomalous "
                "components: the pairing is k-dependent, so the "
                "conventional/geometric decomposition does not apply")
    return delta


def _decomposition_at(es_n,J,K,delta,T,nd,tol):
    """Per-k-point conventional/geometric split in the normal-state band
    basis (see the module docstring).  es_n are the normal-state
    eigenvalues (mu already included), J[a] = W^dag v_a W the current
    matrix and K[(a,b)] = W^dag w_ab W its second derivative, both in that
    basis.  Returns (conv,geom)."""
    n = len(es_n)
    E = np.sqrt(es_n**2 + np.abs(delta)**2) # quasiparticle energies
    ratio = np.where(E>0.,es_n/np.where(E>0.,E,1.),0.) # xi/E
    u = np.sqrt(np.clip((1.+ratio)/2.,0.,None))
    v = np.sqrt(np.clip((1.-ratio)/2.,0.,None))
    # coherence-factor overlaps, chi_{m,+}=(u_m,v_m), chi_{m,-}=(-v_m,u_m)
    ov = {(1,1): np.outer(u,u)+np.outer(v,v),
          (1,-1): -np.outer(u,v)+np.outer(v,u),
          (-1,1): -np.outer(v,u)+np.outer(u,v),
          (-1,-1): np.outer(v,v)+np.outer(u,u)}
    ww = {(s,sp): _divided_difference(s*E,sp*E,T)*np.abs(ov[(s,sp)])**2
            for s in (1,-1) for sp in (1,-1)}
    # normal-state band-energy differences, with degeneracies handled
    de = es_n[:,None]-es_n[None,:]
    scale = max(np.max(np.abs(es_n)),np.abs(delta),1e-12)
    degen = np.abs(de)<tol*scale
    np.fill_diagonal(degen,True)
    inv = np.where(degen,0.,1./np.where(degen,1.,de)) # 1/(eps_m-eps_n)
    same = np.identity(n,dtype=bool) # the intraband (conventional) mask
    tanh = -(_fermi(E,T)-_fermi(-E,T)) # tanh(beta E/2), =1 at T=0
    conv = np.zeros((nd,nd)) ; geom = np.zeros((nd,nd))
    for a in range(nd):
        for b in range(a,nd):
            JJ = J[a]*J[b].T # [J_a]_mn [J_b]_nm
            if np.max(np.abs(JJ[degen & ~same]))>tol*scale**2:
                raise ValueError("degenerate normal-state bands with a "
                    "finite interband current: the conventional/geometric "
                    "decomposition is ill defined here (band touching, or "
                    "spin-orbit-mixed Kramers pair)")
            para_c = 0. ; para_g = 0. # paramagnetic, intra- and interband
            for key in ww:
                w = ww[key]*JJ
                para_c += np.sum(np.where(same,w,0.)).real
                para_g += np.sum(np.where(same,0.,w)).real
            # diamagnetic: <m|w_ab|m> = d_a d_b eps_m - (interband sum)
            kmm = np.real(np.diag(K[(a,b)]))
            inter = 2.*np.real(np.sum(np.where(same,0.,JJ*inv),axis=1))
            d2eps = kmm + inter # exact band curvature
            conv[a,b] = conv[b,a] = para_c - np.sum(ratio*tanh*d2eps)
            geom[a,b] = geom[b,a] = para_g - np.sum(ratio*tanh*(kmm-d2eps))
    return conv,geom


def _band_basis_quantities(ops,perm,k,tol):
    """(normal-state eigenvalues, J, K, Delta) at one k-point, after
    checking the assumptions of the decomposition."""
    hb = ops.hk(k)[np.ix_(perm,perm)]
    A = [a[np.ix_(perm,perm)] for a in ops.A(k)]
    nd = len(ops.dirs)
    Bd = {key: ops.B(k)[key][np.ix_(perm,perm)] for key in ops.B(k)}
    delta = _check_decomposition_assumptions(hb,A,list(Bd.values()),tol)
    nb = hb.shape[0]//2
    (es_n,W) = algebra.eigh(hb[0:nb,0:nb]) # normal-state bands
    wc = np.conjugate(W)
    J = [wc.T@a[0:nb,0:nb]@W for a in A]
    K = {key: wc.T@Bd[key][0:nb,0:nb]@W for key in Bd}
    return es_n,J,K,delta


def superfluid_weight_decomposition(h,nk=20,T=0.0,ks=None,tol=1e-6,
        gauge="atomic"):
    """Superfluid weight split into its conventional and quantum-geometric
    contributions.  The split is exact on any k-mesh (conv+geom equals the
    Kubo result of superfluid_weight point by point) but is only *defined*
    under uniform on-site pairing and time-reversal symmetry, both of which
    are verified numerically at every k-point -- see the module docstring
    for the validity limits.

    Returns a dictionary with keys "total", "conventional", "geometric",
    "delta" (the uniform pairing amplitude) and "gauge"."""
    ops = TwistOperators(h,gauge=gauge)
    if ks is None: ks = kmesh(ops.dim,nk=nk)
    perm = _nambu2block_permutation(ops.h)
    nd = len(ops.dirs)
    conv = np.zeros((nd,nd)) ; geom = np.zeros((nd,nd))
    delta = 0.
    for k in ks:
        es_n,J,K,delta = _band_basis_quantities(ops,perm,k,tol)
        c,g = _decomposition_at(es_n,J,K,delta,T,nd,tol)
        conv = conv + 0.5*c ; geom = geom + 0.5*g
    nrm = len(ks)*ops.volume
    conv = conv/nrm ; geom = geom/nrm
    return {"total": conv+geom, "conventional": conv, "geometric": geom,
            "delta": delta, "gauge": gauge}


def superfluid_weight_conventional_closed(h,nk=20,T=0.0,ks=None,tol=1e-6,
        gauge="atomic"):
    """Conventional superfluid weight in the closed form of Liang et al.
    Eq. (21),

      D_conv = (1/(2 V N_k)) sum_km (|D|^2/E_m^2)
               [tanh(beta E_m/2)/E_m - (beta/2) sech^2(beta E_m/2)]
               d_a eps_m d_b eps_m ,

    which follows from the implemented split by a BZ integration by parts.
    It needs no interband denominators, so it stays finite at band
    touchings, but it agrees with superfluid_weight_decomposition's
    "conventional" only up to the discretisation error of that integration
    by parts.  Provided as an independent cross-check."""
    ops = TwistOperators(h,gauge=gauge)
    if ks is None: ks = kmesh(ops.dim,nk=nk)
    perm = _nambu2block_permutation(ops.h)
    nd = len(ops.dirs)
    D = np.zeros((nd,nd))
    for k in ks:
        es_n,J,K,delta = _band_basis_quantities(ops,perm,k,tol)
        ve = [np.real(np.diag(j)) for j in J] # band velocities d eps/d K
        E = np.sqrt(es_n**2+np.abs(delta)**2)
        tanh = -(_fermi(E,T)-_fermi(-E,T))
        w = np.abs(delta)**2/E**2*tanh/E
        if T>0.: # the -(beta/2) sech^2 piece, zero at T=0
            sech2 = 1./np.cosh(np.clip(E/(2.*T),None,300.))**2
            w = w - np.abs(delta)**2/E**2*sech2/(2.*T)
        for a in range(nd):
            for b in range(a,nd):
                v = 0.5*np.sum(w*ve[a]*ve[b])
                D[a,b] += v
                if a!=b: D[b,a] += v
    return D/(len(ks)*ops.volume)


# ---------------------------------------------------------------------------
# BKT temperature
# ---------------------------------------------------------------------------

def _scalar_stiffness(D):
    """Scalar stiffness entering the BKT criterion, sqrt(det D_s) -- the
    standard way of reducing an anisotropic 2d stiffness tensor to the
    single number of the isotropic BKT theory (it is the stiffness of the
    system rescaled to isotropy), and equal to D_xx when D is isotropic."""
    d = np.linalg.det(D)
    return np.sqrt(d) if d>0. else 0.


def bkt_temperature(h,nk=20,tmax=None,tol=1e-6,maxite=60,**kwargs):
    """Berezinskii-Kosterlitz-Thouless temperature of a 2d BdG Hamiltonian,
    from the Nelson-Kosterlitz criterion

        T_BKT = (pi/8) D_s(T_BKT)

    solved by bisection.  |Delta| is held at the value stored in h (its
    self-consistent T=0 value if h came out of an SCF): there is no Delta(T)
    feedback, so this is the standard "frozen gap" BKT estimate, which
    overestimates T_BKT when it approaches the mean-field T_c.

    Returns 0.0 if the criterion has no solution (D_s(0) too small)."""
    if h.dimensionality!=2:
        raise NotImplementedError("the BKT temperature is only defined for "
                "dimensionality 2")
    ks = kmesh(2,nk=nk)
    def f(T): # T - (pi/8) D_s(T), monotonically increasing in T
        return T - np.pi/8.*_scalar_stiffness(
                superfluid_weight(h,ks=ks,T=T,**kwargs))
    if tmax is None: tmax = np.pi/8.*_scalar_stiffness(
            superfluid_weight(h,ks=ks,T=0.,**kwargs))
    if tmax<=0.: return 0.0
    if f(tmax)<0.: return tmax # stiffness still above the line at tmax
    (t0,t1) = (0.,tmax)
    for i in range(maxite):
        tm = (t0+t1)/2.
        if f(tm)>0.: t1 = tm
        else: t0 = tm
        if (t1-t0)<tol*tmax: break
    return (t0+t1)/2.
