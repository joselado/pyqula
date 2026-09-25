# Multiband (non-Abelian) quantum geometric tensor for multiorbital
# tight-binding Hamiltonians, via the sum-over-states Kubo formula
#
#   Q_{ij}^{mn}(k) = sum_{l not in S} <u_m|dH/dk_i|u_l><u_l|dH/dk_j|u_n>
#                    / [(E_m - E_l)(E_n - E_l)]
#
# for a chosen band subspace S (occ_idxs), with l running over its
# complement. This is the sum-over-states form of the projector-based
# definition Q_ij(k) = <partial_i u|(1-P)|partial_j u> (Provost & Vallee,
# Commun. Math. Phys. 76, 289 (1980); see also Resta, Rev. Mod. Phys. 66,
# 899 (1994), and the multiband/non-Abelian projector generalization
# reviewed e.g. in Yu, Bernevig, Queiroz, Rossi, Toermae, Yang, "Quantum
# Geometry in Quantum Materials", arXiv:2501.00098, Eqs. (1) and (6)).
#
# Because only states outside S enter the energy denominators, this stays
# well defined when S itself contains an exactly (or nearly) degenerate
# multiplet -- e.g. a spin-degenerate pair, or several orbitals meeting at
# a high-symmetry point -- which a single-band (non-degenerate) Kubo
# formula cannot handle. The quantum metric (symmetric part) and Berry
# curvature (antisymmetric part) follow from Q as in the Abelian case.
#
# The band-resolved Q_ij^{mn} is only gauge covariant: a unitary rotation U
# of the states of S, which any eigensolver is free to pick inside a
# degenerate multiplet, turns it into U^dag Q U. So the non-Abelian tensor
# returned here is the gauge-independent one, written in the orbital basis,
#
#   Q_ij(k) = sum_{m,n in S} |u_m> Q_ij^{mn} <u_n| = P d_iP d_jP P,
#
# with P the projector on S. Its trace is the Abelian tensor, and the
# band-resolved tensor in any basis |v_m> of S is <v_m|Q_ij|v_n>.
#
# ORBITAL POSITIONS (the gauge argument). The tensor is built from the
# cell-periodic Bloch states, and these depend on where the orbitals sit
# inside the unit cell. pyqula's hk_gen writes H(k) with the Bloch phase of
# the lattice vector alone, exp(2 pi i k.R) (the "lattice" gauge, all
# orbitals effectively at the cell origin), whereas the Bloch Hamiltonian of
# orbitals at their actual positions carries the full bond vector,
# exp(2 pi i k.(R + f_j - f_i)) with f the fractional orbital coordinates
# (the "atomic" gauge). The two are related by the k-dependent diagonal
# unitary D(k) = diag(exp(2 pi i k.f_j)), H_A = D^dag H_L D, so their
# projectors differ by more than a change of basis, and so do the pointwise
# Berry curvature, the pointwise quantum metric and the BZ integral of the
# metric; only the Chern number is the same in both. The atomic one is the
# physical one: it respects the point group (the lattice-gauge Berry
# curvature is not C3 symmetric on the honeycomb lattice) and it does not
# depend on which unit cell describes the crystal (the BZ average of Tr g,
# the Marzari-Vanderbilt gauge-invariant spread, doubles under a doubled
# cell only in the atomic gauge), both checked in
# tests/topology/test_quantum_geometric_tensor.py. This is the dependence
# on orbital positions of Simon & Rudner, PRB 102, 165148 (2020), and of
# Huhtinen, Herzog-Arbeitman, Chew, Bernevig & Toermae, PRB 106, 014518
# (2022), arXiv:2203.11133, and it is the convention of PythTB, whose
# Bloch Hamiltonian carries tau_j - tau_i in the phase. gauge="atomic" is
# the default; gauge="lattice" is kept for comparisons with results derived
# in the lattice convention, e.g. the flat-band superfluid-weight identity
# of Liang et al. checked in tests/superfluid.
#
# The atomic gauge needs no second Bloch generator. Since
# dH_A/dk_i = D^dag (dH_L/dk_i + 2 pi i [H_L, F_i]) D, with F_i the diagonal
# matrix of the fractional coordinates along a_i, and since
# <m|[H_L,F_i]|l> = (E_m - E_l) <m|F_i|l> in the eigenbasis of H_L, every
# energy-weighted matrix element of the formula above just picks up
# 2 pi i <m|F_i|l>, and the eigenvectors of H_A are D^dag times those of
# H_L, which is how the orbital-basis tensor is brought to the atomic basis.
#
# Numerical conventions (denominator convention, Abelian
# g_ij=Re Q_ij/Omega_ij=-2 Im Q_ij, non-Abelian
# Omega_ij=i(Q_ij-Q_ij^dag)/g_ij=(Q_ij+Q_ij^dag)/2) match
# PythTB's TBModel.quantum_geometric_tensor/berry_curvature/quantum_metric
# (GPLv3, https://github.com/pythtb/pythtb), and in the atomic gauge so do
# the values: PythTB 2.0.2 on the Haldane model with a sublattice mass
# agrees with this module to all printed digits. Correctness is checked in
# tests/topology/test_quantum_geometric_tensor.py against the Chern number
# of the independent Wilson-loop method (topology.chern), the
# model-independent quantum geometric bounds (e.g. Roy, PRB 90, 165139
# (2014)), a finite difference of the projector in both gauges, the
# closed-form two-band and SSH results, the exact massive-Dirac values at
# the K point, the C3 symmetry and the supercell invariance above, and the
# spin Chern numbers of the Kane-Mele model.
import numpy as np
from .. import algebra
from .. import klist


_gauges = ("atomic","lattice") # accepted values of gauge=, see above

_chunk_bytes = 2**28 # memory budget of one batch of k-points in _qgt_batch


def _check_gauge(gauge):
    """Raise unless gauge is one of the accepted values"""
    if gauge not in _gauges: raise ValueError("unknown gauge '"+str(gauge)
            +"'; it must be one of "+str(list(_gauges)))


def _orbital_fractions(h,n):
    """Fractional coordinates f[orbital,i] of every orbital along the
    periodic lattice vectors a_i, the positions entering the atomic gauge
    (see the module comment). They are solved for directly from the
    lattice vectors, r = sum_i f_i a_i plus a part outside the lattice
    that carries no Bloch phase, so a chain or a plane need not be aligned
    with the Cartesian axes. pyqula orders the basis as (site,spin) --
    (site,spin,electron-hole) for Nambu -- with the site index slowest,
    and every component of a site sits at that site: the hole component
    c^dag_{-k} of a Nambu spinor carries the same position phase as c_k."""
    g = h.geometry
    dim = h.dimensionality
    A = np.array([g.a1,g.a2,g.a3][:dim],dtype=float) # (dim,3)
    r = np.array(g.r,dtype=float) # (nsites,3)
    if n%len(r)!=0: raise ValueError("the Hamiltonian dimension "+str(n)
            +" is not a multiple of the number of sites "+str(len(r))
            +", so its orbitals cannot be given positions")
    f = r@A.T@np.linalg.inv(A@A.T) # (nsites,dim)
    return np.repeat(f,n//len(r),axis=0) # (n,dim), one row per orbital


def _multicell_and_orders(h,gauge="atomic"):
    """Return a multicell copy of h, the list of derivative "orders"
    (multicell.derivative's convention) for dH/dk_i (one per periodic
    direction), the Bloch Hamiltonian generator hk_gen=hm.get_hk_gen(),
    built once and reused at every k-point below (rebuilding it inside a
    per-k-point function did real setup work -- filtering/densifying
    every hopping -- on every call, ~65x slower on a mesh sweep), a
    characteristic hopping energy scale used to make degeneracy_tol
    physically meaningful (see _qgt_batch), and the fractional orbital
    positions for gauge="atomic" (None for gauge="lattice").

    h.get_multicell() can hand back matrices stored as the legacy
    numpy.matrix (whose "*" operator means matrix product, not elementwise,
    unlike plain numpy.ndarray) -- e.g. Hamiltonians built via
    get_supercell() keep their hoppings as numpy.matrix. Converting
    hm.intra and every hopping's .m to numpy.ndarray once, here, means
    every elementwise operation downstream in this module (see
    _hk_derivatives_batch and _qgt_batch) only ever sees plain ndarrays --
    no per-k-point or per-use patching needed. h.get_multicell() returns a
    copy, so converting its matrices does not touch the caller's
    Hamiltonian."""
    _check_gauge(gauge)
    dim = h.dimensionality
    if dim not in (1,2,3): raise ValueError("the quantum geometric tensor "
        "needs a periodic Hamiltonian (dimensionality 1, 2 or 3), and this "
        "one has dimensionality "+str(dim))
    hm = h.get_multicell() # a copy, modified below
    # algebra.todense, not np.asarray: the latter turns a scipy sparse
    # matrix (what get_supercell() stores) into a 0-d object array
    hm.intra = algebra.todense(hm.intra)
    for t in hm.hopping: t.m = algebra.todense(t.m)
    orders = [[int(a==i) for a in range(dim)] for i in range(dim)] # d/dk_i
    hkgen = hm.get_hk_gen() # build once, reuse at every k-point
    scale = max(np.max(np.abs(hm.intra)),
                max((np.max(np.abs(t.m)) for t in hm.hopping),default=0.0),
                1e-12) # characteristic hopping energy scale (floored so a
                       # Hamiltonian with a zero intra/hopping norm still
                       # gets a small but nonzero scale)
    if gauge=="atomic": frac = _orbital_fractions(h,hm.intra.shape[0])
    else: frac = None
    return hm,orders,hkgen,scale,frac


def _karray(ks,dim):
    """k-points as a (nk,dim) array of their periodic components, padding
    a k-point given with fewer than three components with zeros"""
    return np.array([(list(k)+[0.,0.,0.])[:3] for k in ks],
            dtype=float)[:,:dim]


def _hk_derivatives_batch(hm,orders,ks):
    """Exact analytic k-derivatives dH/dk_i of a multicell Hamiltonian's
    Bloch matrix at every k in ks, shape (nk,dim,n,n). Since
    H(k) = intra + sum_R t_R exp(2 pi i k.R), dH/dk_i = sum_R
    (2 pi i R_i) t_R exp(2 pi i k.R), with k in reduced coordinates; this
    is current.hk_derivative (multicell.derivative times its missing
    2*pi) evaluated for all k-points at once, in the lattice gauge of
    hk_gen, and it is checked against a finite difference of the projector
    in tests/topology/test_quantum_geometric_tensor.py."""
    dim = len(orders)
    ks = _karray(ks,dim)
    n = hm.intra.shape[0]
    out = np.zeros((len(ks),dim,n,n),dtype=np.complex128)
    for t in hm.hopping:
        d = np.array(t.dir,dtype=float)[:dim]
        phase = np.exp(2j*np.pi*(ks@d)) # (nk,)
        for i,o in enumerate(orders):
            pref = np.prod([(2j*np.pi*d[a])**o[a] for a in range(dim)])
            if pref==0.: continue
            out[:,i] += phase[:,None,None]*(pref*np.asarray(t.m))[None]
    return out


def _qgt_batch(hm,orders,hkgen,ks,occ_idxs,non_abelian,degeneracy_tol,
        scale,frac=None):
    """Core computation for a batch of k-points, given an already-multicell
    Hamiltonian, derivative orders and Bloch generator, in the atomic
    gauge if the fractional orbital positions frac are given and in the
    lattice gauge otherwise (see the module comment). Returns an array
    with one tensor per k-point, (nk,dim,dim) for the trace over the
    subspace or (nk,dim,dim,n,n) for the orbital-basis non-Abelian tensor.

    The diagonalization is batched (htk.eigenvectors), which is safe
    because nothing returned depends on the basis the solver picks inside
    a degenerate multiplet: the band-resolved tensor is built in whatever
    basis comes out and then sandwiched back into the orbital basis.

    degeneracy_tol is interpreted as *relative* to `scale` (the
    Hamiltonian's characteristic hopping energy, from _multicell_and_orders)
    rather than as an absolute energy: an absolute tolerance would be
    meaningless across Hamiltonians at different energy scales -- e.g. it
    would silently accept a nonzero-but-numerically-unresolvable gap on a
    Hamiltonian with O(1) hoppings (letting the sum-over-states denominator
    blow the tensor up to a huge, effectively-noise-dominated value instead
    of raising), while over-eagerly flagging a perfectly healthy gap as
    degenerate on a meV-scale Hamiltonian."""
    from ..htk.eigenvectors import peigh_bloch
    dim = len(orders)
    (es,ws) = peigh_bloch(hkgen,ks) # ws[k][:,n] eigenvector of es[k][n]
    nk,n = es.shape
    occ_idxs = np.array(occ_idxs,dtype=int)
    cond_idxs = np.setdiff1d(np.arange(n),occ_idxs)
    if len(cond_idxs)==0: # subspace is everything, nothing to project onto
        if non_abelian:
            return np.zeros((nk,dim,dim,n,n),dtype=np.complex128)
        return np.zeros((nk,dim,dim),dtype=np.complex128)
    Eo = es[:,occ_idxs]; Ec = es[:,cond_idxs]
    denom = Eo[:,:,None] - Ec[:,None,:] # (nk,n_occ,n_cond)
    if np.any(np.abs(denom)<degeneracy_tol*scale):
        raise ValueError("Degenerate bands across occ_idxs and its "
            "complement: the quantum geometric tensor requires a gap "
            "between the chosen subspace and the rest of the spectrum")
    inv_oc = 1./denom
    dhs = _hk_derivatives_batch(hm,orders,ks) # (nk,dim,n,n)
    wo = ws[:,:,occ_idxs] # (nk,n,n_occ)
    wc = ws[:,:,cond_idxs] # (nk,n,n_cond)
    wod = np.conj(wo.transpose(0,2,1)) # (nk,n_occ,n)
    # <m|dH_i|l> and <l|dH_j|n>, each divided by its energy difference
    voc = wod[:,None]@dhs@wc[:,None] # (nk,dim,no,nc)
    voc = voc*inv_oc[:,None]
    vco = np.conj(wc.transpose(0,2,1))[:,None]@dhs@wo[:,None] # (nk,dim,nc,no)
    vco = vco*inv_oc.transpose(0,2,1)[:,None]
    if frac is not None: # atomic gauge: add 2 pi i <m|F_i|l>, see above
        xoc = np.array([(wod*frac[:,i][None,None,:])@wc
                        for i in range(dim)]).transpose(1,0,2,3)
        voc = voc + 2j*np.pi*xoc
        vco = vco - 2j*np.pi*np.conj(xoc.transpose(0,1,3,2))
    Q = voc[:,:,None]@vco[:,None,:] # (nk,dim,dim,no,no)
    if not non_abelian: return np.trace(Q,axis1=-2,axis2=-1)
    # back to the orbital basis, where the solver's gauge drops out
    Q = wo[:,None,None]@Q@wod[:,None,None]
    if frac is not None: # and to the atomic Bloch basis, Q -> D^dag Q D
        ph = np.exp(2j*np.pi*(_karray(ks,dim)@frac.T)) # (nk,n), D=diag(ph)
        Q = np.conj(ph)[:,None,None,:,None]*Q*ph[:,None,None,None,:]
    return Q


def quantum_geometric_tensor_k(h,k=[0.,0.,0.],occ_idxs=None,
        non_abelian=False,degeneracy_tol=1e-8,gauge="atomic"):
    """Quantum geometric tensor of a multiorbital Bloch Hamiltonian at a
    single k-point.

    occ_idxs selects the band subspace S (default: the bands with E<0,
    matching the E<0 "occupied"/Fermi-level convention used everywhere
    else in this codebase, e.g. topologytk/occstates.py's occupied_states
    and topologytk/operatorberry.py -- not just the lower half of the
    bands, so this tracks h.shift_fermi(...) the same way h.get_chern()
    does). Set non_abelian to True to get the full band-pair-resolved
    tensor instead of its trace (sum_{m in S} Q_ij^{mm}) over the
    subspace. It is returned in the orbital basis,
    Q_ij = sum_{m,n in S} |u_m> Q_ij^{mn} <u_n|, which does not depend on
    the basis the diagonalization picks inside a degenerate multiplet;
    the band-resolved Q_ij^{mn} in a basis |v_m> of S of your choice is
    <v_m|Q_ij|v_n>.

    gauge="atomic" (default) places every orbital at its position in the
    geometry, which is the physical quantum geometry; gauge="lattice"
    drops the positions from the Bloch phase, as h.get_hk_gen() does (see
    the module comment). The derivatives are with respect to k in reduced
    coordinates, k = sum_i k_i b_i with b_i the reciprocal lattice vectors.

    degeneracy_tol is relative to the Hamiltonian's characteristic hopping
    energy scale (see _multicell_and_orders/_qgt_batch),
    not an absolute energy.

    Returns
    -------
    Q : ndarray, complex
      shape (dim,dim,n,n) if non_abelian, n the number of orbitals
      shape (dim,dim) (trace over the subspace) otherwise
    """
    hm,orders,hkgen,scale,frac = _multicell_and_orders(h,gauge=gauge)
    occ_idxs = _resolve_occ_idxs(hkgen,k,occ_idxs)
    return _qgt_batch(hm,orders,hkgen,[k],occ_idxs,non_abelian,
            degeneracy_tol,scale,frac)[0]


def berry_curvature_from_qgt(Q,non_abelian=False):
    """Berry curvature tensor (antisymmetric part) of a QGT array."""
    if non_abelian: return 1j*(Q - np.conjugate(np.swapaxes(Q,-1,-2)))
    return -2.*Q.imag


def quantum_metric_from_qgt(Q,non_abelian=False):
    """Quantum metric tensor (symmetric part) of a QGT array."""
    if non_abelian: return 0.5*(Q + np.conjugate(np.swapaxes(Q,-1,-2)))
    return Q.real


def _resolve_occ_idxs(hkgen,k,occ_idxs):
    """If occ_idxs is None, fix it once from the E<0 bands at a single
    reference k-point -- the caller passes the first point of the
    path/mesh, Gamma for klist.kmesh -- instead of letting
    quantum_geometric_tensor_k re-resolve "E<0" independently at every
    k-point: if the occupied-band count happened to change across the
    loop (e.g. a stray k-point sitting right at a band edge, an easy trap
    since shift_fermi only guarantees E=0 lies in the gap at generic k,
    not that it clears every band edge by a wide margin) that would
    silently change the shape/meaning of the subspace rather than raise
    -- fixing the index set up front turns any such inconsistency into
    the loud degeneracy_tol ValueError instead. Gamma is not necessarily
    a safe reference for every model (e.g. a flat band touching the
    dispersive band exactly at Gamma): if the subspace at that reference
    point is not gapped, this raises there too -- pass occ_idxs explicitly
    for such a model instead of relying on the default."""
    if occ_idxs is not None: return occ_idxs
    es = algebra.eigh(hkgen(k))[0]
    return np.where(es<0.0)[0]


def _chunk_size(n,dim):
    """Number of k-points per batch in _qgt_over_kpoints, so that the
    batch's arrays -- Bloch matrices, eigenvectors, derivatives and the
    orbital-basis output, roughly (dim**2+2*dim+3) complex n x n matrices
    per k-point -- stay within _chunk_bytes. A fixed 256 k-points cost
    4.8 GB of peak memory at 400 orbitals."""
    per_k = 16*(dim*dim+2*dim+3)*n*n
    return int(max(1,min(256,_chunk_bytes//per_k)))


def _qgt_over_kpoints(hm,orders,hkgen,ks,occ_idxs,non_abelian,degeneracy_tol,
        scale,frac=None,chunk=None):
    """Shared core of quantum_geometric_tensor_path/_mesh: resolve
    occ_idxs once from the first k-point (see _resolve_occ_idxs) and
    evaluate the QGT at every k in ks, reusing the same hm/orders/hkgen/
    scale/frac throughout. Returns (occ_idxs,Qs).

    The k-points go through _qgt_batch in chunks of `chunk` (by default
    sized from the number of orbitals, see _chunk_size), which bounds the
    memory of the per-k-point arrays without affecting the result. This
    used to be a serial per-k-point loop, because the non-Abelian tensor
    was returned in the band basis and a batched solver picks a different
    basis inside a degenerate multiplet; it is now returned in the orbital
    basis, where that choice drops out."""
    occ_idxs = _resolve_occ_idxs(hkgen,ks[0],occ_idxs)
    if chunk is None: chunk = _chunk_size(hm.intra.shape[0],len(orders))
    Qs = [_qgt_batch(hm,orders,hkgen,ks[i:i+chunk],occ_idxs,non_abelian,
            degeneracy_tol,scale,frac) for i in range(0,len(ks),chunk)]
    return occ_idxs,np.concatenate(Qs,axis=0)


def quantum_geometric_tensor_path(h,kpath=None,nk=100,occ_idxs=None,
        non_abelian=False,degeneracy_tol=1e-8,gauge="atomic"):
    """quantum_geometric_tensor_k evaluated along a k-path. Returns the
    path index, the quantum metric and the Berry curvature at each point"""
    hm,orders,hkgen,scale,frac = _multicell_and_orders(h,gauge=gauge)
    kpath = klist.get_kpath(h.geometry,kpath=kpath,nk=nk)
    occ_idxs,Qs = _qgt_over_kpoints(hm,orders,hkgen,kpath,occ_idxs,
            non_abelian,degeneracy_tol,scale,frac)
    g = quantum_metric_from_qgt(Qs,non_abelian=non_abelian)
    omega = berry_curvature_from_qgt(Qs,non_abelian=non_abelian)
    inds = np.array(range(len(Qs)))
    return inds,g,omega


def quantum_geometric_tensor_mesh(h,nk=30,occ_idxs=None,non_abelian=False,
        degeneracy_tol=1e-8,gauge="atomic"):
    """quantum_geometric_tensor_k evaluated on a uniform k-mesh of nk points
    per periodic direction. Returns the k-points and the QGT at every
    point, e.g. for BZ integration or pointwise validation."""
    hm,orders,hkgen,scale,frac = _multicell_and_orders(h,gauge=gauge)
    ks = klist.kmesh(h.dimensionality,nk=nk)
    occ_idxs,Qs = _qgt_over_kpoints(hm,orders,hkgen,ks,occ_idxs,
            non_abelian,degeneracy_tol,scale,frac)
    return ks,Qs


def chern_from_qgt(h,nk=30,occ_idxs=None,gauge="atomic"):
    """Chern number of the chosen band subspace (default: the E<0 bands,
    see quantum_geometric_tensor_k), obtained by integrating the xy
    component of the Berry curvature that
    comes out of the sum-over-states quantum geometric tensor over a
    uniform BZ mesh: C = (1/2pi) sum_k Omega_xy(k) dkx dky. This offers an
    independent cross-check of quantum_geometric_tensor_k against
    topology.chern (Fukui-Hatsugai-Suzuki Wilson-loop method). Both gauges
    give the same Chern number, since the two Berry curvatures differ by
    the curl of a periodic function."""
    if h.dimensionality!=2: raise ValueError("the Chern number needs a "
        "two-dimensional Hamiltonian, and this one has dimensionality "
        +str(h.dimensionality))
    ks,Qs = quantum_geometric_tensor_mesh(h,nk=nk,occ_idxs=occ_idxs,
                non_abelian=False,gauge=gauge)
    omega_xy = berry_curvature_from_qgt(Qs,non_abelian=False)[:,0,1]
    dA = 1./(nk*nk) # reduced-coordinate area element (full BZ area = 1)
    return (np.sum(omega_xy)*dA/(2.*np.pi)).real
