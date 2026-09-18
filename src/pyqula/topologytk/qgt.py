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
# Numerical conventions (denominator convention, Abelian
# g_ij=Re Q_ij/Omega_ij=-2 Im Q_ij, non-Abelian
# Omega_ij=i(Q_ij-Q_ij^dag)/g_ij=(Q_ij+Q_ij^dag)/2) match
# PythTB's tb_model.quantum_geometric_tensor/berry_curvature/quantum_metric
# (GPLv3, https://github.com/pythtb/pythtb). Correctness here is checked in
# tests/topology/test_quantum_geometric_tensor.py against: (1) the Chern
# number of pyqula's own, independently implemented Fukui-Hatsugai-Suzuki
# Wilson-loop method, topology.chern; (2) the model-independent quantum
# geometric bounds (e.g. Roy, PRB 90, 165139 (2014)); and (3) for the
# spinless Haldane model (an exactly solvable two-band d(k).sigma model)
# the closed-form quantum metric/Berry curvature in terms of the
# normalized Bloch vector d_hat(k), which pins the absolute scale of the
# quantum metric (the two checks above are both invariant under an
# overall rescaling of Q, so neither alone would catch a scale error).
import numpy as np
from .. import algebra
from .. import klist


def _multicell_and_orders(h):
    """Return a multicell copy of h, the list of derivative "orders"
    (multicell.derivative's convention) for dH/dk_i (one per spatial
    dimension), the Bloch Hamiltonian generator hk_gen=hm.get_hk_gen(),
    built once and reused at every k-point below (rebuilding it inside a
    per-k-point function did real setup work -- filtering/densifying
    every hopping -- on every call, ~65x slower on a mesh sweep), and a
    characteristic hopping energy scale used to make degeneracy_tol
    physically meaningful (see _qgt_batch).

    h.get_multicell() can hand back matrices stored as the legacy
    numpy.matrix (whose "*" operator means matrix product, not elementwise,
    unlike plain numpy.ndarray) -- e.g. Hamiltonians built via
    get_supercell() keep their hoppings as numpy.matrix. Converting
    hm.intra and every hopping's .m to numpy.ndarray once, here, means
    multicell.derivative and every elementwise operation downstream in
    this module (see _hk_derivatives_batch and
    _qgt_batch) only ever see plain ndarrays -- no
    per-k-point or per-use patching needed. h.get_multicell() returns a
    copy, so converting its matrices does not touch the caller's
    Hamiltonian."""
    hm = h.get_multicell() # a copy, modified below
    # algebra.todense, not np.asarray: the latter turns a scipy sparse
    # matrix (what get_supercell() stores) into a 0-d object array
    hm.intra = algebra.todense(hm.intra)
    for t in hm.hopping: t.m = algebra.todense(t.m)
    dim = h.dimensionality
    if dim==1: orders = [[1]]
    elif dim==2: orders = [[1,0],[0,1]]
    else: raise NotImplementedError(
        "quantum geometric tensor only implemented for dimensionality 1 or 2")
    hkgen = hm.get_hk_gen() # build once, reuse at every k-point
    scale = max(np.max(np.abs(hm.intra)),
                max((np.max(np.abs(t.m)) for t in hm.hopping),default=0.0),
                1e-12) # characteristic hopping energy scale (floored so a
                       # Hamiltonian with a zero intra/hopping norm still
                       # gets a small but nonzero scale)
    return hm,orders,hkgen,scale


def _hk_derivatives_batch(hm,orders,ks):
    """Exact analytic k-derivatives dH/dk_i of a multicell Hamiltonian's
    Bloch matrix at every k in ks, shape (nk,dim,n,n). Since
    H(k) = intra + sum_R t_R exp(2 pi i k.R), dH/dk_i = sum_R
    (2 pi i R_i) t_R exp(2 pi i k.R), with k in reduced coordinates; this
    is current.hk_derivative (multicell.derivative times its missing
    2*pi) evaluated for all k-points at once, and it is checked against a
    finite difference of the projector in
    tests/topology/test_quantum_geometric_tensor.py."""
    ks = np.array([np.array(k,dtype=float)[:3] for k in ks])
    n = hm.intra.shape[0]
    dim = len(orders)
    out = np.zeros((len(ks),dim,n,n),dtype=np.complex128)
    for t in hm.hopping:
        d = np.array(t.dir,dtype=float)
        phase = np.exp(2j*np.pi*(ks[:,:len(d)]@d[:ks.shape[1]])) # (nk,)
        for i,o in enumerate(orders):
            pref = np.prod([(2j*np.pi*d[a])**o[a] for a in range(len(o))])
            if pref==0.: continue
            out[:,i] += phase[:,None,None]*(pref*np.asarray(t.m))[None]
    return out


def _qgt_batch(hm,orders,hkgen,ks,occ_idxs,non_abelian,degeneracy_tol,
        scale):
    """Core computation for a batch of k-points, given an already-multicell
    Hamiltonian, derivative orders and Bloch generator. Returns an array
    with one tensor per k-point, (nk,dim,dim) for the trace over the
    subspace or (nk,dim,dim,n,n) for the orbital-basis non-Abelian tensor
    (see the module comment).

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
    nocc = len(occ_idxs)
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
    # <m|dH_i|l> and <l|dH_j|n>, each divided by its energy difference
    voc = np.conj(wo.transpose(0,2,1))[:,None]@dhs@wc[:,None] # (nk,dim,no,nc)
    voc = voc*inv_oc[:,None]
    vco = np.conj(wc.transpose(0,2,1))[:,None]@dhs@wo[:,None] # (nk,dim,nc,no)
    vco = vco*inv_oc.transpose(0,2,1)[:,None]
    Q = voc[:,:,None]@vco[:,None,:] # (nk,dim,dim,no,no)
    if not non_abelian: return np.trace(Q,axis1=-2,axis2=-1)
    # back to the orbital basis, where the solver's gauge drops out
    return wo[:,None,None]@Q@np.conj(wo.transpose(0,2,1))[:,None,None]


def quantum_geometric_tensor_k(h,k=[0.,0.,0.],occ_idxs=None,
        non_abelian=False,degeneracy_tol=1e-8):
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

    degeneracy_tol is relative to the Hamiltonian's characteristic hopping
    energy scale (see _multicell_and_orders/_qgt_batch),
    not an absolute energy.

    Returns
    -------
    Q : ndarray, complex
      shape (dim,dim,n,n) if non_abelian, n the number of orbitals
      shape (dim,dim) (trace over the subspace) otherwise
    """
    hm,orders,hkgen,scale = _multicell_and_orders(h)
    occ_idxs = _resolve_occ_idxs(hkgen,k,occ_idxs)
    return _qgt_batch(hm,orders,hkgen,[k],occ_idxs,non_abelian,
            degeneracy_tol,scale)[0]


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


def _qgt_over_kpoints(hm,orders,hkgen,ks,occ_idxs,non_abelian,degeneracy_tol,
        scale,chunk=256):
    """Shared core of quantum_geometric_tensor_path/_mesh: resolve
    occ_idxs once from the first k-point (see _resolve_occ_idxs) and
    evaluate the QGT at every k in ks, reusing the same hm/orders/hkgen/
    scale throughout. Returns (occ_idxs,Qs).

    The k-points go through _qgt_batch in chunks of `chunk`, which bounds
    the memory of the (nk,dim,n,n) derivative and eigenvector arrays. This
    used to be a serial per-k-point loop, because the non-Abelian tensor
    was returned in the band basis and a batched solver picks a different
    basis inside a degenerate multiplet; it is now returned in the orbital
    basis, where that choice drops out."""
    occ_idxs = _resolve_occ_idxs(hkgen,ks[0],occ_idxs)
    Qs = [_qgt_batch(hm,orders,hkgen,ks[i:i+chunk],occ_idxs,non_abelian,
            degeneracy_tol,scale) for i in range(0,len(ks),chunk)]
    return occ_idxs,np.concatenate(Qs,axis=0)


def quantum_geometric_tensor_path(h,kpath=None,nk=100,occ_idxs=None,
        non_abelian=False,degeneracy_tol=1e-8):
    """quantum_geometric_tensor_k evaluated along a k-path. Returns the
    path index, the quantum metric and the Berry curvature at each point"""
    hm,orders,hkgen,scale = _multicell_and_orders(h) # build once, not per k
    kpath = klist.get_kpath(h.geometry,kpath=kpath,nk=nk)
    occ_idxs,Qs = _qgt_over_kpoints(hm,orders,hkgen,kpath,occ_idxs,
            non_abelian,degeneracy_tol,scale)
    g = quantum_metric_from_qgt(Qs,non_abelian=non_abelian)
    omega = berry_curvature_from_qgt(Qs,non_abelian=non_abelian)
    inds = np.array(range(len(Qs)))
    return inds,g,omega


def quantum_geometric_tensor_mesh(h,nk=30,occ_idxs=None,non_abelian=False,
        degeneracy_tol=1e-8):
    """quantum_geometric_tensor_k evaluated on a uniform 2D k-mesh (only
    for dimensionality 2 Hamiltonians). Returns the k-points and the QGT
    at every point, e.g. for BZ integration or pointwise validation."""
    if h.dimensionality!=2: raise NotImplementedError(
        "quantum geometric tensor mesh only implemented for dimensionality 2")
    hm,orders,hkgen,scale = _multicell_and_orders(h) # build once, not per k
    ks = klist.kmesh(2,nk=nk)
    occ_idxs,Qs = _qgt_over_kpoints(hm,orders,hkgen,ks,occ_idxs,
            non_abelian,degeneracy_tol,scale)
    return ks,Qs


def chern_from_qgt(h,nk=30,occ_idxs=None):
    """Chern number of the chosen band subspace (default: the E<0 bands,
    see quantum_geometric_tensor_k), obtained by integrating the xy
    component of the Berry curvature that
    comes out of the sum-over-states quantum geometric tensor over a
    uniform BZ mesh: C = (1/2pi) sum_k Omega_xy(k) dkx dky. This offers an
    independent cross-check of quantum_geometric_tensor_k against
    topology.chern (Fukui-Hatsugai-Suzuki Wilson-loop method)."""
    ks,Qs = quantum_geometric_tensor_mesh(h,nk=nk,occ_idxs=occ_idxs,
                non_abelian=False)
    omega_xy = berry_curvature_from_qgt(Qs,non_abelian=False)[:,0,1]
    dA = 1./(nk*nk) # reduced-coordinate area element (full BZ area = 1)
    return (np.sum(omega_xy)*dA/(2.*np.pi)).real
