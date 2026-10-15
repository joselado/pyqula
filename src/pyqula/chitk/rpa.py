import numpy as np
from scipy.optimize import linear_sum_assignment
from .. import algebra
from ..multihopping import MultiHopping


def interaction_at_q(V,h,q):
    """Return the interaction matrix V(q) that RPA dressing needs at a
    given q-vector, given a real-space interaction V. V can be:
     - None -> returned as None
     - a plain matrix/array -> returned unchanged, i.e. treated as a
       q-independent (onsite-only) interaction. This is the only form V
       used to take before extended interactions were supported, so this
       keeps every existing caller (which always passes a plain matrix)
       working identically.
     - a dict (or MultiHopping) {(n1,n2,n3): matrix} -- a real-space
       interaction with support beyond the onsite (0,0,0) term, keyed by
       lattice-vector offset in exactly the same convention as
       Hamiltonian.hopping[i].dir. Returned as the Bloch/Fourier sum
       sum_d V[d]*exp(2 pi i q.d), i.e. the same bloch_phase convention
       multicell.hk_gen already uses to build H(k) from a hopping dict --
       an extended interaction is Fourier transformed exactly like an
       extended hopping.
    q=None (no q-point specified by the caller) is treated as the Gamma
    point (q=0), under which every dict reduces to the plain sum of its
    matrices. The RPA response functions (chi_AB_RPA, chi_ops_RPA) never
    reach here with q=None: their local response dresses every q of the
    mesh on its own, see _local_rpa_response. The pole finders below do,
    and a pole is a per-q quantity, so q=None there is the kernel of the
    averaged bare response at the Gamma point, as it always was."""
    if V is None: return None
    if isinstance(V,MultiHopping): V = V.get_dict()
    if isinstance(V,dict):
        if q is None: q = [0.,0.,0.]
        out = None
        for d,m in V.items():
            phase = h.geometry.bloch_phase(d,q)
            out = m*phase if out is None else out + m*phase
        return out
    return V # plain matrix, q-independent


def _local_rpa_response(h,f,**kwargs):
    """The local (q=None) RPA response: f(q) evaluated at every point of
    the q-mesh chiAB averages over when it is given no q (the k-mesh of
    the geometry, nk points per direction, the same nk the response
    integrates over) and averaged afterwards, so that every q is dressed
    on its own, chi0(q)(1-V(q)chi0(q))^-1, before the average. That is the
    RPA of the local response, the one a probe on a single site sees, and
    what the pair basis (chitk.pairchi, through
    spinchi._pair_route_response) computes too: the two agree to 3e-13 on
    a half-filled Neel Hubbard chain at nk=6, where the site vertex is
    exact. Dressing the averaged bare response with V(q=0) instead, which
    is what chi_AB_RPA/chi_ops_RPA used to do, is the response of nothing:
    on that chain it was off by 18 against a maximum of 19, since the
    averaged bare response has no reason to keep the pole structure of any
    single q. The bare response (V=None) is averaged by chiAB itself over
    the same mesh, so this is only for the dressed one; a 0D geometry and
    nk=1 have a single q-point and are unchanged."""
    qs = h.geometry.get_kmesh(nk=kwargs.get("nk",60)) # chiAB's own mesh
    out = [f(q) for q in qs] # dressed at every q
    return out[0][0],np.mean([o[1] for o in out],axis=0)


# compute general RPA response function
def chi_AB_RPA(h,V=None,q=None,**kwargs):
    """Compute the RPA chi for a hamiltonian. With q=None and an
    interaction, the local response: dressed at every q of the mesh and
    averaged afterwards, see _local_rpa_response"""
    from ..chi import chiAB # get response function
    if V is not None and q is None: # the local dressed response
        return _local_rpa_response(h,lambda qi: chi_AB_RPA(h,V=V,q=qi,
                                                            **kwargs),**kwargs)
    es,chis = chiAB(h,mode="matrix",q=q,**kwargs) # non-interacting response
    iden = np.identity(chis[0].shape[0],dtype=np.complex128) # identity
    if V is not None: # finite interaction, RPA summation
        Vq = interaction_at_q(V,h,q) # Fourier transform if V has non-onsite support
        chis_rpa = [chi@algebra.inv(iden - Vq@chi) for chi in chis]
    else: chis_rpa = chis
    return es,np.array(chis_rpa)


mode_rpa = "vectorized"

def build_ops_projectors(h,ops):
    """Return (pAs,pBs), the per-site operator projector tensor for a list
    of local operators (e.g. Sx,Sy,Sz). This tensor does not depend on q or
    on the frequency grid, so callers that scan many q-points for the same
    operators (e.g. magnon_bands) should build it once with this function
    and reuse it via chi_ops_RPA/rpa_kernel_poles_ops's pAs/pBs arguments,
    instead of paying for the dense operator/projector products again at
    every q."""
    from .. import operators
    nop = len(ops) # number of operators
    pAs = [] # empty list
    pBs = [] # empty list
    projs = [operators.index(h,n=[i]) for i in range(len(h.geometry.r))]
    for i in range(nop): # loop over first operator
        A = ops[i] # first operator
        B = ops[i] # second operator
        A = algebra.todense(h.get_operator(A).get_matrix())
        B = algebra.todense(h.get_operator(B).get_matrix())
        for pi in projs: # products
            pAs.append(pi@A)
            pBs.append(pi@B)
    return pAs,pBs


def _chi_ops_matrix_vectorized(h,ops=None,pAs=None,pBs=None,q=None,**kwargs):
    """Compute the non-interacting response tensor for a list of local
    operators (e.g. Sx,Sy,Sz), evaluated on every lattice site. Shared by
    chi_ops_RPA and rpa_kernel_poles_ops, so both consume the exact same
    operator/projector tensor and stay consistent with each other. If
    pAs/pBs are not given they are built from ops via build_ops_projectors;
    passing them in directly lets a caller looping over many q-points reuse
    the same (q-independent) tensor instead of rebuilding it every time."""
    from ..chi import chiAB # get response function
    if pAs is None or pBs is None:
        pAs,pBs = build_ops_projectors(h,ops)
    return chiAB(h,mode="matrix",pAs=pAs,pBs=pBs,q=q,**kwargs) # non-interacting response

def chi_ops_RPA(h,ops=None,V=None,pAs=None,pBs=None,q=None,**kwargs):
    """Compute the RPA chi for a hamiltonian,
    return a tensor given a list of operators. This is
    for example useful to compute the full spin response
    function. With q=None and an interaction, the local response: dressed
    at every q of the mesh and averaged afterwards, see
    _local_rpa_response"""
    from ..chi import chiAB # get response function
    nop = len(ops) # number of operators
    if V is not None and q is None: # the local dressed response
        if mode_rpa=="vectorized" and (pAs is None or pBs is None):
            pAs,pBs = build_ops_projectors(h,ops) # once, not at every q
        return _local_rpa_response(h,lambda qi: chi_ops_RPA(h,ops=ops,V=V,
                    pAs=pAs,pBs=pBs,q=qi,**kwargs),**kwargs)
    # storage for the full response
    if mode_rpa=="sequential": # one by one
        chis = [[None for i in range(nop)] for j in range(nop)]
        for i in range(nop): # loop over first operator
            for j in range(nop): # loop over second operator
                A = ops[i] # first operator
                B = ops[j] # second operator
                es,chisi = chiAB(h,mode="matrix",A=A,B=B,q=q,
                                **kwargs) # non-interacting response
                chis[i][j] = chisi # store in the list
        # now make it a block matrix, and reshpae accordingly
        chis_tmp = np.array(chis) # convert to array
        chis = [] # empty list
        for i in range(len(es)): # loop over energies
            chi = chis_tmp[:,:,i,:,:] # get this one
            chi = [[chi[i,j,:,:] for i in range(nop)] for j in range(nop)]
            chis.append(np.block(chi)) # store
    elif mode_rpa=="vectorized": # all at once
        es,chis = _chi_ops_matrix_vectorized(h,ops=ops,pAs=pAs,pBs=pBs,q=q,**kwargs)
    else:
        raise ValueError("unknown mode_rpa; the accepted ones are "
                "'sequential' and 'vectorized'")
    iden = np.identity(chis[0].shape[0],dtype=np.complex128) # identity
    if V is not None: # finite interaction, RPA summation
        Vq = interaction_at_q(V,h,q) # Fourier transform if V has non-onsite support
        chis_rpa = [chi@algebra.inv(iden - Vq@chi) for chi in chis]
    else: chis_rpa = chis
    return es,np.array(chis_rpa)




def chi_AB_RPA_scf(scf):
    """Return the RPA response function for an SCF object"""
    return chi_AB_RPA(scf.hamiltonian,scf.v) # scf.v is a hopping dict,
                                              # Fourier transformed by chi_AB_RPA


def _track_eigenvalue_branches(eigs):
    """eigs has shape (nw,N): N complex eigenvalues of the RPA kernel at
    each of the nw frequencies, in the arbitrary order returned by the
    eigensolver at every step independently. Reorder them into continuous
    branches by matching consecutive frequency steps with the assignment
    (Hungarian algorithm) that minimizes the total eigenvalue displacement.
    Near-degenerate crossings can occasionally swap branch labels; this is
    a known limitation of per-step eigenvalue tracking, not solved here."""
    nw,n = eigs.shape
    tracked = np.array(eigs,dtype=np.complex128,copy=True)
    for k in range(1,nw):
        prev = tracked[k-1]
        cur = tracked[k]
        cost = np.abs(prev[:,None] - cur[None,:])
        row_ind,col_ind = linear_sum_assignment(cost)
        order = np.empty(n,dtype=int)
        order[row_ind] = col_ind
        tracked[k] = cur[order]
    return tracked


def _zero_crossings(es,eigs):
    """Return the (npoles,2) [frequency, imaginary part] list of the zeros
    of the eigenvalue branches eigs (shape (nw,nbranch)) of an RPA kernel,
    1 - V chi0 here or 1 + K chi0 in chitk/pairchi.py, on the grid es.

    A sign change of Re lambda is either a zero of lambda or a pole of
    it, the eigenvalue going through infinity next to a pole of chi0, and
    the sign of Im lambda times the slope of Re lambda tells which. chi0 is
    evaluated at w - i delta (the 1/(de - w + i delta) of chitk/chiAB.py
    and chitk/pairchi.py), so its poles and the zeros of a causal kernel
    lie at Im w = +delta and above. Near a zero, lambda goes as
    c (w - w*), and at the crossing Im lambda * dRe lambda/dw = -Im(w*)
    abs(c)^2 < 0 for any complex c. Near a pole, lambda goes as
    A/(w* - w), and the same product is +Im(w*) abs(A)^2/D^2 > 0. So a
    crossing with a positive product is a pole of lambda and is dropped.
    The test needs no threshold and does not depend on whether the grid
    resolves delta: on the U=3, V1=0.5 honeycomb Neel state at nk=6 the
    pair basis lost its crossings at 1.0024 (Gamma) and at 1.2430 and
    1.3369 (q=(0.05,0.05)) with delta=1e-3 on a 400-point grid, and the
    matching ones on a 1600-point grid with delta=1e-2, and kept every TDHF
    magnon in both cases, the damped one at 1.28 included.

    A third kind of crossing is a smooth sign change of Re lambda with
    abs(lambda) flat and of order one, the eigenvalue passing far from
    zero. The linear extrapolation then puts the zero on either side, so
    the test drops some of these and keeps others, and the kept ones come
    back with a large abs(gamma), which is the measure to judge them by.
    Measured on the spinful chain with V1=0.6 at q=1/2 (plasmon_bands,
    nk=2000, delta=2e-2): Re lambda changes sign at 0.596 while abs(lambda)
    stays between 0.594 and 0.603 from 0.47 to 0.71, so that crossing,
    returned before as a pole with abs(gamma)=0.60, is dropped."""
    poles = []
    n = len(es)
    for ib in range(eigs.shape[1]): # loop over eigenvalue branches
        re,im = eigs[:,ib].real,eigs[:,ib].imag
        for k in range(n): # scan the grid, including the last point
            if re[k]==0.0: # on the grid, the slope from the neighbors
                slope = re[min(k+1,n-1)] - re[max(k-1,0)]
                if im[k]*slope>0.: continue # a pole of lambda
                poles.append((es[k],im[k]))
            elif k+1<n and re[k]*re[k+1]<0.0: # a sign change
                t = -re[k]/(re[k+1]-re[k]) # linear interpolation factor
                g = im[k] + t*(im[k+1]-im[k])
                if g*(re[k+1]-re[k])>0.: continue # a pole of lambda
                poles.append((es[k] + t*(es[k+1]-es[k]),g))
    if len(poles)==0: return np.zeros((0,2))
    poles.sort(key=lambda p: p[0]) # sort by frequency
    return np.array(poles)


def _scan_from_zero(kwargs):
    """Add w=0 to a frequency grid given as energies= that starts above
    zero. The Goldstone mode of an antiferromagnet at Q=0 is a double zero
    of the kernel at w=0, which the broadening opens into a simple zero at
    w=delta, so a grid that starts at or above delta would miss it."""
    es = kwargs.get("energies",None)
    if es is None: return kwargs
    es = np.array(es,dtype=np.float64)
    if np.min(es)>0.: es = np.concatenate([[0.],es])
    return dict(kwargs,energies=es)


def _kernels_from_chis(chis,V):
    """RPA kernel matrices 1 - V*chi(omega), one per entry of chis (a
    matrix per frequency). Shared by _poles_from_chi_matrix (which only
    reports the interpolated zero-crossings of these kernels) and
    rpa_kernel_ops (which returns the raw kernels themselves, e.g. to
    inspect np.linalg.eigvals directly at a single frequency)."""
    iden = np.identity(chis[0].shape[0],dtype=np.complex128) # identity
    return [iden - V@chi for chi in chis] # RPA kernel, 1 - U*chi


def _poles_from_chi_matrix(es,chis,V):
    """Given the non-interacting response chi(omega) (a matrix per
    frequency) and an interaction matrix V, locate the poles of the RPA
    kernel 1 - V*chi(omega): every frequency where an eigenvalue of the
    kernel crosses zero (a collective mode / RPA-Stoner instability).
    Returns an (npoles,2) array: [frequency, residual imaginary part] --
    the residual imaginary part is signed (interpolated directly from the
    kernel eigenvalue, which can lie on either side of the real axis), so
    judge how sharp/well-defined a mode is by its *magnitude*, not its raw
    value; do not filter with e.g. `gamma < tol` -- use `abs(gamma) < tol`.
    A sign change of the real part where the eigenvalue goes through
    infinity, next to a pole of chi, is a bare electron-hole pair and not
    a mode, and is not returned, see _zero_crossings."""
    if V is None: raise ValueError("V (interaction matrix) is required "
                                    "to locate the poles of the RPA kernel")
    kernels = _kernels_from_chis(chis,V)
    raw_eigs = np.array([np.linalg.eigvals(k) for k in kernels]) # (nw,N)
    eigs = _track_eigenvalue_branches(raw_eigs) # continuous branches
    return _zero_crossings(es,eigs) # zeros only, not the crossings at infinity


def rpa_kernel_poles(h,V=None,q=None,**kwargs):
    """Return the poles of the generic RPA kernel 1 - V(q)*chi(q,omega), i.e.
    the frequencies at which chi_RPA = chi@(1-V*chi)^-1 diverges (collective
    modes/instabilities). A and B are forwarded through kwargs exactly
    as in chi_AB_RPA (defaulting to the charge channel and q=0 if not
    given). V can be a plain matrix (q-independent, onsite-only) or a
    real-space hopping dict/MultiHopping with support beyond (0,0,0) --
    see interaction_at_q for how the latter is turned into V(q) at this
    same q. Returns an (npoles,2) array: [frequency, residual imaginary
    part], one row per collective mode found, sorted by frequency -- see
    _poles_from_chi_matrix's docstring for how to interpret the residual
    imaginary part's sign. An energies= grid that starts above zero gets
    w=0 added, see _scan_from_zero."""
    from ..chi import chiAB # get response function
    es,chis = chiAB(h,mode="matrix",q=q,**_scan_from_zero(kwargs)) # non-interacting response
    Vq = interaction_at_q(V,h,q) if V is not None else None
    return _poles_from_chi_matrix(es,chis,Vq)


def rpa_kernel_poles_ops(h,ops=None,V=None,pAs=None,pBs=None,q=None,**kwargs):
    """Same as rpa_kernel_poles, but for the tensor response of a list of
    local operators (e.g. Sx,Sy,Sz), as used by chi_ops_RPA. pAs/pBs can be
    passed in (see build_ops_projectors) to reuse a q-independent operator
    tensor across many calls at different q, instead of rebuilding it from
    ops every time. As in rpa_kernel_poles, w=0 is added to a grid that
    starts above zero, so the Goldstone mode is found at about delta."""
    es,chis = _chi_ops_matrix_vectorized(h,ops=ops,pAs=pAs,pBs=pBs,q=q,
                                         **_scan_from_zero(kwargs))
    Vq = interaction_at_q(V,h,q) if V is not None else None
    return _poles_from_chi_matrix(es,chis,Vq)


def rpa_kernel_ops(h,ops=None,V=None,pAs=None,pBs=None,q=None,**kwargs):
    """Same non-interacting response and interaction-matrix construction
    as rpa_kernel_poles_ops, but returns the raw RPA kernel matrices
    1 - V(q)*chi0(q,omega) themselves (one per frequency in `energies`),
    rather than only the interpolated pole locations
    _poles_from_chi_matrix extracts from them. Useful e.g. to inspect
    np.linalg.eigvals(kernels[i]) directly at a single frequency, instead
    of scanning a frequency grid for zero crossings."""
    es,chis = _chi_ops_matrix_vectorized(h,ops=ops,pAs=pAs,pBs=pBs,q=q,**kwargs)
    Vq = interaction_at_q(V,h,q) if V is not None else None
    if Vq is None: raise ValueError("V (interaction matrix) is required "
                                     "to build the RPA kernel")
    return es,_kernels_from_chis(chis,Vq)


