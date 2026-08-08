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
    matrices -- consistent with a caller that never mentions q at all
    only ever having meant the onsite/zero-momentum interaction."""
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


# compute general RPA response function
def chi_AB_RPA(h,V=None,q=None,**kwargs):
    """Compute the RPA chi for a hamiltonian"""
    from ..chi import chiAB # get response function
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
    function"""
    from ..chi import chiAB # get response function
    nop = len(ops) # number of operators
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
    else: raise
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
    value; do not filter with e.g. `gamma < tol` -- use `abs(gamma) < tol`."""
    if V is None: raise ValueError("V (interaction matrix) is required "
                                    "to locate the poles of the RPA kernel")
    kernels = _kernels_from_chis(chis,V)
    raw_eigs = np.array([np.linalg.eigvals(k) for k in kernels]) # (nw,N)
    eigs = _track_eigenvalue_branches(raw_eigs) # continuous branches
    poles = [] # storage for the poles found
    nw = len(es)
    for ib in range(eigs.shape[1]): # loop over eigenvalue branches
        re = eigs[:,ib].real
        im = eigs[:,ib].imag
        for k in range(nw): # scan the frequency grid, including the last point
            if re[k]==0.0: # exactly on the grid (rare)
                poles.append((es[k],im[k]))
            elif k+1<nw and re[k]*re[k+1]<0.0: # sign change -> a zero crossing
                t = -re[k]/(re[k+1]-re[k]) # linear interpolation factor
                w0 = es[k] + t*(es[k+1]-es[k])
                g0 = im[k] + t*(im[k+1]-im[k])
                poles.append((w0,g0))
    if len(poles)==0: return np.zeros((0,2))
    poles.sort(key=lambda p: p[0]) # sort by frequency
    return np.array(poles)


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
    imaginary part's sign."""
    from ..chi import chiAB # get response function
    es,chis = chiAB(h,mode="matrix",q=q,**kwargs) # non-interacting response
    Vq = interaction_at_q(V,h,q) if V is not None else None
    return _poles_from_chi_matrix(es,chis,Vq)


def rpa_kernel_poles_ops(h,ops=None,V=None,pAs=None,pBs=None,q=None,**kwargs):
    """Same as rpa_kernel_poles, but for the tensor response of a list of
    local operators (e.g. Sx,Sy,Sz), as used by chi_ops_RPA. pAs/pBs can be
    passed in (see build_ops_projectors) to reuse a q-independent operator
    tensor across many calls at different q, instead of rebuilding it from
    ops every time."""
    es,chis = _chi_ops_matrix_vectorized(h,ops=ops,pAs=pAs,pBs=pBs,q=q,**kwargs)
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


