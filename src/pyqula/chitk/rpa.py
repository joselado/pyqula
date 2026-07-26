import numpy as np
from scipy.optimize import linear_sum_assignment
from .. import algebra

# compute general RPA response function
def chi_AB_RPA(h,V=None,**kwargs):
    """Compute the RPA chi for a hamiltonian"""
    from ..chi import chiAB # get response function
    es,chis = chiAB(h,mode="matrix",**kwargs) # non-interacting response
    iden = np.identity(chis[0].shape[0],dtype=np.complex128) # identity
    if V is not None: # finite interaction, RPA summation
        chis_rpa = [chi@algebra.inv(iden - V@chi) for chi in chis]
    else: chis_rpa = chis
    return es,np.array(chis_rpa)


mode_rpa = "vectorized"

def _chi_ops_matrix_vectorized(h,ops=None,**kwargs):
    """Compute the non-interacting response tensor for a list of local
    operators (e.g. Sx,Sy,Sz), evaluated on every lattice site. Shared by
    chi_ops_RPA and rpa_kernel_poles_ops, so both consume the exact same
    operator/projector tensor and stay consistent with each other."""
    from ..chi import chiAB # get response function
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
    return chiAB(h,mode="matrix",pAs=pAs,pBs=pBs,**kwargs) # non-interacting response

def chi_ops_RPA(h,ops=None,V=None,**kwargs):
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
                es,chisi = chiAB(h,mode="matrix",A=A,B=B,
                                **kwargs) # non-interacting response
                chis[i][j] = chisi # store in the list
        # now make it a block matrix, and reshpae accordingly
        chis_tmp = np.array(chis) # convert to array
        chis = [] # empty list
        for i in range(len(es)): # loop over energies
            chi = chis_tmp[:,:,i,:,:] # get this one
            chi = [[chi[i,j,:,:] for i in range(nop)] for j in range(nop)]
            chis.append(np.bmat(chi)) # store
    elif mode_rpa=="vectorized": # all at once
        es,chis = _chi_ops_matrix_vectorized(h,ops=ops,**kwargs)
    else: raise
    iden = np.identity(chis[0].shape[0],dtype=np.complex128) # identity
    if V is not None: # finite interaction, RPA summation
        chis_rpa = [chi@algebra.inv(iden - V@chi) for chi in chis]
    else: chis_rpa = chis
    return es,np.array(chis_rpa)




def chi_AB_RPA_scf(scf):
    """Return the RPA response function for an SCF object"""
    if len(scf.v)==1: # just the onsite term
        return chi_AB_RPA(scf.hamiltonian,scf.v[(0,0,0)])
    else: raise # not implemented


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


def _poles_from_chi_matrix(es,chis,V):
    """Given the non-interacting response chi(omega) (a matrix per
    frequency) and an interaction matrix V, locate the poles of the RPA
    kernel 1 - V*chi(omega): every frequency where an eigenvalue of the
    kernel crosses zero (a collective mode / RPA-Stoner instability).
    Returns an (npoles,2) array: [frequency, residual imaginary part]."""
    if V is None: raise ValueError("V (interaction matrix) is required "
                                    "to locate the poles of the RPA kernel")
    iden = np.identity(chis[0].shape[0],dtype=np.complex128) # identity
    kernels = [iden - V@chi for chi in chis] # RPA kernel, 1 - U*chi
    raw_eigs = np.array([np.linalg.eigvals(k) for k in kernels]) # (nw,N)
    eigs = _track_eigenvalue_branches(raw_eigs) # continuous branches
    poles = [] # storage for the poles found
    for ib in range(eigs.shape[1]): # loop over eigenvalue branches
        re = eigs[:,ib].real
        im = eigs[:,ib].imag
        for k in range(len(es)-1): # scan the frequency grid
            if re[k]==0.0: # exactly on the grid (rare)
                poles.append((es[k],im[k]))
            elif re[k]*re[k+1]<0.0: # sign change -> a zero crossing
                t = -re[k]/(re[k+1]-re[k]) # linear interpolation factor
                w0 = es[k] + t*(es[k+1]-es[k])
                g0 = im[k] + t*(im[k+1]-im[k])
                poles.append((w0,g0))
    if len(poles)==0: return np.zeros((0,2))
    poles.sort(key=lambda p: p[0]) # sort by frequency
    return np.array(poles)


def rpa_kernel_poles(h,V=None,**kwargs):
    """Return the poles of the generic RPA kernel 1 - V*chi(q,omega), i.e.
    the frequencies at which chi_RPA = chi@(1-V*chi)^-1 diverges (collective
    modes/instabilities). A, B and q are forwarded through kwargs exactly
    as in chi_AB_RPA (defaulting to the charge channel and q=0 if not
    given). Returns an (npoles,2) array: [frequency, residual imaginary
    part], one row per collective mode found, sorted by frequency."""
    from ..chi import chiAB # get response function
    es,chis = chiAB(h,mode="matrix",**kwargs) # non-interacting response
    return _poles_from_chi_matrix(es,chis,V)


def rpa_kernel_poles_ops(h,ops=None,V=None,**kwargs):
    """Same as rpa_kernel_poles, but for the tensor response of a list of
    local operators (e.g. Sx,Sy,Sz), as used by chi_ops_RPA."""
    es,chis = _chi_ops_matrix_vectorized(h,ops=ops,**kwargs)
    return _poles_from_chi_matrix(es,chis,V)


def spinchi_pm_RPA(h,U=0.,v=[0.,0.,1.],**kwargs):
    """Compute the spin RPA response for a hamiltonian.
     - v is the chosen quantization axis of the ladder operators
     - U is the Hubbard interaction"""
     # v needs to be implemented
    sx = h.get_operator("sx") # spin operator, eigen +-1
    sy = h.get_operator("sy") # spin operator, eigen +-1
    sz = h.get_operator("sz") # spin operator, eigen +-1
    v = np.array(v) # convert to array
    sp = (sx + 1j*sy)/2. # ladder operator
    sm = (sx - 1j*sy)/2. # ladder operator
    from ..chi import chiAB # get response function
    es,chis = chiAB(h,A=sp,B=sm,mode="matrix",**kwargs) # non-interacting response functions
    iden = np.identity(chis[0].shape[0],dtype=np.complex128) # identity
    # NOTE (2026-07-21): this U/2 prefactor looks wrong, not "should be here".
    # spinchi_ladder (chitk/spinchi.py), which computes the same S+/S- RPA
    # channel and is the one actually reachable from Hamiltonian.get_spinchi_ladder,
    # uses a bare U (no 1/2) here. Cross-checked spinchi_ladder's bare-U prefactor
    # against an exact 2-site Hubbard dimer diagonalization (staggered spin
    # susceptibility grows correctly with U, matching the exact result at U=0
    # and its trend for small U) and it is correct; this function would predict
    # the RPA/Stoner-like pole at 2x the correct U. Left unfixed because this
    # function is never called anywhere in the codebase (dead code) -- fix here
    # too if it is ever wired up.
    chisrpa = [chi@algebra.inv(iden + U/2.*chi) for chi in chis] # RPA summation
    return es,np.array(chisrpa) # return energies and RPA response function




