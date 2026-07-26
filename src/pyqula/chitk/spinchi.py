import numpy as np
from .rpa import chi_AB_RPA
from .rpa import chi_ops_RPA

# spin-response functions


def spinchi_ladder(H,v=[0.,0.,1.],RPA=True,**kwargs):
    """Return the spin response function"""
    if H.has_eh: 
        print("Not implemented with Nambu basis")
        raise
    sx = H.get_operator("sx") # spin operator, eigen +-1
    sy = H.get_operator("sy") # spin operator, eigen +-1
    sz = H.get_operator("sz") # spin operator, eigen +-1
    v = np.array(v) # convert to array
    # this is not finished yet
    sp = (sx + 1j*sy)/2. # ladder operator
    sm = (sx - 1j*sy)/2. # ladder operator
    if RPA: # RPA mode
        U = H.V # get the interaction
        if U is not None: # finite interaction
            if len(U)>1: raise # not implemented for momentum dependent
            U = U[(0,0,0)] # onsite interaction matrix
            # up to here U is an off-diagonal matrix, with
            # finite elements between up and down
            # now let us pick up the up-down elements and sum them
            U = V2U_matrix(U) # transform the U matrix (2N) into the (N)
            U = -U # beware of this minus sign for spin response (!!!)
    else: U = None # no RPA
    return chi_AB_RPA(H,A=sp,B=sm,V=U,**kwargs) # RPA interacting response



def V2U_matrix(V):
    """Transform the V interaction into the U matrix needed for RPA"""
    # V is a 2N matrix with individual U values in the diagonal
    N = V.shape[0]//2 # dimension
    U = np.zeros((N,N),dtype=np.complex128) # initialize
    for i in range(N): # loop over orbitals
        U[i,i] = V[2*i,2*i+1] # add contribution
        U[i,i] += V[2*i+1,2*i] # add contribution
    return U # return the matrix

def replicateU(U,n=3):
    """Take an interaction matrix U and replicate 3 times for different
    channels"""
    out = [[U*0. for i in range(n)] for j in range(n)]
    for i in range(n): out[i][i] = U
    from .. import algebra
    return np.bmat(out) # return the full matrix



def _full_spin_U(H):
    """Return the interaction matrix for the full (Sx,Sy,Sz) spin channel,
    in the convention used by spinchi_full/chi_ops_RPA (onsite Hubbard U
    only, replicated across the 3 spin channels with the -2 prefactor).
    Returns None if the Hamiltonian carries no mean-field interaction."""
    U = H.V # get the interaction
    if U is None: return None # no interaction, no RPA dressing
    if len(U)>1: raise # not implemented for momentum dependent U
    U = U[(0,0,0)] # onsite interaction matrix
    U = V2U_matrix(U) # transform the U matrix (2N) into the (N)
    U = replicateU(U,n=3) # replicate for all the channels
    U = -2*U # beware of this minus sign for spin response (!!!)
    return U


def _full_spin_operators(H):
    """Return the (Sx,Sy,Sz)/2 operators used by spinchi_full/magnon_bands,
    projected onto the electron subspace for Nambu Hamiltonians."""
    sx = H.get_operator("sx") # spin operator, eigen +-1
    sy = H.get_operator("sy") # spin operator, eigen +-1
    sz = H.get_operator("sz") # spin operator, eigen +-1
    # this is technically not correct, as it will ignore e-h components
    # of the response. Nevertheless, it can be good enough as starting
    # point
    if H.has_eh: # for Nambu basis, quick workaround
        el = H.get_operator("electron")
        sx = sx@el
        sy = sy@el
        sz = sz@el
    return [sx/2.,sy/2.,sz/2.] # pauli matrices, with eigen +-1/2


def spinchi_full(H,RPA=True,**kwargs):
    """Return the spin response function"""
    Ss = _full_spin_operators(H)
    U = _full_spin_U(H) if RPA else None
    return chi_ops_RPA(H,ops=Ss,V=U,**kwargs) # non-interacting response


def magnon_bands(H,qpath=None,nq=20,**kwargs):
    """Return the magnon bands: the poles of the full spin RPA kernel
    (the same Sx,Sy,Sz channel used by spinchi_full/get_iets_ldos), scanned
    along a q-path. This is the collective-mode dispersion of the spin
    response -- where chi_spin_RPA diverges -- rather than the response
    itself.

    Requires a Hamiltonian with a mean-field interaction set (H.V), e.g.
    the output of get_mean_field_hamiltonian.

    Returns (qs,ws,gammas): qs is the integer index of the q-point along
    the path (the same convention used by get_bands for the k-axis),
    ws the pole frequency and gammas its residual imaginary part -- signed,
    so judge how sharp/well-defined a mode is by abs(gammas), not gammas
    directly (see rpa.py's _poles_from_chi_matrix docstring). Different
    q-points can have different numbers of poles, so all three are
    returned as flat 1D arrays -- ready for a scatter-style dispersion
    plot -- rather than a ragged per-q array."""
    from .rpa import rpa_kernel_poles_ops, build_ops_projectors
    from .. import parallel
    Ss = _full_spin_operators(H)
    U = _full_spin_U(H)
    if U is None: raise ValueError("Hamiltonian has no mean-field "
            "interaction (H.V); set one first, e.g. via "
            "get_mean_field_hamiltonian")
    # the operator/projector tensor is q-independent: build it once and
    # reuse it at every q-point instead of rebuilding it from Ss each time
    pAs,pBs = build_ops_projectors(H,Ss)
    qpath = H.geometry.get_kpath(qpath,nk=nq) # generate the q-path
    def f(q):
        return rpa_kernel_poles_ops(H,V=U,pAs=pAs,pBs=pBs,q=q,**kwargs)
    outs = parallel.pcall(f,qpath) # compute the poles at every q
    qs,ws,gammas = [],[],[] # flat storage
    for iq,poles in enumerate(outs): # loop over q-points
        for (w,g) in poles: # loop over poles found at this q
            qs.append(iq)
            ws.append(w)
            gammas.append(g)
    return np.array(qs),np.array(ws),np.array(gammas)



def get_iets_ldos(H,nk=1,delta=1e-2,e=0.,**kwargs):
    """Return the IETS local density of state by computing the full
    spin response function"""
    from ..checkclass import is_iterable
    if is_iterable(e): energies = np.array(e) # assume it is an array 
    else: energies = np.array([e]) # list with energies
    es,chis = H.get_spinchi_full(nk=nk,
                                 energies=energies,delta=delta,
                                 imode="mesh",**kwargs)
    # chi is a 3Nx3N tensor, resum the relevant elements
    r = H.geometry.r # positions
    n = len(r) # number of sites
    dout = [] # list
    for chi in chis:
        chi = chi.imag # take the imaginary part of the chi
        d = [np.sum([chi[n*j+i,n*j+i] for j in range(3)]) for i in range(n)]
        dout.append(d) # store
    dout = np.array(dout) # as array
    if len(energies)==1: # just one requested
        return r,dout[0] # return positions and IETS ldos
    else:
        return r,dout # return positions and IETS ldos




def get_qdos_iets(H,energies=np.linspace(0.,1.,100),
                  qpath=None,nq=20,
                  nk=10,delta=1e-2,**kwargs):
    """Return the momentum-resolved spin respose function"""
    def f(q):
        return H.get_spinchi_full(q=q,nk=nk,energies=energies,
                                delta=delta,**kwargs)
    #out = parallel.pcall_deep(f,qs,cores=1) # compute all
    qpath = H.geometry.get_kpath(qpath,nk=nq) # generate kpath
#    out = [f(q) for q in qpath] # compute all
    from .. import parallel
    out = parallel.pcall(f,qpath) # compute all
    qout = [] # empty list
    chimap = [] # storage
    for o in out: # loop over qvectors
        es,chis = o[0],o[1]
        cs = [np.trace(c).imag for c in chis]
        chimap.append(cs) # store
    for q in qpath: # loop over qvectors
        qout.append([q for c in chis])
    return np.array(qout),energies,np.array(chimap) # return everythin




