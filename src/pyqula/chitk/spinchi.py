import numpy as np
from .rpa import chi_AB_RPA
from .rpa import chi_ops_RPA

# spin-response functions


def _require_onsite_only_V(H):
    """Raise ValueError unless H.V is a plain onsite (Hubbard-like)
    interaction -- a single (0,0,0) key, or a plain matrix (the only form
    that predates non-onsite support and is always implicitly onsite).

    Automatically dressing the spin (Sx,Sy,Sz)/(S+,S-) RPA channel from
    H.V beyond that is not properly verified yet: bond exchange (J) alone
    and density-density (V) alone are individually well-behaved
    mathematically (V2K_matrix is exact for both, see its docstring, and
    rotational symmetry has been checked numerically for each separately),
    but a VJinteraction Hamiltonian with BOTH set simultaneously only ever
    exposes the combined z-channel SCF matrix as H.V (never the separately
    built x/y-channel ones -- see scftk.spinspin's
    "scf.hamiltonian.V = vz"), so the spin RPA vertex built from it cannot
    be trusted in general, and no independent (e.g. exact-diagonalization)
    cross-check exists yet for any non-onsite case. Rather than silently
    returning a result that ranges from "probably fine" (isotropic J
    alone) to "structurally wrong" (anisotropic J + V), every non-onsite
    H.V is rejected here until that validation exists.

    If you understand the caveats and want to proceed anyway (e.g. for
    regression-testing the underlying vertex math), build the interaction
    matrix yourself and call chitk.rpa.chi_ops_RPA/rpa_kernel_poles_ops
    directly instead of going through H.V -- the same way
    chitk.densitychi's charge-channel functions (which never read H.V)
    already do."""
    from ..multihopping import MultiHopping
    V = H.V
    if V is None: return # no interaction at all, nothing to check
    if isinstance(V,MultiHopping): V = V.get_dict()
    if not isinstance(V,dict): return # plain matrix: always onsite
    if set(V.keys()) != {(0,0,0)}:
        raise ValueError(
            "get_magnon_bands/get_rpa_kernel_poles/get_spinchi_full/"
            "get_spinchi_ladder only support a plain onsite (Hubbard-like) "
            "H.V (a single (0,0,0) key); this Hamiltonian's H.V has "
            f"non-onsite support ({sorted(V.keys())}). Non-onsite spin-"
            "channel RPA is not yet properly verified -- see "
            "chitk.spinchi._require_onsite_only_V's docstring. Build the "
            "interaction yourself and call chitk.rpa.chi_ops_RPA/"
            "rpa_kernel_poles_ops directly if you want to proceed anyway.")


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
        _require_onsite_only_V(H) # raises for non-onsite H.V, see docstring
        U = H.V # get the interaction
        if U is not None: # finite interaction
            # up to here U is a real-space hopping dict (just the onsite
            # (0,0,0) key, enforced above). V2K_matrix/the linearity of
            # the map are per-direction, so transforming every key
            # independently here and only Fourier-summing afterwards
            # (done downstream by chi_AB_RPA's interaction_at_q, at
            # whatever q the caller asks for) is exactly equivalent to
            # transforming a single q-summed matrix.
            U = {d: V2K_matrix(m) for d,m in U.items()}
    else: U = None # no RPA
    return chi_AB_RPA(H,A=sp,B=sm,V=U,**kwargs) # RPA interacting response



def V2U_matrix(V):
    """Transform the V interaction into the U matrix needed for RPA.
    V is a 2N matrix (spin-orbital basis, up/down doubled per orbital);
    returns the N-dimensional matrix of up-down + down-up cross terms
    between every pair of orbitals (i,j), not just i==j -- needed once V
    can carry off-diagonal structure, e.g. a bond/neighbor-shell
    interaction connecting different sites of a multi-orbital unit cell.
    Reduces to the previous diagonal-only result whenever V only
    populates i==j entries (e.g. a plain onsite Hubbard U), so this is a
    strict generalization, not a behavior change, for every case that
    worked before.

    NOTE: only reads the up-down/down-up cross terms, which is a complete
    description of V only when V has no same-spin (up-up/down-down)
    component -- true for a plain onsite Hubbard U, but NOT true for a
    direct S_i.S_j bond term (scftk.spinspin._build_v), whose
    matrix carries an equally-weighted same-spin component that this
    function silently ignores. Kept around only because
    tests/chi/test_v2u_matrix_offdiagonal.py pins its (intentionally
    partial) behavior; RPA vertex construction should use V2K_matrix
    below instead -- see its docstring for why."""
    N = V.shape[0]//2 # dimension
    U = np.zeros((N,N),dtype=np.complex128) # initialize
    for i in range(N): # loop over orbitals
        for j in range(N): # loop over orbitals
            U[i,j] = V[2*i,2*j+1] + V[2*i+1,2*j] # up-down + down-up cross term
    return U # return the matrix


def V2K_matrix(V):
    """Transform the V interaction into the K matrix needed for the spin
    RPA vertex: K_ij is the coefficient of Sz_i Sz_j when the spin-orbital
    density-density interaction V (2N matrix, up/down doubled per
    orbital, same convention as V2U_matrix/H.V) is expanded in the
    {n_i n_j, n_i Sz_j, Sz_i n_j, Sz_i Sz_j} operator basis for each pair
    of orbitals (i,j).

    This generalizes V2U_matrix, which is only a complete description of V
    when it has no same-spin (up-up/down-down) component (true for plain
    onsite Hubbard U, where n_up n_down = n^2/4 - Sz^2 has no such
    component -- there V2K_matrix reduces to exactly -V2U_matrix, see
    below). A direct S_i.S_j bond term
    (scftk.spinspin._build_v's +/-1/4 sign pattern) is instead
    PURELY a same-spin-plus-cross-spin combination with no free n_i n_j or
    n_i Sz_j piece, and V2U_matrix's cross-term-only read discards exactly
    half of it -- this was the root cause of a real (not just
    approximation-quality) gap in the Goldstone mode for bond exchange (J1)
    seen in magnon_bands/get_rpa_kernel_poles: the RPA vertex it fed in was
    off by a factor of 2 for any interaction with a genuine same-spin
    component.

    Derivation: writing the 2x2 (up/down) block for orbitals (i,j) as
    M = [[V[2i,2j],V[2i,2j+1]],[V[2i+1,2j],V[2i+1,2j+1]]], and
    Sz_i Sz_j = 1/4 (n_iu n_ju - n_iu n_jd - n_id n_ju + n_id n_jd), the
    coefficient of Sz_i Sz_j alone is K_ij = M[0,0]-M[0,1]-M[1,0]+M[1,1]
    (project out the n_i n_j / n_i Sz_j / Sz_i n_j components, which are
    symmetric combinations that this alternating-sign contraction kills).
    For onsite Hubbard (M[0,0]=M[1,1]=0, M[0,1]=M[1,0]=U/2), this gives
    K_ii = -U = -V2U_matrix(V)[i,i] exactly -- so using V2K_matrix in
    _full_spin_U/spinchi_ladder in place of -V2U_matrix/V2U_matrix is a
    strict generalization, identical to the previous (already validated)
    result for a plain onsite interaction, and only changes behavior when
    V carries a genuine same-spin component."""
    N = V.shape[0]//2 # dimension
    K = np.zeros((N,N),dtype=np.complex128) # initialize
    for i in range(N): # loop over orbitals
        for j in range(N): # loop over orbitals
            K[i,j] = (V[2*i,2*j] - V[2*i,2*j+1]
                      - V[2*i+1,2*j] + V[2*i+1,2*j+1]) # coefficient of Sz_i Sz_j
    return K # return the matrix


def replicateU(U,n=3):
    """Take an interaction matrix U and replicate 3 times for different
    channels"""
    out = [[U*0. for i in range(n)] for j in range(n)]
    for i in range(n): out[i][i] = U
    return np.block(out) # return the full matrix



def _full_spin_U(H):
    """Return the interaction for the full (Sx,Sy,Sz) spin channel, in the
    convention used by spinchi_full/chi_ops_RPA (replicated across the 3
    spin channels with the +2 prefactor -- see V2K_matrix/replicateU).

    Returned as a real-space hopping dict with just the onsite (0,0,0)
    key: raises ValueError (via _require_onsite_only_V, see its docstring
    for why) if H.V carries any non-onsite (neighbor-shell) support --
    e.g. a VJinteraction run with V1/J1 density-density or exchange
    couplings. Identical to the old, pre-non-onsite-generalization result
    for that allowed onsite case (see V2K_matrix's docstring for why: it
    reduces to -V2U_matrix there, so +2*V2K_matrix reduces to exactly the
    previous -2*V2U_matrix).

    Returns None if the Hamiltonian carries no mean-field interaction."""
    _require_onsite_only_V(H) # raises for non-onsite H.V, see docstring
    U = H.V # get the interaction
    if U is None: return None # no interaction, no RPA dressing
    return {d: 2*replicateU(V2K_matrix(m),n=3) for d,m in U.items()}


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
    the output of get_mean_field_hamiltonian, and H.V must be a plain
    onsite (Hubbard-like) interaction -- a single (0,0,0) key. Raises
    ValueError for any non-onsite H.V (bond exchange, density-density, or
    a combination of the two, e.g. from VJinteraction) -- see
    _require_onsite_only_V's docstring for why that support is not yet
    properly verified.

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




