import numpy as np
from .. import gpu
from .rpa import chi_AB_RPA
from .rpa import chi_ops_RPA

# spin-response functions


def _interaction_matrices(H):
    """Every real-space interaction matrix H carries, h.V and the recorded
    exchange channels alike, as a list of {(n1,n2,n3): matrix} dicts."""
    from ..bsetk.interaction import V2dict
    out = []
    if getattr(H,"V",None) is not None: out.append(V2dict(H.V))
    ch = getattr(H,"Vchannels",None)
    if ch is not None:
        for name in ("x","y","z","d"):
            if ch.get(name,None) is not None: out.append(V2dict(ch[name]))
    return out


def _is_site_local(H,tol=1e-12):
    """True if every interaction H carries couples each site only to
    itself: nonzero only in the same-site 2x2 spin blocks of the (0,0,0)
    matrix. That is the only case the site-basis vertex can carry; it is
    exact there for the transverse response, and the longitudinal one
    misses only the coupling of Sz to charge fluctuations, which a
    spin-only vertex has no channel for (2e-5 on a half-filled Neel
    Hubbard chain). A coupling between two different sites, in the same
    cell or not, density-density or exchange, has a Fock rung on the
    electron-hole PAIR index that a site-separable vertex has no place
    for, and the lattice-vector keys alone do not say whether one is
    there: an SCF stores every neighbor-shell key even at zero coupling,
    and a 0D island keeps all of its bonds under (0,0,0)."""
    for V in _interaction_matrices(H):
        for d,m in V.items():
            m = np.array(m)
            if tuple(d)!=(0,0,0):
                if np.max(np.abs(m))>tol: return False
                continue
            off = m.copy()
            for i in range(m.shape[0]//2): off[2*i:2*i+2,2*i:2*i+2] = 0.
            if np.max(np.abs(off))>tol: return False
    return True


def _use_pair_basis(H):
    """True if the RPA spin response of H has to be summed in the pair
    basis (chitk.pairchi) rather than with the site-basis vertex: H
    carries an interaction and it is not site local, see _is_site_local.

    Measured on the J1=3 Neel honeycomb, where both routes run, the site
    vertex puts the acoustic magnon at 1.3296 (q=0.1) and 2.4488 (q=0.2)
    against 1.3687 and 2.5074 from the pair basis and from time-dependent
    Hartree-Fock, which a brute-force reference confirms; the difference
    is exactly the Fock rung of the exchange bonds. For a V1-ordered
    ferromagnet the site vertex is identically zero and gives no magnon
    at all. So the public entry points take this route instead of
    returning either number."""
    return len(_interaction_matrices(H))>0 and not _is_site_local(H)


_PAIR_ROUTE_KWARGS = ("energies","delta","nk","T","q","imode","ij_mode",
                      "chi_prec")


def _pair_route_kwargs(H,kwargs):
    """Translate the keyword arguments of the site-basis response into the
    pair-basis ones, keeping the site-basis defaults (chitk.chiAB.chiAB_q:
    energies from -3 to 3, delta=0.1, nk=60, a temperature equal to delta)
    so that the same call means the same response on either route."""
    unknown = sorted(set(kwargs) - set(_PAIR_ROUTE_KWARGS))
    if len(unknown)>0:
        raise ValueError("this Hamiltonian's interaction couples different "
            "sites, so its RPA spin response is summed in the pair basis "
            "(chitk.pairchi), which does not take %s; the accepted "
            "keyword arguments are %s"%(unknown,list(_PAIR_ROUTE_KWARGS)))
    if gpu.get_gpu():
        raise NotImplementedError("the GPU backend is not implemented for "
            "an interaction that couples different sites: its RPA spin "
            "response is summed in the pair basis (chitk.pairchi), which "
            "has no device backend; call pyqula.gpu.set_gpu(False)")
    if kwargs.get("chi_prec",None) not in (None,"double"): # unset is double here
        raise NotImplementedError("chi_prec=%r is not implemented for an "
            "interaction that couples different sites: its RPA spin "
            "response is summed in the pair basis (chitk.pairchi), which "
            "is double precision; leave chi_prec unset or use "
            "'double'"%(kwargs["chi_prec"],))
    if kwargs.get("imode","mesh")!="mesh":
        raise NotImplementedError("imode=%r is not implemented for an "
            "interaction that couples different sites, whose RPA spin "
            "response is summed in the pair basis (chitk.pairchi) on a "
            "k-mesh; use imode='mesh'"%(kwargs["imode"],))
    if H.has_eh:
        raise NotImplementedError("the RPA spin response of a Nambu "
            "Hamiltonian whose interaction couples different sites is not "
            "implemented: the site-basis vertex drops the Fock rung of "
            "those bonds, and the pair basis (chitk.pairchi) does not "
            "handle the Nambu basis")
    delta = kwargs.get("delta",0.1)
    T = kwargs.get("T",None)
    return dict(energies=kwargs.get("energies",np.linspace(-3.0,3.0,100)),
                delta=delta,nk=kwargs.get("nk",60),
                T=delta if T is None else T)


def _pair_route_response(H,ops,opsB,**kwargs):
    """The RPA response of the local spin operators ops/opsB (lists of 2x2
    matrices) summed in the pair basis, in the layout of the site-basis
    response: (energies, one (nop*N)x(nop*N) matrix per energy).

    q=None follows chitk.chiAB.chiAB: the response averaged over the q
    points of the k-mesh, i.e. the local one. Here each q is dressed on
    its own before the average, which is the RPA of the local response
    (the site basis averages the bare response and dresses it at q=0)."""
    from .pairchi import pair_chi_rpa
    kw = _pair_route_kwargs(H,kwargs)
    q = kwargs.get("q",None)
    if q is not None:
        return pair_chi_rpa(H,q=q,ops=ops,opsB=opsB,**kw)
    qs = H.geometry.get_kmesh(nk=kw["nk"])
    chis = [pair_chi_rpa(H,q=qi,ops=ops,opsB=opsB,**kw)[1] for qi in qs]
    return np.array(kw["energies"],dtype=np.float64),np.mean(chis,axis=0)


_SPIN_OPS = [np.array([[0.,1.],[1.,0.]],dtype=np.complex128)/2.,
             np.array([[0.,-1j],[1j,0.]],dtype=np.complex128)/2.,
             np.array([[1.,0.],[0.,-1.]],dtype=np.complex128)/2.]


def _require_onsite_only_V(H):
    """Raise ValueError unless the site-basis (Sx,Sy,Sz)/(S+,S-) vertex can
    be built from what this Hamiltonian carries.

    The public entry points (spinchi_full, spinchi_ladder, magnon_bands)
    never reach this for an interaction between different sites: they
    send it to the pair basis first (_use_pair_basis). It is the guard of
    the vertex builders _full_spin_U/_transverse_spin_K themselves, which
    tests and internal callers use directly on the vertex math.

    Three cases pass. A plain onsite (Hubbard-like) H.V -- a single
    (0,0,0) key, or a plain matrix. A Hamiltonian carrying H.Vchannels
    whose density-density part is onsite, for which the vertex is built
    per channel by _channel_spin_U; for a neighbor-shell exchange that
    vertex is the site-separable part only, missing the Fock rung of the
    bonds. And no interaction at all.

    What is rejected: a non-onsite density-density interaction, whose
    contribution to the spin response is entirely a Fock rung on the
    electron-hole pair index (V2K_matrix maps a spin-independent V_ij to
    exactly zero), and a non-onsite H.V with no H.Vchannels beside it,
    for which nothing says which interaction it came from: an isotropic
    J1 and an anisotropic J1z leave exactly the same z-channel matrix in
    H.V."""
    from ..multihopping import MultiHopping
    V = H.V
    if V is None: return # no interaction at all, nothing to check
    if _channel_spin_U(H) is not None:
        # the SCF recorded the three exchange channels separately, so the
        # vertex is built per channel rather than guessed from this single
        # matrix -- a neighbor-shell EXCHANGE interaction is fine here.
        # _channel_spin_U itself returns None (and so falls through to the
        # check below) when the density-density part is non-onsite, which
        # is the case it cannot represent
        return
    if isinstance(V,MultiHopping): V = V.get_dict()
    if not isinstance(V,dict): return # plain matrix: always onsite
    if set(V.keys()) != {(0,0,0)}:
        raise ValueError(
            "get_magnon_bands/get_rpa_kernel_poles/get_spinchi_full/"
            "get_spinchi_ladder need an interaction their site-basis spin "
            "vertex can represent: an onsite (Hubbard-like) H.V, or a "
            "neighbor-shell EXCHANGE one whose three spin channels the SCF "
            "recorded in H.Vchannels. This Hamiltonian's H.V has non-onsite "
            f"support ({sorted(V.keys())}) and neither applies -- see "
            "chitk.spinchi._require_onsite_only_V's docstring. If the "
            "non-onsite part is a DENSITY-DENSITY interaction, no "
            "site-separable vertex can carry its Fock rung at all; use a "
            "route that keeps the pair index and so can, both with an "
            "exact Goldstone mode: h.get_magnon_bands(method='pair') or "
            "method='tdhf'. If this is "
            "a hand-built H.V, build the vertex yourself and call "
            "chitk.rpa.chi_ops_RPA/rpa_kernel_poles_ops directly.")


def spinchi_ladder(H,v=[0.,0.,1.],RPA=True,**kwargs):
    """Return the spin response function"""
    if H.has_eh:
        raise NotImplementedError("the ladder spin response is not "
                "implemented in the Nambu basis")
    sx = H.get_operator("sx") # spin operator, eigen +-1
    sy = H.get_operator("sy") # spin operator, eigen +-1
    sz = H.get_operator("sz") # spin operator, eigen +-1
    v = np.array(v) # convert to array
    # this is not finished yet
    sp = (sx + 1j*sy)/2. # ladder operator
    sm = (sx - 1j*sy)/2. # ladder operator
    if RPA and _use_pair_basis(H): # an interaction between sites
        spin_p = np.array([[0.,1.],[0.,0.]],dtype=np.complex128)
        return _pair_route_response(H,[spin_p],[spin_p.T],**kwargs)
    if RPA: # RPA mode
        U = _transverse_spin_K(H) # per-channel vertex, if the SCF kept one
        if U is not None:
            return chi_AB_RPA(H,A=sp,B=sm,V=U,**kwargs)
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


def blockU(Us):
    """Take one interaction matrix per spin channel and build the
    block-diagonal matrix chi_ops_RPA's (Sx,Sy,Sz) tensor expects.

    replicateU below is the special case Us = [U,U,U]. The general one is
    needed as soon as the three channels differ, i.e. for an anisotropic
    exchange interaction, where using a single matrix for all three is not
    an approximation but the wrong vertex."""
    n = len(Us)
    out = [[Us[0]*0. for i in range(n)] for j in range(n)]
    for i in range(n): out[i][i] = Us[i]
    return np.block(out)


def replicateU(U,n=3):
    """Take an interaction matrix U and replicate 3 times for different
    channels"""
    out = [[U*0. for i in range(n)] for j in range(n)]
    for i in range(n): out[i][i] = U
    return np.block(out) # return the full matrix



def _density_part_is_onsite(ch,tol=1e-12):
    """True if the density-density part recorded in H.Vchannels has no
    support beyond the onsite (0,0,0) term (or is absent/zero)."""
    vd = ch.get("d",None)
    if vd is None: return True
    for d,m in vd.items():
        if tuple(d)==(0,0,0): continue
        if np.max(np.abs(np.array(m)))>tol: return False
    return True


def _transverse_spin_K(H,tol=1e-9):
    """Return the transverse (S+/S-) vertex of H as a real-space dict, or
    None if this Hamiltonian does not carry separate spin channels.

    The ladder response chi_{+-} lives in the transverse channel, so its
    vertex is the x (equivalently y) exchange coupling plus the
    density-density part, not the z one. They coincide for everything that
    worked before this existed -- an onsite Hubbard U, or an isotropic
    exchange -- and differ for an anisotropic one, where S+/S- is not an
    eigen-channel of the interaction at all: Kx != Ky is refused rather
    than silently averaged."""
    ch = getattr(H,"Vchannels",None)
    if ch is None or not _density_part_is_onsite(ch): return None
    keys = set()
    for k in ["x","y","z","d"]:
        if ch.get(k,None) is not None: keys |= set(ch[k].keys())
    if len(keys)==0: return None
    shape = None
    for k in ["x","y","z","d"]:
        if ch.get(k,None) is not None:
            shape = list(ch[k].values())[0].shape ; break
    def at(name,d):
        m = ch.get(name,None)
        if m is None or d not in m: return np.zeros(shape,dtype=np.complex128)
        return np.array(m[d])
    out = {}
    for d in keys:
        Kx,Ky = V2K_matrix(at("x",d)),V2K_matrix(at("y",d))
        if np.max(np.abs(Kx-Ky))>tol:
            raise ValueError("this mean field has an anisotropic in-plane "
                "exchange (Kx != Ky), so S+/S- is not an eigen-channel of "
                "its interaction and the ladder response has no single "
                "vertex. Use get_spinchi_full, which treats the three spin "
                "channels separately")
        out[d] = Kx + V2K_matrix(at("d",d))
    return out


def _channel_spin_U(H):
    """Build the (Sx,Sy,Sz) RPA vertex from H.Vchannels, the three exchange
    channel matrices the SCF decoupled plus its density-density part, or
    return None if this Hamiltonian does not carry them.

    This is what makes a neighbor-shell EXCHANGE interaction usable in the
    spin RPA. The mean field for one is already the right thing --
    scftk.spinspin's SCF decouples the x and y channels too, by rotating
    the density matrix into the frame where that axis is the computational
    z (see _run_anisotropic_scf) -- so the only thing missing was a vertex
    that matches it, and H.V alone cannot provide one: an isotropic J1 and
    an anisotropic J1z leave exactly the same z-channel matrix there.
    With the channels kept separately the vertex is simply built per
    channel, and isotropic and anisotropic exchange are equally correct
    (and equally distinguishable).

    The density-density part enters every channel identically -- it is
    spin-rotation invariant -- so it is added to each of the three. For an
    isotropic interaction this reproduces _full_spin_U's replicated vertex
    exactly, which is why turning it on changes no existing result."""
    ch = getattr(H,"Vchannels",None)
    if ch is None: return None
    if not _density_part_is_onsite(ch):
        # a neighbor-shell density-density interaction is a different
        # problem from a neighbor-shell exchange one: its contribution to
        # the spin response is a Fock rung on the electron-hole PAIR
        # index, which no site-separable vertex can carry, and V2K_matrix
        # maps it to exactly zero. Whether that matters depends on the
        # converged state rather than on the interaction (it cancels on a
        # Neel state and is fatal on a V1-ordered ferromagnet), so this
        # falls back to the gate rather than deciding for the caller
        return None
    keys = set()
    for k in ["x","y","z","d"]:
        if ch.get(k,None) is not None: keys |= set(ch[k].keys())
    if len(keys)==0: return None
    def at(name,d,shape): # the channel's matrix at this lattice vector
        m = ch.get(name,None)
        if m is None or d not in m: return np.zeros(shape,dtype=np.complex128)
        return np.array(m[d])
    shape = None
    for k in ["x","y","z","d"]:
        if ch.get(k,None) is not None:
            shape = list(ch[k].values())[0].shape ; break
    out = {}
    for d in keys:
        Kd = V2K_matrix(at("d",d,shape)) # isotropic, common to all channels
        Ks = [V2K_matrix(at(a,d,shape)) + Kd for a in ["x","y","z"]]
        out[d] = 2*blockU(Ks)
    return out


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

    Returns None if the Hamiltonian carries no mean-field interaction.

    A Hamiltonian that carries H.Vchannels (one from the exchange SCF)
    takes the _channel_spin_U route above instead, which builds the vertex
    per spin channel rather than replicating one -- see that function for
    why that is what lets a neighbor-shell exchange interaction through."""
    out = _channel_spin_U(H) # per-channel vertex, if the SCF recorded one
    if out is not None: return out
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
    if RPA and _use_pair_basis(H): # an interaction between sites
        return _pair_route_response(H,_SPIN_OPS,_SPIN_OPS,**kwargs)
    Ss = _full_spin_operators(H)
    U = _full_spin_U(H) if RPA else None
    return chi_ops_RPA(H,ops=Ss,V=U,**kwargs) # non-interacting response


def _map_over_q(f,qpath,**kwargs):
    """Map f over a q-path, serially when the device backend is in use.

    parallel.pcall forks several worker processes; with a single GPU that
    is contention rather than parallelism (each process would build its own
    device context and its own copy of the operator tensors), which is the
    failure mode documentation/gpu_porting_plan.md item 4 suspects behind
    classicalspin.py's unconditional CPU forcing. So under
    pyqula.gpu.set_gpu(True) the q-loop stays in this process."""
    from .. import parallel
    if gpu.get_gpu(): # one device, one process
        return [f(q) for q in qpath]
    return parallel.pcall(f,qpath) # CPU path, as before


def magnon_bands(H,qpath=None,nq=20,**kwargs):
    """Return the magnon bands: the poles of the full spin RPA kernel
    (the same Sx,Sy,Sz channel used by spinchi_full/get_iets_ldos), scanned
    along a q-path. This is the collective-mode dispersion of the spin
    response -- where chi_spin_RPA diverges -- rather than the response
    itself.

    Requires a Hamiltonian with a mean-field interaction set (H.V), e.g.
    the output of get_mean_field_hamiltonian. An interaction that couples
    each site only to itself (a Hubbard U) uses the site-basis vertex,
    which is exact there for the transverse response. One that couples
    different sites, density-density
    or exchange, is scanned with the pair-basis ladder instead
    (chitk.pairchi.magnon_bands_pair, see _use_pair_basis), with the
    energies/delta/nk/T defaults of this function.

    Returns (qs,ws,gammas): qs is the integer index of the q-point along
    the path (the same convention used by get_bands for the k-axis),
    ws the pole frequency and gammas its residual imaginary part -- signed,
    so judge how sharp/well-defined a mode is by abs(gammas), not gammas
    directly (see rpa.py's _poles_from_chi_matrix docstring). Different
    q-points can have different numbers of poles, so all three are
    returned as flat 1D arrays -- ready for a scatter-style dispersion
    plot -- rather than a ragged per-q array."""
    from .rpa import rpa_kernel_poles_ops, build_ops_projectors
    if _use_pair_basis(H): # an interaction between sites
        from .pairchi import magnon_bands_pair
        if "q" in kwargs:
            raise ValueError("magnon_bands scans a q-path, pass qpath or "
                             "nq rather than q")
        return magnon_bands_pair(H,qpath=qpath,nq=nq,
                                 **_pair_route_kwargs(H,kwargs))
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
    outs = _map_over_q(f,qpath,**kwargs) # compute the poles at every q
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
    out = _map_over_q(f,qpath,**kwargs) # compute all
    qout = [] # empty list
    chimap = [] # storage
    for o in out: # loop over qvectors
        es,chis = o[0],o[1]
        cs = [np.trace(c).imag for c in chis]
        chimap.append(cs) # store
    for q in qpath: # loop over qvectors
        qout.append([q for c in chis])
    return np.array(qout),energies,np.array(chimap) # return everythin




