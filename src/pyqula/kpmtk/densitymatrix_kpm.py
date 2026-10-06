# KPM (Chebyshev/sparse) density matrix, restricted to the elements
# actually required by a density-density interaction.
#
# The existing SCF machinery (scftk/densitydensity.py) always
# computes a *dense* n x n density-matrix block for every lattice vector
# appearing in the interaction dictionary "v" (see get_dm/full_dm there),
# using exact diagonalization on a k-mesh. Here we instead:
#   1) sample the same k-mesh the exact-diagonalization path would use
#      (h.geometry.get_kmesh), and at each k build the small Bloch
#      Hamiltonian H(k) -- still sparse/no bigger than the unit cell --
#      then get each needed <i|P_occ(H(k))|j> occupied-projector element
#      from Chebyshev moments instead of diagonalizing H(k), every pair and
#      every k-point in one block recursion (kpmtk/pairmomentsjax.py), and
#   2) only evaluate the (i, j) pairs that "v" actually has nonzero
#      couplings for, instead of a dense block; the same per-k values are
#      reused across every direction that needs them (see
#      _dm_kpm_from_needed), with the per-direction dependence entering
#      only through the Bloch phase applied during the k-sum -- matching
#      the exact-diagonalization path's own phase convention
#      (dmtk/fulldm.py's exp(2*pi*i*k.d)).
#
# BdG/Nambu Hamiltonians (h.has_eh) need a separate "which elements"
# function (required_elements_eh) instead of just required_elements,
# because the extra electron-hole doubling is stored in a different index
# convention than v's -- see required_elements_eh's docstring -- but reuse
# the exact same per-k Bloch KPM engine (_dm_kpm_from_needed) once the
# needed (direction,row,col) entries are known.
import numpy as np
from numba import jit
from scipy.sparse import csr_matrix
from scipy.special import expit

from .. import kpm
from .. import parallel
from .bandwidth import estimate_bandwidth
from .momenttoprofile import generate_profile
from .kernels import jackson_kernel

# Shared defaults for the KPM SCF's tuning knobs. scftk/
# densitydensity_kpm.py's generic_densitydensity_kpm/densitydensity_kpm
# reference these same constants (rather than separately hardcoding their
# own copies) so the density-matrix computation and the Fermi-energy
# search can never silently drift apart on what "unspecified" means.
DEFAULT_NK = 8
DEFAULT_NPOL = 200


def required_elements(v, tol=1e-10):
    """Given the interaction dictionary v (lattice vector -> matrix),
    return the set of (direction, i, j) density-matrix entries actually
    read by scftk/densitydensity.py for every nonzero v[d][i,j]:
      - normal_term_ij (via get_mf_normal) reads dm[d2][j,i] (d2=-d, and
        indices SWAPPED relative to v's own (i,j)) -- so that transposed
        entry is requested at direction d2, not the raw (d,i,j) location;
      - get_dc_energy instead reads dm[d][i,j] directly, un-transposed, at
        v's own (d,i,j) location.
    Both are required (they are different matrix entries in general), so
    each nonzero v[d][i,j] contributes both. This does not need v to
    contain both +d and -d as a symmetry assumption: processing direction
    d alone already yields the exact entries both consumers read for that
    (d,i,j) pair, regardless of whether v happens to be Hermitian.
    Also adds the onsite occupations dm[(0,0,0)][i,i]/[j,j] that the
    Hartree term needs."""
    needed = set()
    for d, m in v.items():
        m = np.asarray(m)
        rows, cols = np.nonzero(np.abs(m) > tol)
        d = tuple(d)
        d2 = tuple(-x for x in d)
        for i, j in zip(rows, cols):
            i, j = int(i), int(j)
            needed.add((d, i, j))    # raw: get_dc_energy's dm[d][i,j]
            needed.add((d2, j, i))   # transposed at -d: get_mf_normal's dm[d2][j,i]
            needed.add(((0, 0, 0), i, i))
            needed.add(((0, 0, 0), j, j))
    return needed


def _local_nambu_index(orb, sector):
    """Map a v-space (spin-doubled, electron-sector-only) orbital index
    into its position inside the per-site interleaved Nambu unit cell that
    h.intra actually uses when h.has_eh (sctk/reorder.py's
    block2nambu_matrix: each site's 4 Nambu slots are, in order,
    [up-electron, down-electron, down-hole, up-hole]). "sector" is "e" for
    the electron partner of orb, or "h" for its hole partner.

    A Nambu Hamiltonian is always spinful in pyqula, so there are always
    four slots per site."""
    site, spin = orb//2, orb % 2
    if sector == "e": return 4*site + spin
    elif sector == "h": return 4*site + 2 + spin
    else: raise ValueError(sector)


def required_anomalous_elements(v, tol=1e-10):
    """Pairing (anomalous) density-matrix entries the BdG mean field needs,
    in the same "block" index convention v itself uses (electron indices
    0..N-1, hole-sector-local indices 0..N-1) -- see
    scftk/superscf.py's anomalous_term_ij_jit, which for a given
    (spinless-site i, spinless-site j) pair reads:
        out[2i,2j]     = v[2i,2j+1]  * dm[2j,2i]
        out[2i,2j+1]   = v[2i,2j]    * dm[2j+1,2i]
        out[2i+1,2j+1] = v[2i+1,2j]  * dm[2j+1,2i+1]
        out[2i+1,2j]   = v[2i+1,2j+1]* dm[2j,2i+1]
    Relabelling each case by the (a,b) index pair of the v[..] factor that
    gates it, every case reduces to the same rule: dm[b^1,a] is read
    whenever v[d][a,b] is nonzero (b^1 flips the spin index at fixed site
    -- the up/down partner needed by the pairing channel), and (per
    get_mf_anomalous) this dm is read at direction d2=-d, not d."""
    needed = set()
    for d, m in v.items():
        m = np.asarray(m)
        rows, cols = np.nonzero(np.abs(m) > tol)
        d2 = tuple(-x for x in d)
        for a, b in zip(rows, cols):
            a, b = int(a), int(b)
            needed.add((d2, b ^ 1, a))
    return needed


def required_elements_eh(v, tol=1e-10):
    """Alternative to required_elements for BdG/Nambu Hamiltonians
    (h.has_eh): returns the (direction, row, col) entries actually read
    out of dm, in dm's native per-site-interleaved Nambu-local indexing
    (matching h.intra's own layout), instead of the whole dense (2n)x(2n)
    block per direction.

    get_mf's has_eh branch (scftk/densitydensity.py) extracts
    two sub-blocks out of each dm[key] via superconductivity.get_eh_sector
    (which internally reorders dm[key] with sctk/reorder.py's
    nambu2block): the electron-electron block dme[key] = dm[key]'s "ee"
    corner, fed into the *same* get_mf_normal used for non-SC Hamiltonians
    -- so it needs exactly required_elements(v)'s (d,i,j) triples, just
    with i and j each remapped from v's electron-sector index space into
    their Nambu-local position (_local_nambu_index(.,"e")); and the
    electron-hole ("anomalous"/pairing) block dma10[key], read at
    required_anomalous_elements(v)'s (d,p,q) triples with p remapped via
    _local_nambu_index(.,"e") and q via _local_nambu_index(.,"h").
    get_dc_energy (same file) additionally reads dm[(0,0,0)][i,i] and
    dm[d][i,j] directly, un-reordered, at exactly required_elements(v)'s
    own raw (d,i,j) positions -- so those are needed a second time, at
    their *un-mapped* location."""
    ee = required_elements(v, tol=tol)
    anomalous = required_anomalous_elements(v, tol=tol)
    needed = set()
    for d, i, j in ee:
        needed.add((d, _local_nambu_index(i, "e"), _local_nambu_index(j, "e")))
        needed.add((d, i, j))  # raw, un-mapped: what get_dc_energy reads
    for d, p, q in anomalous:
        needed.add((d, _local_nambu_index(p, "e"), _local_nambu_index(q, "h")))
    return needed


@jit(nopython=True,cache=True)
def _chebyshev_basis(xs,n_moments):
    """T[n,:] = T_n(xs) for n=0..n_moments-1, via the standard 3-term
    Chebyshev recursion T_0=1, T_1=x, T_{n+1}=2x*T_n-T_{n-1}. Building this
    array once (shared by every (row,col,k) triple -- xs/n_moments never
    change within one _dm_kpm_from_needed call) instead of implicitly
    recomputing it inside a per-pair kpm.dm_ij_energy/generate_profile call
    is the difference between one O(n_moments*ne) recursion total and one
    per pair -- see _dm_kpm_from_needed's docstring for the measured
    effect."""
    ne = len(xs)
    T = np.zeros((n_moments,ne))
    T[0,:] = 1.0
    if n_moments>1: T[1,:] = xs
    for n in range(2,n_moments):
        T[n,:] = 2.*xs*T[n-1,:]-T[n-2,:]
    return T


def _estimate_kpm_scale(hk_gen,ks):
    """Shared KPM energy-rescaling estimate -- 1.1x the largest per-k
    Gershgorin bandwidth bound (kpmtk.bandwidth.estimate_bandwidth) over
    the sampled k-mesh -- used by both _dm_kpm_from_needed and
    get_fermi4filling_kpm whenever scale=None. Factored out so a caller
    that needs both on the SAME Hamiltonian (e.g. scftk.spinspin
    ._run_anisotropic_scf's integration="kpm" branch, which calls
    get_fermi4filling_kpm then _dm_kpm_from_needed every SCF iteration) can
    estimate it once and pass the same value to both, instead of each
    independently re-sweeping the whole k-mesh through estimate_bandwidth
    for an identical result."""
    return 1.1*max(estimate_bandwidth(hk_gen(k)) for k in ks)


def _check_scale_covers_spectrum(mus, scale, given, kpm_prec="double"):
    """Raise if Chebyshev moments of H(k)/scale show part of its spectrum
    outside [-1,1]. For a Hermitian matrix with its spectrum inside that
    interval every moment is bounded, |<a|T_n(H)|b>| <= 1 for unit vectors
    a and b, while T_n grows exponentially outside it, so a moment past one
    (or a NaN/inf) means that the scale is too small. This is exact rather
    than a comparison with the Gershgorin bound _estimate_kpm_scale uses,
    which is only an upper bound on the spectral radius and would refuse a
    valid scale between the two; the tolerance only absorbs the roundoff a
    spectrum edge sitting exactly at +-1 picks up along the recursion.
    given says whether the scale came from the caller or from
    _estimate_kpm_scale, which can only fail on a non-Hermitian H(k).
    kpm_prec is the precision the moments were computed in: a single
    precision recursion drifts by about n*eps, so a state near zero energy,
    where |T_2n(0)|=1, takes a moment past one by more than the double
    precision tolerance and would be refused with a valid scale."""
    from .scaleguard import moments_within_bound
    if moments_within_bound(mus, kpm_prec=kpm_prec): return
    if given:
        raise ValueError("the KPM scale=%g does not cover the spectrum of "
                "H(k): the Chebyshev expansion needs every eigenvalue inside "
                "[-scale,scale], and in a mean-field loop that is the "
                "spectrum after the Fermi shift, not the bare band "
                "structure. Leave scale=None to have it estimated on the "
                "Hamiltonian actually expanded, or pass a larger one"
                % scale)
    raise ValueError("the Chebyshev moments of H(k) diverge although the KPM "
            "scale=%g was estimated from its Gershgorin bound, which only "
            "happens when H(k) is not Hermitian; check that the Hamiltonian "
            "and any mean-field guess passed in are Hermitian" % scale)


def resolve_kpm_prec(kpm_prec):
    """The precision of the Chebyshev recursion of the KPM mean field:
    None picks single precision on the device, where the card's float32
    makes it the fast option and it agrees with double to about 1e-8 in
    the density matrix (future_development/gpu_kpm_mean_field.md), and
    double precision on the CPU. Both backends take either one explicitly."""
    from .pairmomentsjax import get_precision_names
    if kpm_prec is None:
        from .. import gpu
        return "single" if gpu.get_gpu() else "double"
    if kpm_prec not in get_precision_names():
        raise ValueError("kpm_prec must be one of %s (or None, single on "
                "the GPU and double on the CPU), got %r"
                % (", ".join(repr(x) for x in get_precision_names()), kpm_prec))
    return kpm_prec


def _dm_kpm_from_needed(h, needed, nk=DEFAULT_NK, scale=None,
                         npol=DEFAULT_NPOL, ne=None, cores=None, T=0.0,
                         kpm_prec=None):
    """Shared Bloch-KPM engine: given the (direction, row, col)
    density-matrix entries to compute (in whatever index convention the
    caller's "needed" set already uses -- see required_elements/
    required_elements_eh), sample the same k-mesh the exact-diagonalization
    path uses (h.geometry.get_kmesh(nk=nk)), and get each needed
    <i|P_occ(H(k))|j> occupied-projector element from the Chebyshev
    moments of the Bloch Hamiltonian H(k) instead of diagonalizing it. A
    given (i,j) pair is computed once per k (not once per direction): every
    direction that needs it reuses the same per-k value, weighted by the
    Bloch phase exp(2*pi*i*k.d) and summed over k, exactly mirroring the
    exact-diagonalization path's own phase convention (dmtk/fulldm.py).

    T is the same finite-temperature smearing scftk/
    densitydensity.py's ED path applies via Fermi-Dirac occupation
    (densitymatrix.py's full_dm(h,T=...)): rather than a hard cutoff at
    the Fermi energy (E=0), the occupied-window integration is weighted by
    the Fermi function at temperature T, and the window is extended a bit
    above 0 so that weight isn't dropped. T=0 (the default) is treated the
    same tiny regularization (1e-15) full_dm itself uses, recovering an
    effectively-hard cutoff.

    The moments of every pair and every k-point come from one block
    recursion (kpmtk/pairmomentsjax.py) on the jax device that the
    package-wide switch selects, the CPU or the GPU: the distinct starting
    columns e_j form one dense block, and each step reads every pair it
    needs with one gather. kpm_prec is the precision of that recursion (see
    resolve_kpm_prec). This replaced one numba recursion per pair and per
    k, which also never reached the device: the per-pair loop ran the same
    recursion two or more times per site, and on a 1728-orbital island
    took 27 s per evaluation against 1.6 s for the block in double
    precision and 0.46 s in single on a consumer card, and about 5 s and
    2.5 s on the six-core CPU of the same machine
    (future_development/gpu_kpm_mean_field.md). Since every k-point goes in
    one call, cores no longer splits the k-mesh here; it still sets the
    package-wide core count, which the Fermi search uses.

    A density-matrix entry is linear in its moments, so the energy
    integral of the Jackson-damped Chebyshev series against the Fermi
    weights is done once, on the basis, and each pair's value is the
    contraction of its moments with the resulting coefficients."""
    kpm_prec = resolve_kpm_prec(kpm_prec)
    if ne is None: ne = npol*4
    norb = h.intra.shape[0]
    ks = [list(k) for k in h.geometry.get_kmesh(nk=nk)]
    hk_gen = h.get_hk_gen()

    needed = sorted(needed)
    ds = sorted({d for (d, i, j) in needed})
    # H(k) is Hermitian, and so is any function of it, so the (j,i) value
    # at each k is the conjugate of the (i,j) one. Computing both from
    # their own moments lets roundoff open an anti-Hermitian part in the
    # density matrix, which the mean-field loop amplifies by a fixed factor
    # every iteration until the recursion diverges. Only i<=j is computed
    # and i>j is set by conjugation below, which makes dm[-d][j,i] =
    # conj(dm[d][i,j]) hold by construction and halves the moment work.
    # This is Hermiticity of the whole matrix, so it holds just as well for
    # an electron-hole entry of a Nambu H(k), whose partner is the
    # hole-electron entry, not another pairing one.
    pairs = sorted({(min(i, j), max(i, j)) for (_, i, j) in needed})
    pair_index = {p: idx for idx, p in enumerate(pairs)}
    diagonal = np.array([i == j for (i, j) in pairs], dtype=bool)

    given = scale is not None # see _check_scale_covers_spectrum
    if scale is None:
        # one global scale for every k, so the occupied-energy window
        # used below means the same thing at every k-point
        scale = _estimate_kpm_scale(hk_gen, ks)
    if scale <= 0:
        raise ValueError("H(k) has zero bandwidth on the sampled k-mesh "
                "(it vanishes at every k) -- cannot set a KPM energy "
                "scale; check that this Hamiltonian actually has "
                "hopping/onsite terms in this sector")
    Tsafe = abs(T) if T != 0. else 1e-15
    upper = min(0.99*scale, 30.*Tsafe)
    xin = np.linspace(-0.99*scale, upper, ne)
    weights = expit(-xin/Tsafe)  # Fermi-Dirac occupation at temperature T

    # 2*npol moments, the length kpmnumba's own moment routines return
    # for n=npol, which the tests and the rest of the module assume
    n_moments = 2*npol
    xs_reduced = xin/scale
    Tbasis = _chebyshev_basis(xs_reduced, n_moments)  # (n_moments, ne)
    jack_w = jackson_kernel(np.ones(n_moments))  # depends only on n_moments
    coef = np.ones(n_moments); coef[1:] = 2.0  # mu_0 has coefficient 1, mu_{n>=1} has 2
    denom = np.sqrt(1.-xs_reduced**2)*scale
    # basis[n,:], dotted into a pair's real (or imaginary) moments and
    # summed over n, reproduces exactly what
    # generate_profile(mus,xs,kernel="jackson")/scale*np.pi used to (the pi
    # from generate_profile's own normalization and dm_ij_energy's external
    # *np.pi cancel, leaving the plain /scale here)
    basis = (coef*jack_w)[:, None] * Tbasis / denom[None, :]  # (n_moments, ne)
    # the entry of a pair is conj(sum_n cint[n]*mu_n) for real cint: the
    # profile of its moments, weighted by the occupation and integrated
    cint = np.trapezoid(basis*weights[None, :], x=xin, axis=1)/np.pi

    if cores is not None: parallel.set_cores(cores)
    from .pairmomentsjax import pair_values
    ms = [csr_matrix(hk_gen(k))/scale for k in ks]
    # the pair (i,j) holds <e_i|T_n(H(k))|e_j>: the recursion starts from
    # e_j and is read on row i, which is the element dm[i,j] (see
    # densitymatrix.py's restricted_dm for the convention)
    vals, mumax = pair_values(ms, pairs, cint, kpm_prec=kpm_prec)
    _check_scale_covers_spectrum([mumax], scale, given, kpm_prec=kpm_prec)
    vals = vals.conj()
    vals[:, diagonal] = vals[:, diagonal].real # a diagonal entry is its own conjugate

    needed_by_d = dict()
    for d, i, j in needed: needed_by_d.setdefault(d, []).append((i, j))

    dm = {d: np.zeros((norb, norb), dtype=np.complex128) for d in ds}
    fac = 1./len(ks)
    for d in ds:
        phases = np.array([np.exp(2j*np.pi*np.dot(k, d)) for k in ks])
        for (i, j) in needed_by_d.get(d, []):
            col = vals[:, pair_index[(min(i, j), max(i, j))]]
            if i > j: col = col.conj() # see the pairs above
            dm[d][i, j] = fac*np.sum(phases*col)
    return dm


def get_dm_kpm(h, v, nk=DEFAULT_NK, scale=None, npol=DEFAULT_NPOL, ne=None,
               cores=None, T=0.0, kpm_prec=None, **kwargs):
    """KPM-based analogue of scftk.densitydensity.get_dm: return
    a dictionary {direction: matrix} with the density matrix, but computing
    only the entries that v actually requires, each one through a sparse
    Chebyshev-moment (KPM) correlator instead of full diagonalization.
    Meant for sparse/large Hamiltonians where exact diagonalization of a
    dense k-mesh becomes the bottleneck.

    For BdG/Nambu Hamiltonians (h.has_eh) the required entries are
    determined by required_elements_eh instead of required_elements (see
    its docstring) -- both are then handed to the same per-k Bloch-KPM
    engine, _dm_kpm_from_needed. kpm_prec is the precision of its
    Chebyshev recursion, see resolve_kpm_prec."""
    ds = [(0, 0, 0)] + [d for d in v if d != (0, 0, 0)]
    if getattr(h, "has_eh", False):
        needed = required_elements_eh(v)
    else:
        needed = required_elements(v)
    dm = _dm_kpm_from_needed(h, needed, nk=nk, scale=scale, npol=npol,
                              ne=ne, cores=cores, T=T, kpm_prec=kpm_prec)
    # every direction v has a key for must be present in the output, even
    # if it happened to contribute no required entries of its own
    for d in ds:
        if d not in dm:
            dm[d] = np.zeros((h.intra.shape[0], h.intra.shape[0]),
                              dtype=np.complex128)
    return dm


def _cumulative_trapz(y, x):
    """cumulative_trapz(y,x)[k] = trapezoidal integral of y from x[0] to
    x[k] (a small local helper so this module doesn't depend on scipy's
    cumulative_trapezoid, whose name/availability has moved across scipy
    versions)."""
    dx = np.diff(x)
    avg = (y[1:]+y[:-1])/2.
    return np.concatenate([[0.], np.cumsum(avg*dx)])


def _kpm_dos_moments(h, nk, scale, npol, ne, cores, kpm_prec=None):
    """Shared per-k Bloch-KPM engine for get_fermi4filling_kpm and
    get_total_energy_kpm: samples the k-mesh, gets the Chebyshev moments of
    the local density of states averaged over every orbital in the cell at
    each k via kpm.full_trace -- a deterministic sum over all sites/
    orbitals (looping i=0..norb-1), not a stochastic random-vector estimate
    -- then k-averages those moments into a single total-DOS-per-orbital
    profile (valid because moments are linear in the density of states, so
    the k-average of the moments equals the moments of the k-averaged
    DOS), and reconstructs it on the standard reduced-energy grid via the
    Jackson kernel. Returns (scale, xs, ys): xs the reduced-energy grid
    ([-0.99,0.99]), ys the (real-valued) reconstructed DOS profile on it --
    ready for either a cumulative-integral inversion (Fermi search) or an
    energy integral (total energy) downstream, so the two functions that
    use this can never silently disagree about what "the DOS" means.

    The trace is one recursion per orbital. On the CPU it is numba's
    batched kernel (kpm.full_trace, one vector per thread, with the moment
    doubling that halves the recursion), and on the device the block
    kernel of the density matrix (pairmomentsjax.trace_moments) with every
    k-point in one call, since the batched jax kernel behind full_trace is
    a BCOO scatter-add that was no faster than numba on a consumer card.
    kpm_prec is the precision of either, see resolve_kpm_prec."""
    if ne is None: ne = npol*4
    ks = [list(k) for k in h.geometry.get_kmesh(nk=nk)]
    hk_gen = h.get_hk_gen()
    given = scale is not None # see _check_scale_covers_spectrum
    if scale is None:
        scale = _estimate_kpm_scale(hk_gen, ks)
    if scale <= 0:
        raise ValueError("H(k) has zero bandwidth on the sampled k-mesh "
                "(it vanishes at every k) -- cannot set a KPM energy "
                "scale; check that this Hamiltonian actually has "
                "hopping/onsite terms in this sector")

    kpm_prec = resolve_kpm_prec(kpm_prec)
    from .. import gpu
    if gpu.get_gpu():
        from .pairmomentsjax import trace_moments
        ms = [csr_matrix(hk_gen(k))/scale for k in ks]
        musk, mumax = trace_moments(ms, 2*npol, kpm_prec=kpm_prec)
        _check_scale_covers_spectrum([mumax], scale, given, kpm_prec=kpm_prec)
        mus = np.mean(musk, axis=0)  # k-average of the moments
    else:
        def moments_for_k(k):
            Hk = csr_matrix(hk_gen(k))
            mus = kpm.full_trace(Hk/scale, n=npol, kpm_prec=kpm_prec) # an average of bounded moments
            _check_scale_covers_spectrum(mus, scale, given, kpm_prec=kpm_prec)
            return mus
        if cores is not None: parallel.set_cores(cores)
        results = parallel.pcall(moments_for_k, ks)
        mus = sum(results)/len(results)  # k-average of the moments

    xs = np.linspace(-1.0, 1.0, ne, endpoint=True)*0.99  # reduced energies
    ys = generate_profile(mus, xs, kernel="jackson").real
    return scale, xs, ys


def get_fermi4filling_kpm(h, filling, nk=DEFAULT_NK, scale=None,
        npol=DEFAULT_NPOL, ne=None, cores=None, T=0., kpm_prec=None):
    """KPM analogue of spectrum.get_fermi4filling: find the Fermi energy
    for a given filling without ever diagonalizing anything, so the KPM
    SCF (scftk/densitydensity_kpm.py) stays fully
    diagonalization-free end to end -- otherwise it would still need
    spectrum.get_fermi4filling's own per-k diagonalization just to locate
    the Fermi level, even though the density matrix itself is computed via
    KPM.

    Gets the k-averaged, Jackson-kernel-reconstructed density-of-states
    profile from _kpm_dos_moments (see its docstring). Its cumulative
    integral gives the fraction of orbitals occupied as a function of
    energy (0 at the sampled window's bottom, 1 at its top); inverting it
    at the target filling gives the Fermi energy directly, with no
    diagonalization anywhere. The cumulative integral is forced monotonic
    (np.maximum.accumulate) before inversion: a finite-npol Jackson-kernel
    KPM reconstruction is not guaranteed nonnegative everywhere (Gibbs-type
    ringing near band edges/gaps/van Hove singularities), which without
    this would make the inversion via np.interp silently ill-defined.

    For BdG/Nambu Hamiltonians (h.has_eh), mirrors spectrum.
    get_fermi4filling's own workaround (an approximation, per that
    function's comment): the Fermi energy is estimated from the
    electron-only spectrum, obtained here by projecting out the Nambu
    doubling via h.remove_nambu() before proceeding -- not by locating a
    zero-energy quasiparticle level, since there generally isn't a well
    defined "filling" of a superconductor's own BdG spectrum."""
    if h.has_eh:
        h0 = h.copy()
        h0.remove_nambu()
        return get_fermi4filling_kpm(h0, filling, nk=nk, scale=scale,
                npol=npol, ne=ne, cores=cores, T=T, kpm_prec=kpm_prec)
    scale, xs, ys = _kpm_dos_moments(h, nk, scale, npol, ne, cores,
            kpm_prec=kpm_prec)
    cdf = _cumulative_trapz(ys, xs)
    cdf = np.maximum.accumulate(cdf)  # enforce monotonicity, see docstring
    cdf = cdf/cdf[-1]  # normalize exactly to 1 across the sampled window
    ef_reduced = np.interp(filling, cdf, xs)
    if T is None or T<=0.: return scale*ef_reduced # T=0, the step count
    # Finite temperature: inverting the cumulative DOS is a STEP count, and
    # the density matrix this Fermi level is handed to is built with
    # Fermi-Dirac occupations instead (see _dm_kpm_from_needed). Wherever
    # the density of states is not symmetric about mu the two hold
    # different numbers of electrons, so the converged filling drifts away
    # from the requested one as T grows -- the same defect repaired in
    # spectrum.get_fermi_energy_T for the exact-diagonalization path, of
    # which this is the KPM analogue. The electron count is monotonic in
    # mu, so bracket it on the sampled window and bisect.
    # The count is the cumulative DOS convolved with -df/dE, sampled in
    # units of T around mu rather than on the energy grid. A Fermi-Dirac
    # weight evaluated on the grid itself is a step whenever T is below
    # the grid spacing, as the mean field's default T=1e-7 always is, which
    # made the count a staircase in mu: the Fermi level was pinned to a
    # grid point and jumped by a whole grid step as the mean field moved,
    # and the KPM loop cycled at a floor instead of converging. With the
    # cumulative DOS interpolated linearly the count is continuous in mu at
    # any T, and below the grid spacing it is the T=0 inversion above.
    Tr = T/scale # the grid is in reduced energies, so the temperature is too
    u = np.linspace(-40.,40.,801) # energies around mu, in units of T
    w = expit(u)*expit(-u) # -df/du, the derivative of the Fermi function
    w = w/np.trapezoid(w,u)
    def nelec(mu): # occupied fraction at this chemical potential
        occ = np.interp(mu+Tr*u,xs,cdf,left=0.,right=1.)
        return np.trapezoid(occ*w,u)
    lo,hi = xs[0]-40.*Tr, xs[-1]+40.*Tr # well outside the window at this T
    if nelec(lo)>filling or nelec(hi)<filling: # not bracketed, keep T=0
        return scale*ef_reduced
    from scipy.optimize import brentq
    return scale*brentq(lambda mu: nelec(mu)-filling,lo,hi,xtol=1e-12)


def get_total_energy_kpm(h, fermi=0.0, nk=DEFAULT_NK, scale=None,
        npol=DEFAULT_NPOL, ne=None, cores=None, kpm_prec=None):
    """KPM analogue of spectrum.total_energy's exact-diagonalization path
    (its nbands=None default, which VJinteraction's integration="kpm"
    branch used to call unconditionally for its post-convergence total
    energy -- scftk/spinspin.py -- forcing a dense
    diagonalization there despite everything else in that branch staying
    diagonalization-free): the k-averaged sum of occupied eigenvalues of h
    (those below `fermi`), obtained by integrating E*rho(E) up to `fermi`
    instead of diagonalizing anything.

    Reuses get_fermi4filling_kpm's exact machinery (_kpm_dos_moments: the
    same k-averaged, Jackson-kernel-reconstructed per-orbital density of
    states) so the two functions can never silently disagree about what
    "the DOS" means, and renormalizes it the same way
    get_fermi4filling_kpm does -- dividing by its own cumulative integral's
    endpoint, since a finite-npol/ne reconstruction is not exactly
    normalized to one state per orbital -- before integrating, so the
    energy is internally self-consistent with whatever Fermi energy was
    located via that function on the same h.

    E*rho(E) is integrated over the reduced-energy grid up to fermi/scale
    (a plain grid-resolution truncation, not an interpolated boundary --
    consistent with how the rest of this module treats a finite ne grid,
    e.g. get_fermi4filling_kpm's own np.interp-based inversion), then
    rescaled: rho_E(E) = rho_x(x)/scale (x=E/scale, so dE=scale*dx), and
    the result is multiplied by norb since _kpm_dos_moments' profile is
    normalized per orbital (one state per orbital across the whole
    window) while this returns the EXTENSIVE total (summed over all norb
    orbitals), matching spectrum.total_energy's own per-k
    sum-of-eigenvalues convention (not an average over orbitals).

    For BdG/Nambu Hamiltonians, raises NotImplementedError rather than
    silently reusing get_fermi4filling_kpm's electron-only-spectrum
    workaround: that approximation is defensible for LOCATING a Fermi
    level (an already fuzzy concept for a superconductor's own BdG
    spectrum), but silently reusing it here would return the energy of the
    wrong (unpaired, non-superconducting) electron-only sector instead of
    the actual BdG spectrum's -- worth raising loudly rather than silently
    approximating twice over. VJinteraction's integration="kpm" path
    already excludes Nambu Hamiltonians entirely (see
    _run_anisotropic_scf's docstring), so this restriction is not new
    relative to what's already reachable.

    Verified against spectrum.total_energy on a frozen (non-SCF) 18-site
    honeycomb Hamiltonian with a random exchange field and sublattice
    imbalance (nk=6, npol=500): agreed to ~0.1% relative -- see
    tests/scf/test_densitydensity_kpm.py."""
    if h.has_eh:
        raise NotImplementedError("get_total_energy_kpm does not support "
                "BdG/Nambu Hamiltonians -- see its own docstring for why "
                "reusing get_fermi4filling_kpm's electron-only-spectrum "
                "workaround here specifically would silently return the "
                "wrong sector's energy rather than just being approximate")
    norb = h.intra.shape[0]
    scale, xs, ys = _kpm_dos_moments(h, nk, scale, npol, ne, cores,
            kpm_prec=kpm_prec)
    cdf = _cumulative_trapz(ys, xs)
    cdf = np.maximum.accumulate(cdf)  # enforce monotonicity, see get_fermi4filling_kpm
    norm = cdf[-1]  # same renormalization get_fermi4filling_kpm applies
    x_fermi = fermi/scale
    mask = xs <= x_fermi
    if not np.any(mask): return 0.0  # nothing occupied in the sampled window
    integral = np.trapezoid((xs*ys)[mask], x=xs[mask])
    return norb*scale/norm*integral
