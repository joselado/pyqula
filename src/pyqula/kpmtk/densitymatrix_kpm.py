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
#      via Chebyshev recursion (kpm.dm_ij_energy) instead of diagonalizing
#      H(k), and
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
from .kpmnumba import kpm_moments_ij as get_moments_ij
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

    Only valid for spinful Nambu Hamiltonians (h.has_eh and h.has_spin) --
    a spinless Nambu Hamiltonian has only 2 Nambu slots per site (electron,
    hole; no spin), a different convention this formula does not describe
    -- see get_dm_kpm's has_spin check."""
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


def _dm_kpm_from_needed(h, needed, nk=DEFAULT_NK, scale=None,
                         npol=DEFAULT_NPOL, ne=None, cores=None, T=0.0):
    """Shared per-k Bloch-KPM engine: given the (direction, row, col)
    density-matrix entries to compute (in whatever index convention the
    caller's "needed" set already uses -- see required_elements/
    required_elements_eh), sample the same k-mesh the exact-diagonalization
    path uses (h.geometry.get_kmesh(nk=nk)), and at each k build the small
    Bloch Hamiltonian H(k) and get each needed <i|P_occ(H(k))|j>
    occupied-projector element via kpm.dm_ij_energy instead of
    diagonalizing H(k). A given (i,j) pair's H(k)-projector only needs to
    be computed once per k (not once per direction) -- every direction
    that needs it reuses the same per-k value, weighted by the Bloch phase
    exp(2*pi*i*k.d) and summed over k, exactly mirroring the
    exact-diagonalization path's own phase convention (dmtk/fulldm.py).

    T is the same finite-temperature smearing scftk/
    densitydensity.py's ED path applies via Fermi-Dirac occupation
    (densitymatrix.py's full_dm(h,T=...)): rather than a hard cutoff at
    the Fermi energy (E=0), the occupied-window integration is weighted by
    the Fermi function at temperature T, and the window is extended a bit
    above 0 so that weight isn't dropped. T=0 (the default) is treated the
    same tiny regularization (1e-15) full_dm itself uses, recovering an
    effectively-hard cutoff.

    Per-pair moments (get_moments_ij) are still computed one call per
    (row,col,k) triple, each running its own O(npol) Chebyshev VECTOR
    recursion -- but converting those moments into the requested
    density-matrix element used to also go through kpm.dm_ij_energy's own
    call to generate_profile per pair, which recomputes the
    Jackson-kernel-damped Chebyshev-polynomial basis (a (2*npol,
    ne)-shaped array) from scratch every single call even though it
    depends only on scale/npol/ne/xin, none of which vary across pairs or
    k. Building it once (`basis` below) and reducing each pair's moments
    to its density-matrix value via one matrix-vector product against it
    (batched into one matrix-matrix product across all pairs at a given k)
    turned out to dominate the entire computation: profiled at 81% of
    _dm_kpm_from_needed's total time on a 98-site/196-orbital honeycomb
    Hubbard system (nk=4, npol=200, 392 needed pairs) before this change,
    cutting the isolated density-matrix computation from ~20.7s to ~6.9s
    there (~3x) -- verified against the exact-diagonalization ("ed") path
    to ~1e-7.

    NOT yet batched, and left for a future pass (2026-07-27): the moment
    recursion itself. get_moments_ij(m,i=a,j=b) internally starts a
    Chebyshev VECTOR recursion from e_a and projects it onto e_b at each
    step -- so every needed pair with the SAME starting index `a` (shared
    whenever multiple density-matrix rows read the same column, e.g. all
    4 entries of Hubbard's onsite 2x2 spin block per site share 2 starting
    columns between them) redundantly reruns that recursion from scratch
    instead of computing it once and extracting multiple projections from
    it. Even with the fix already applied here, KPM remains far SLOWER
    than "ed" at the system sizes actually measured (order 100-500 sites:
    ED's dense per-k LAPACK diagonalization is extremely fast regardless
    of algorithmic complexity at that scale) -- see VJinteraction's
    docstring (scftk/spinspin.py) for the full measured
    comparison. get_fermi4filling_kpm's own O(n_orb) per-orbital Fermi
    search (below) is a separate, also-unaddressed cost of comparable or
    greater size."""
    if ne is None: ne = npol*4
    norb = h.intra.shape[0]
    ks = [list(k) for k in h.geometry.get_kmesh(nk=nk)]
    hk_gen = h.get_hk_gen()

    needed = sorted(needed)
    ds = sorted({d for (d, i, j) in needed})
    pairs = sorted({(i, j) for (_, i, j) in needed})
    pair_index = {p: idx for idx, p in enumerate(pairs)}

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

    # get_moments_ij(...,n=npol) returns 2*npol moments (kpmnumba's own
    # convention -- see numba_kpm_moments_ij), so the basis needs the same
    # length to pair up with them below.
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

    def compute_for_k(k):
        Hk = csr_matrix(hk_gen(k))
        Hk_scaled = Hk/scale  # hoisted out of the pair loop: same for every pair at this k
        mus_batch = np.zeros((len(pairs), n_moments), dtype=np.complex128)
        for idx, (i, j) in enumerate(pairs):
            # get_moments_ij(m,i=a,j=b) yields the moments for the density-
            # matrix element conventionally written dm[b,a] (see
            # densitymatrix.py's restricted_dm, which cross-checks its
            # "KPM" mode called with (i=a,j=b) against its "full" mode's
            # dm[b,a] for the same (a,b) pair) -- so to land in dm[i,j]
            # here the call needs its arguments swapped.
            mus_batch[idx] = get_moments_ij(Hk_scaled, i=j, j=i, n=npol)
        ysr = mus_batch.real @ basis  # (len(pairs), ne)
        ysi = mus_batch.imag @ basis
        ys = ysr - 1j*ysi
        return np.trapezoid(ys*weights[None, :], x=xin, axis=1)/np.pi

    if cores is not None: parallel.set_cores(cores)
    results = parallel.pcall(compute_for_k, ks)  # one array of pair values per k

    needed_by_d = dict()
    for d, i, j in needed: needed_by_d.setdefault(d, []).append((i, j))

    dm = {d: np.zeros((norb, norb), dtype=np.complex128) for d in ds}
    fac = 1./len(ks)
    for d in ds:
        phases = np.array([np.exp(2j*np.pi*np.dot(k, d)) for k in ks])
        for (i, j) in needed_by_d.get(d, []):
            idx = pair_index[(i, j)]
            col = np.array([r[idx] for r in results])
            dm[d][i, j] = fac*np.sum(phases*col)
    return dm


def get_dm_kpm(h, v, nk=DEFAULT_NK, scale=None, npol=DEFAULT_NPOL, ne=None,
               cores=None, T=0.0, **kwargs):
    """KPM-based analogue of scftk.densitydensity.get_dm: return
    a dictionary {direction: matrix} with the density matrix, but computing
    only the entries that v actually requires, each one through a sparse
    Chebyshev-moment (KPM) correlator instead of full diagonalization.
    Meant for sparse/large Hamiltonians where exact diagonalization of a
    dense k-mesh becomes the bottleneck.

    For BdG/Nambu Hamiltonians (h.has_eh) the required entries are
    determined by required_elements_eh instead of required_elements (see
    its docstring) -- both are then handed to the same per-k Bloch-KPM
    engine, _dm_kpm_from_needed. required_elements_eh/_local_nambu_index
    assume a *spinful* Nambu Hamiltonian (h.has_spin too); a spinless BdG
    Hamiltonian uses a different (2 slots/site) Nambu convention this path
    does not implement, so that combination is rejected explicitly rather
    than silently mapped through the wrong formula."""
    if getattr(h, "has_eh", False) and not getattr(h, "has_spin", True):
        raise NotImplementedError("get_dm_kpm's BdG/Nambu path only "
                "supports spinful Hamiltonians (h.has_spin=True); "
                "spinless_nambu uses a different Nambu index convention "
                "not implemented here")
    ds = [(0, 0, 0)] + [d for d in v if d != (0, 0, 0)]
    if getattr(h, "has_eh", False):
        needed = required_elements_eh(v)
    else:
        needed = required_elements(v)
    dm = _dm_kpm_from_needed(h, needed, nk=nk, scale=scale, npol=npol,
                              ne=ne, cores=cores, T=T)
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


def _kpm_dos_moments(h, nk, scale, npol, ne, cores):
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

    PERFORMANCE NOTE (not addressed as of 2026-07-27, unlike
    _dm_kpm_from_needed's per-pair profile reconstruction, which was): the
    "deterministic sum over all sites/orbitals" above is exactly that --
    one full kpm.full_trace moment computation per orbital, i.e. O(n_orb)
    separate calls per k, each its own O(npol) recursion. This does not
    share any of _dm_kpm_from_needed's batching (different call path
    entirely), and measured as a comparable-or-larger fraction of a
    VJinteraction "kpm" SCF iteration's total cost than the density matrix
    itself once that was optimized (e.g. ~2.6s of a ~9s iteration on a
    98-site/196-orbital honeycomb system, nk=4, npol=200). A stochastic
    trace estimator (a handful of random vectors instead of one deterministic
    vector per orbital) would be the standard KPM fix, at the cost of
    trading exactness for statistical noise -- not attempted here."""
    if ne is None: ne = npol*4
    ks = [list(k) for k in h.geometry.get_kmesh(nk=nk)]
    hk_gen = h.get_hk_gen()
    if scale is None:
        scale = _estimate_kpm_scale(hk_gen, ks)
    if scale <= 0:
        raise ValueError("H(k) has zero bandwidth on the sampled k-mesh "
                "(it vanishes at every k) -- cannot set a KPM energy "
                "scale; check that this Hamiltonian actually has "
                "hopping/onsite terms in this sector")

    def moments_for_k(k):
        Hk = csr_matrix(hk_gen(k))
        return kpm.full_trace(Hk/scale, n=npol)

    if cores is not None: parallel.set_cores(cores)
    results = parallel.pcall(moments_for_k, ks)
    mus = sum(results)/len(results)  # k-average of the moments

    xs = np.linspace(-1.0, 1.0, ne, endpoint=True)*0.99  # reduced energies
    ys = generate_profile(mus, xs, kernel="jackson").real
    return scale, xs, ys


def get_fermi4filling_kpm(h, filling, nk=DEFAULT_NK, scale=None,
        npol=DEFAULT_NPOL, ne=None, cores=None):
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
                npol=npol, ne=ne, cores=cores)
    scale, xs, ys = _kpm_dos_moments(h, nk, scale, npol, ne, cores)
    cdf = _cumulative_trapz(ys, xs)
    cdf = np.maximum.accumulate(cdf)  # enforce monotonicity, see docstring
    cdf = cdf/cdf[-1]  # normalize exactly to 1 across the sampled window
    ef_reduced = np.interp(filling, cdf, xs)
    return scale*ef_reduced


def get_total_energy_kpm(h, fermi=0.0, nk=DEFAULT_NK, scale=None,
        npol=DEFAULT_NPOL, ne=None, cores=None):
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
    scale, xs, ys = _kpm_dos_moments(h, nk, scale, npol, ne, cores)
    cdf = _cumulative_trapz(ys, xs)
    cdf = np.maximum.accumulate(cdf)  # enforce monotonicity, see get_fermi4filling_kpm
    norm = cdf[-1]  # same renormalization get_fermi4filling_kpm applies
    x_fermi = fermi/scale
    mask = xs <= x_fermi
    if not np.any(mask): return 0.0  # nothing occupied in the sampled window
    integral = np.trapezoid((xs*ys)[mask], x=xs[mask])
    return norb*scale/norm*integral
