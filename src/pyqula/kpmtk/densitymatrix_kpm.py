# KPM (Chebyshev/sparse) density matrix, restricted to the elements
# actually required by a density-density interaction.
#
# The existing SCF machinery (selfconsistency/densitydensity.py) always
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
from scipy.sparse import csr_matrix
from scipy.special import expit

from .. import kpm
from .. import parallel
from .bandwidth import estimate_bandwidth
from .momenttoprofile import generate_profile

# Shared defaults for the KPM SCF's tuning knobs. selfconsistency/
# densitydensity_kpm.py's generic_densitydensity_kpm/densitydensity_kpm
# reference these same constants (rather than separately hardcoding their
# own copies) so the density-matrix computation and the Fermi-energy
# search can never silently drift apart on what "unspecified" means.
DEFAULT_NK = 8
DEFAULT_NPOL = 200


def required_elements(v, tol=1e-10):
    """Given the interaction dictionary v (lattice vector -> matrix),
    return the set of (direction, i, j) density-matrix entries actually
    read by selfconsistency/densitydensity.py for every nonzero v[d][i,j]:
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
    selfconsistency/superscf.py's anomalous_term_ij_jit, which for a given
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

    get_mf's has_eh branch (selfconsistency/densitydensity.py) extracts
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

    T is the same finite-temperature smearing selfconsistency/
    densitydensity.py's ED path applies via Fermi-Dirac occupation
    (densitymatrix.py's full_dm(h,T=...)): rather than a hard cutoff at
    the Fermi energy (E=0), the occupied-window integration is weighted by
    the Fermi function at temperature T, and the window is extended a bit
    above 0 so that weight isn't dropped. T=0 (the default) is treated the
    same tiny regularization (1e-15) full_dm itself uses, recovering an
    effectively-hard cutoff."""
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
        scale = 1.1*max(estimate_bandwidth(hk_gen(k)) for k in ks)
    if scale <= 0:
        raise ValueError("H(k) has zero bandwidth on the sampled k-mesh "
                "(it vanishes at every k) -- cannot set a KPM energy "
                "scale; check that this Hamiltonian actually has "
                "hopping/onsite terms in this sector")
    Tsafe = abs(T) if T != 0. else 1e-15
    upper = min(0.99*scale, 30.*Tsafe)
    xin = np.linspace(-0.99*scale, upper, ne)
    weights = expit(-xin/Tsafe)  # Fermi-Dirac occupation at temperature T

    def compute_for_k(k):
        Hk = csr_matrix(hk_gen(k))
        out = np.zeros(len(pairs), dtype=np.complex128)
        for idx, (i, j) in enumerate(pairs):
            # kpm.dm_ij_energy(m,i=a,j=b,...) integrates to the density-
            # matrix element conventionally written dm[b,a] (see
            # densitymatrix.py's restricted_dm, which cross-checks its
            # "KPM" mode called with (i=a,j=b) against its "full" mode's
            # dm[b,a] for the same (a,b) pair) -- so to land in dm[i,j]
            # here the KPM call needs its arguments swapped.
            (x, y) = kpm.dm_ij_energy(Hk, i=j, j=i, scale=scale, npol=npol,
                                       ne=ne, x=xin)
            out[idx] = np.trapz(y*weights, x=x)/np.pi
        return out

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
    """KPM-based analogue of selfconsistency.densitydensity.get_dm: return
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


def get_fermi4filling_kpm(h, filling, nk=DEFAULT_NK, scale=None,
        npol=DEFAULT_NPOL, ne=None, cores=None):
    """KPM analogue of spectrum.get_fermi4filling: find the Fermi energy
    for a given filling without ever diagonalizing anything, so the KPM
    SCF (selfconsistency/densitydensity_kpm.py) stays fully
    diagonalization-free end to end -- otherwise it would still need
    spectrum.get_fermi4filling's own per-k diagonalization just to locate
    the Fermi level, even though the density matrix itself is computed via
    KPM.

    Samples the same k-mesh as the rest of the KPM path
    (h.geometry.get_kmesh(nk=nk)), and at each k gets the Chebyshev moments
    of the local density of states averaged over every orbital in the cell
    via kpm.full_trace -- a deterministic sum over all sites/orbitals
    (looping i=0..norb-1), not a stochastic random-vector estimate -- then
    k-averages those moments into a single total-DOS-per-orbital profile
    (valid because moments are linear in the density of states, so the
    k-average of the moments equals the moments of the k-averaged DOS).
    Its cumulative integral gives the fraction of orbitals occupied as a
    function of energy (0 at the sampled window's bottom, 1 at its top);
    inverting it at the target filling gives the Fermi energy directly,
    with no diagonalization anywhere. The cumulative integral is forced
    monotonic (np.maximum.accumulate) before inversion: a finite-npol
    Jackson-kernel KPM reconstruction is not guaranteed nonnegative
    everywhere (Gibbs-type ringing near band edges/gaps/van Hove
    singularities), which without this would make the inversion via
    np.interp silently ill-defined.

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
    if ne is None: ne = npol*4
    ks = [list(k) for k in h.geometry.get_kmesh(nk=nk)]
    hk_gen = h.get_hk_gen()
    if scale is None:
        scale = 1.1*max(estimate_bandwidth(hk_gen(k)) for k in ks)
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
    cdf = _cumulative_trapz(ys, xs)
    cdf = np.maximum.accumulate(cdf)  # enforce monotonicity, see docstring
    cdf = cdf/cdf[-1]  # normalize exactly to 1 across the sampled window
    ef_reduced = np.interp(filling, cdf, xs)
    return scale*ef_reduced
