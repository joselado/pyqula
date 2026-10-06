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
#      every k-point in one block recursion (kpmtk/pairmomentsnumba.py on
#      the CPU, kpmtk/pairmomentsjax.py on the GPU), and
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
    double precision on the CPU. Both backends take either one explicitly,
    and the names are those of the backend in use."""
    get_precision_names = _pair_moments().get_precision_names
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
                         kpm_prec=None, trace=False):
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

    The moments of every pair and every k-point come from a block
    recursion whose columns are the distinct starting vectors e_j, on the
    CPU with numba (kpmtk/pairmomentsnumba.py) and on the GPU with jax
    (kpmtk/pairmomentsjax.py), as the package-wide switch selects.
    kpm_prec is the precision of that recursion (see resolve_kpm_prec).
    This replaced one numba recursion per pair and per k: the per-pair
    loop ran the same recursion two or more times per site, and on a
    1728-orbital island took 27 s per evaluation against 1.6 s for the
    jax block in double precision and 0.46 s in single on a consumer card
    (future_development/gpu_kpm_mean_field.md). cores no longer splits
    the k-mesh here; it still sets the package-wide core count.

    trace=True also returns the trace moments of H(k), see
    _kpm_pair_values, as (dm, trace).

    A density-matrix entry is linear in its moments, so the energy
    integral of the Jackson-damped Chebyshev series against the Fermi
    weights is done once, on the basis, and each pair's value is the
    contraction of its moments with the resulting coefficients."""
    norb = h.intra.shape[0]
    needed = sorted(needed)
    ds = sorted({d for (d, i, j) in needed})
    # H(k) is Hermitian, and so is any function of it, so the (j,i) value
    # at each k is the conjugate of the (i,j) one. Computing both from
    # their own moments lets roundoff open an anti-Hermitian part in the
    # density matrix, which the mean-field loop amplifies by a fixed factor
    # every iteration until the recursion diverges. Only i<=j is computed
    # (_kpm_pair_values) and i>j is set by conjugation below, which makes
    # dm[-d][j,i] = conj(dm[d][i,j]) hold by construction and halves the
    # moment work. This is Hermiticity of the whole matrix, so it holds
    # just as well for an electron-hole entry of a Nambu H(k), whose
    # partner is the hole-electron entry, not another pairing one.
    pairs = sorted({(min(i, j), max(i, j)) for (_, i, j) in needed})
    pair_index = {p: idx for idx, p in enumerate(pairs)}
    out = _kpm_pair_values(h, np.array(pairs, dtype=np.int64).reshape(-1, 2),
            nk=nk, scale=scale, npol=npol, ne=ne, cores=cores, T=T,
            kpm_prec=kpm_prec, trace=trace)
    ks, vals = out[0], out[1]

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
    if trace: return dm, out[2]
    return dm


def _pair_moments():
    """The module of the block recursion the package-wide switch selects:
    numba's on the CPU, jax's on the GPU. Both have pair_values and
    trace_moments, and agree to roundoff"""
    from .. import gpu
    if gpu.get_gpu():
        from . import pairmomentsjax
        return pairmomentsjax
    from . import pairmomentsnumba
    return pairmomentsnumba


def _kpm_pair_values(h, pairs, nk=DEFAULT_NK, scale=None, npol=DEFAULT_NPOL,
        ne=None, cores=None, T=0.0, kpm_prec=None, trace=False):
    """The occupied-projector entry <c^dag_i c_j> of every pair (i,j) of
    pairs, an (npairs,2) integer array with i<=j, at every k-point of the
    mesh: returns the k-points and an (nk,npairs) array. This is the part
    of the KPM density matrix that both assemblies share, the dense one
    (_dm_kpm_from_needed) and the sparse one (get_dm_kpm_sparse), so the
    two engines compute the same numbers; see _dm_kpm_from_needed for the
    method.

    trace=True also returns (scale, mus), the scale of the expansion and
    the first 2*npol Chebyshev moments of the density of states of H(k)
    averaged over the orbitals and the k-mesh, (1/N) Tr T_n(H(k)/scale),
    which is what the Fermi search inverts (see LaggedFermi). They come
    from the same recursion as the density matrix, with the diagonal pair
    of every orbital added to the pairs, which costs little when the
    interaction already reads most diagonals."""
    kpm_prec = resolve_kpm_prec(kpm_prec)
    if ne is None: ne = npol*4
    ks = [list(k) for k in h.geometry.get_kmesh(nk=nk)]
    hk_gen = h.get_hk_gen()
    pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    diagonal = pairs[:, 0] == pairs[:, 1]

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
    backend = _pair_moments()
    ms = [csr_matrix(hk_gen(k))/scale for k in ks]
    norb = ms[0].shape[0]
    # the pair (i,j) holds <e_i|T_n(H(k))|e_j>: the recursion starts from
    # e_j and is read on row i, which is the element dm[i,j] (see
    # densitymatrix.py's restricted_dm for the convention)
    if trace:
        # every diagonal pair once, so the summed diagonal moments are the trace
        keys = pairs[:, 0]*norb + pairs[:, 1]
        every = np.union1d(keys, np.arange(norb, dtype=np.int64)*(norb + 1))
        both = np.stack([every//norb, every % norb], axis=1)
        vals, mumax, tr = backend.pair_values(ms, both, cint,
                kpm_prec=kpm_prec, trace=True)
        vals = vals[:, np.searchsorted(every, keys)]
        mus = np.mean(tr, axis=0)/norb # k-average of the moments
    else:
        vals, mumax = backend.pair_values(ms, pairs, cint, kpm_prec=kpm_prec)
    _check_scale_covers_spectrum([mumax], scale, given, kpm_prec=kpm_prec)
    vals = vals.conj()
    vals[:, diagonal] = vals[:, diagonal].real # a diagonal entry is its own conjugate
    if trace: return ks, vals, (scale, mus.real)
    return ks, vals


def get_dm_kpm_sparse(h, needed, nk=DEFAULT_NK, scale=None,
        npol=DEFAULT_NPOL, ne=None, cores=None, T=0.0, kpm_prec=None,
        trace=False):
    """Sparse counterpart of _dm_kpm_from_needed, for a large sparse
    Hamiltonian: needed is a dictionary {direction: (rows, cols)} of index
    arrays, and the density matrix comes back as {direction: csr_matrix}
    holding those entries only, so its memory is linear in their number
    rather than norb^2 per direction. The values are those of
    _dm_kpm_from_needed, from the same _kpm_pair_values, and the
    bookkeeping is done on integer arrays (one int64 key per pair) rather
    than on Python sets and dictionaries of tuples, which at 10^5 sites
    would be millions of objects. trace=True also returns the trace
    moments of H(k), see _kpm_pair_values, as (dm, trace)."""
    norb = h.intra.shape[0]
    ds = list(needed)
    rows = [np.asarray(needed[d][0], dtype=np.int64) for d in ds]
    cols = [np.asarray(needed[d][1], dtype=np.int64) for d in ds]
    allr, allc = np.concatenate(rows), np.concatenate(cols)
    # one key per unordered pair, i<=j, see _dm_kpm_from_needed
    keys = np.minimum(allr, allc)*norb + np.maximum(allr, allc)
    ukeys, inverse = np.unique(keys, return_inverse=True)
    pairs = np.stack([ukeys//norb, ukeys % norb], axis=1)
    out = _kpm_pair_values(h, pairs, nk=nk, scale=scale, npol=npol,
            ne=ne, cores=cores, T=T, kpm_prec=kpm_prec, trace=trace)
    ks, vals = out[0], out[1]
    fac = 1./len(ks)
    dm = dict()
    start = 0
    for d, r, c in zip(ds, rows, cols):
        idx = inverse[start:start+len(r)] ; start += len(r)
        phases = np.exp(2j*np.pi*np.array(ks) @ np.array(d, dtype=float))
        v = vals[:, idx]
        flip = r > c # the entries below the diagonal, by conjugation
        v[:, flip] = v[:, flip].conj()
        data = fac*(phases @ v)
        dm[d] = csr_matrix((data, (r, c)), shape=(norb, norb))
    if trace: return dm, out[2]
    return dm


def get_dm_kpm(h, v, nk=DEFAULT_NK, scale=None, npol=DEFAULT_NPOL, ne=None,
               cores=None, T=0.0, kpm_prec=None, trace=False, **kwargs):
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
    Chebyshev recursion, see resolve_kpm_prec. trace=True also returns the
    trace moments of H(k), see _kpm_pair_values, as (dm, trace)."""
    ds = [(0, 0, 0)] + [d for d in v if d != (0, 0, 0)]
    if getattr(h, "has_eh", False):
        needed = required_elements_eh(v)
    else:
        needed = required_elements(v)
    out = _dm_kpm_from_needed(h, needed, nk=nk, scale=scale, npol=npol,
            ne=ne, cores=cores, T=T, kpm_prec=kpm_prec, trace=trace)
    dm = out[0] if trace else out
    # every direction v has a key for must be present in the output, even
    # if it happened to contribute no required entries of its own
    for d in ds:
        if d not in dm:
            dm[d] = np.zeros((h.intra.shape[0], h.intra.shape[0]),
                              dtype=np.complex128)
    if trace: return dm, out[1]
    return dm


def _cumulative_trapz(y, x):
    """cumulative_trapz(y,x)[k] = trapezoidal integral of y from x[0] to
    x[k] (a small local helper so this module doesn't depend on scipy's
    cumulative_trapezoid, whose name/availability has moved across scipy
    versions)."""
    dx = np.diff(x)
    avg = (y[1:]+y[:-1])/2.
    return np.concatenate([[0.], np.cumsum(avg*dx)])


def _kpm_trace(h, nk, scale, npol, cores, kpm_prec=None):
    """The scale of the expansion and the first 2*npol Chebyshev moments of
    the density of states of H(k), averaged over the orbitals and the
    k-mesh, (1/N) Tr T_n(H(k)/scale): the trace is exact, one recursion
    started on every orbital, not a stochastic estimate. The moments are
    linear in the density of states, so their k-average is the moments of
    the k-averaged density of states.

    The trace comes from the block recursion of the density matrix
    (pairmomentsnumba on the CPU, pairmomentsjax on the GPU) with every
    orbital as a starting column and doubling, every k-point in one call.
    kpm_prec is its precision, see resolve_kpm_prec."""
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
    if cores is not None: parallel.set_cores(cores)
    ms = [csr_matrix(hk_gen(k))/scale for k in ks]
    musk, mumax = _pair_moments().trace_moments(ms, 2*npol, kpm_prec=kpm_prec)
    _check_scale_covers_spectrum([mumax], scale, given, kpm_prec=kpm_prec)
    return scale, np.mean(musk, axis=0).real # k-average of the moments


def _dos_profile(mus, ne):
    """The Jackson-kernel density of states per orbital of the moments mus,
    on the reduced-energy grid xs ([-0.99,0.99], ne points): returns
    (xs, ys)"""
    xs = np.linspace(-1.0, 1.0, ne, endpoint=True)*0.99  # reduced energies
    ys = generate_profile(mus, xs, kernel="jackson").real
    return xs, ys


def _kpm_dos_moments(h, nk, scale, npol, ne, cores, kpm_prec=None):
    """Shared per-k Bloch-KPM engine for get_fermi4filling_kpm and
    get_total_energy_kpm: the k-averaged trace moments of _kpm_trace,
    reconstructed on the standard reduced-energy grid via the Jackson
    kernel. Returns (scale, xs, ys): xs the reduced-energy grid
    ([-0.99,0.99]), ys the (real-valued) reconstructed DOS profile on it --
    ready for either a cumulative-integral inversion (Fermi search) or an
    energy integral (total energy) downstream, so the two functions that
    use this can never silently disagree about what "the DOS" means."""
    if ne is None: ne = npol*4
    scale, mus = _kpm_trace(h, nk, scale, npol, cores, kpm_prec=kpm_prec)
    xs, ys = _dos_profile(mus, ne)
    return scale, xs, ys


def _filling_count(scale, xs, ys, T=0.):
    """From the density of states ys per orbital on the reduced-energy grid
    xs of an expansion with this scale, the fraction of the states occupied
    at a chemical potential, as a function count(mu) of mu in the energy
    units of H, and the Fermi energy of a filling, as a function
    fermi(filling). count(fermi(f)) is f, up to the grid.

    The cumulative integral of the density of states gives the fraction of
    orbitals occupied as a function of energy (0 at the sampled window's
    bottom, 1 at its top), forced monotonic (np.maximum.accumulate) before
    it is inverted: a finite-npol Jackson-kernel reconstruction is not
    guaranteed nonnegative everywhere (Gibbs-type ringing near band
    edges, gaps and van Hove singularities), which without this would make
    the inversion via np.interp ill-defined.

    At a finite T the count is the cumulative density of states convolved
    with -df/dE, so that it holds as many electrons as the density matrix
    built with Fermi-Dirac occupations at the same T (_dm_kpm_from_needed):
    with a step count instead, the converged filling drifts away from the
    requested one wherever the density of states is not symmetric about
    mu, the defect repaired in spectrum.get_fermi_energy_T for the
    exact-diagonalization path. The convolution is sampled in units of T
    around mu rather than on the energy grid: a Fermi-Dirac weight
    evaluated on the grid itself is a step whenever T is below the grid
    spacing, as the mean field's default T=1e-7 always is, which made the
    count a staircase in mu, pinned the Fermi level to a grid point that
    jumped by a whole grid step as the mean field moved, and made the KPM
    loop cycle at a floor instead of converging. With the cumulative
    density of states interpolated linearly the count is continuous in mu
    at any T, and below the grid spacing it is the T=0 count. The count is
    monotonic in mu, so its inversion brackets it on the sampled window
    and bisects."""
    cdf = _cumulative_trapz(ys, xs)
    cdf = np.maximum.accumulate(cdf)  # enforce monotonicity, see above
    cdf = cdf/cdf[-1]  # normalize exactly to 1 across the sampled window
    if T is None or T<=0.: # the step count
        def count(mu): return np.interp(mu/scale, xs, cdf)
        def fermi(filling): return scale*np.interp(filling, cdf, xs)
        return count, fermi
    Tr = T/scale # the grid is in reduced energies, so the temperature is too
    u = np.linspace(-40.,40.,801) # energies around mu, in units of T
    w = expit(u)*expit(-u) # -df/du, the derivative of the Fermi function
    w = w/np.trapezoid(w,u)
    def nelec(x): # occupied fraction at the reduced chemical potential x
        occ = np.interp(x+Tr*u,xs,cdf,left=0.,right=1.)
        return np.trapezoid(occ*w,u)
    def count(mu): return nelec(mu/scale)
    lo,hi = xs[0]-40.*Tr, xs[-1]+40.*Tr # well outside the window at this T
    def fermi(filling):
        if nelec(lo)>filling or nelec(hi)<filling: # not bracketed, keep T=0
            return scale*np.interp(filling, cdf, xs)
        from scipy.optimize import brentq
        return scale*brentq(lambda x: nelec(x)-filling,lo,hi,xtol=1e-12)
    return count, fermi


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
    profile from _kpm_dos_moments (see its docstring) and inverts its
    cumulative integral at the filling, at the temperature T (see
    _filling_count).

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
    count, fermi = _filling_count(scale, xs, ys, T=T)
    return fermi(filling)


class LaggedFermi:
    """The Fermi level of a KPM mean-field loop at a fixed filling, taken
    from the recursion of the previous iteration instead of a search of
    its own. The trace that the Fermi search inverts, Tr T_n(H), is the
    sum of the diagonal moments, which the density-matrix recursion
    computes anyway once every orbital is a starting column, so on the CPU
    the Fermi level costs nothing beyond the diagonals the interaction
    does not read. Its price is a lag of one iteration: the Fermi level
    of the density matrix of iteration n is that of the Hamiltonian of
    iteration n-1, so the iterates are not at the requested filling
    until the mean field stops moving. The loop then reads the filling
    error of every density matrix from the same trace, and holds it to
    the convergence tolerance together with the change in the mean field.
    At convergence the Hamiltonian no longer changes, and the Fermi level
    is the exact one.

    The first Hamiltonian, which has no previous iteration, gets an exact
    search (get_fermi4filling_kpm), and so does every Hamiltonian with
    Nambu: its Fermi level is that of the electron-only Hamiltonian, which
    is a different matrix from the one the recursion runs on.

    The Fermi level a converged loop ends at is the one whose filling, read
    from the trace of the expansion of the shifted Hamiltonian, is the
    requested one to the tolerance. An exact search finds it from the
    expansion of the unshifted Hamiltonian, whose scale is different, so
    the two differ by what npol resolves, 2e-3 in the Fermi level of a
    16-site Hubbard island at npol=150, which is not an error of either
    but the KPM resolution of the Fermi level.

    Use: shift(h) before the density matrix of h, which also sets h.fermi,
    then update(h, trace) with the trace moments of that same shifted h
    (get_dm_kpm and get_dm_kpm_sparse with trace=True), and wants_trace(h)
    says whether update needs them. error is the filling error of the
    last density matrix, the occupied fraction at the Fermi level it was
    computed at minus the requested filling."""
    def __init__(self, filling, nk=DEFAULT_NK, scale=None, npol=DEFAULT_NPOL,
            ne=None, cores=None, T=0., kpm_prec=None):
        self.filling = filling
        self.kw = dict(nk=nk, scale=scale, npol=npol, ne=ne, cores=cores,
                T=T, kpm_prec=kpm_prec)
        self.ne = npol*4 if ne is None else ne
        self.T = T
        self.mu = None # the Fermi level the next Hamiltonian is shifted by
        self.error = 0.

    def wants_trace(self, h):
        return not h.has_eh

    def shift(self, h):
        """Shift h by the Fermi level, searching for it when there is no
        previous iteration to take it from"""
        if self.mu is None or h.has_eh:
            self.mu = get_fermi4filling_kpm(h, self.filling, **self.kw)
        h.fermi = self.mu
        h.shift_fermi(-self.mu)
        return h

    def update(self, h, trace):
        """trace, the (scale, moments) of the shifted h of this iteration:
        records its filling error, and moves the Fermi level of the next
        iteration to the Fermi level of h"""
        if not self.wants_trace(h):
            self.error = 0.
            return
        scale, mus = trace
        xs, ys = _dos_profile(mus, self.ne)
        count, fermi = _filling_count(scale, xs, ys, T=self.T)
        self.error = abs(count(0.) - self.filling)
        self.mu = self.mu + fermi(self.filling)


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


def get_band_energy_kpm(h, nk=DEFAULT_NK, scale=None, npol=DEFAULT_NPOL,
        ne=None, cores=None, T=0.0, kpm_prec=None):
    """The band energy Tr(H rho) per unit cell, from the KPM density matrix
    on the entries of H only, sum_d sum_ij H_d[i,j] dm[d][i,j], so that it
    costs memory linear in the number of hoppings: the total-energy term
    that h.get_total_energy gets by diagonalizing. For a Nambu Hamiltonian
    it is (sum_{E<0} E + Tr h_e)/2, as spectrum.total_energy explains, with
    Tr h_e averaged over the same k-mesh. It is the energy of the
    occupation the KPM density matrix describes, the Fermi function at T
    resolved with npol moments, and so it differs from the sum of the
    occupied eigenvalues by the KPM truncation, the error of the density
    matrix itself, and by the thermal smearing at a finite T."""
    from scipy.sparse import coo_matrix
    hd = {tuple(int(x) for x in d): csr_matrix(m) for (d, m) in h.get_dict().items()}
    needed = dict()
    for d, m in hd.items():
        c = coo_matrix(m)
        needed[d] = (c.row.astype(np.int64), c.col.astype(np.int64))
    dm = get_dm_kpm_sparse(h, needed, nk=nk, scale=scale, npol=npol, ne=ne,
            cores=cores, T=T, kpm_prec=kpm_prec)
    e = sum(m.multiply(dm[d]).sum() for (d, m) in hd.items())
    if h.has_eh:
        from ..superconductivity import get_eh_sector
        ks = np.array(h.geometry.get_kmesh(nk=nk))
        tr = 0.
        for d, m in hd.items(): # Tr h_e(k), averaged over the k-mesh
            phase = np.mean(np.exp(2j*np.pi*ks @ np.array(d, dtype=float)))
            tr += get_eh_sector(m, i=0, j=0).diagonal().sum()*phase
        e = (e + tr)/2.
    return np.real(e)
