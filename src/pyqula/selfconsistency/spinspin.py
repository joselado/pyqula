# Mean-field (Hartree-Fock) treatment of spin-spin (S_i.S_j) interactions
# in spinful Hamiltonians.
#
# Sz_i Sz_j = 1/4 (n_iu - n_id)(n_ju - n_jd)
#           = 1/4 (n_iu n_ju - n_iu n_jd - n_id n_ju + n_id n_jd)
#
# is already a density-density interaction (sum of V_ab n_a n_b over
# spin-orbitals a,b) of exactly the form that selfconsistency.densitydensity's
# generic Hartree-Fock engine decouples -- it does not know or care what the
# spin-orbital index physically represents. So SzSz needs no new decoupling
# math, only a v matrix with the right +/-1/4 sign pattern in the 2x2 spin
# blocks (see _build_v below), fed into the existing densitydensity() engine.
#
# SxSx and SySy are obtained by a global spin rotation that maps the physical
# x (or y) axis onto the computational z axis, running the SzSz machinery
# there, and rotating the converged Hamiltonian back. Concretely, writing
# c' = U c for a site-independent spin rotation U, a Hamiltonian h written in
# terms of c becomes h_tilde = U h U^dagger written in terms of c'; the
# density matrix transforms the same way. Requiring U^dagger sz U = sx (so
# that "Sz" measured with the primed operators is "Sx" of the original ones)
# is satisfied by the existing rotate_spin.global_spin_rotation machinery
# (exposed as Hamiltonian.global_spin_rotation) for:
#   x-axis: vector=[0,1,0], angle=+0.5   (undo with angle=-0.5)
#   y-axis: vector=[1,0,0], angle=-0.5   (undo with angle=+0.5)
# verified numerically against the Pauli matrices.

import numpy as np

from .. import specialhopping
from ..multihopping import MultiHopping
from ..rotate_spin import global_spin_rotation as _gsr

# NOTE: densitydensity is imported lazily (inside each function, not here at
# module level). densitydensity.py itself does `from ..meanfield import
# identify_symmetry_breaking` at its very bottom, and meanfield.py imports
# this module (spinspin) to re-export SzSz/SxSx/SySy/Jinteraction -- an
# eager top-level import here would close that cycle and fail (with an
# AttributeError on a partially-initialized module) depending on which
# module a caller happens to import first.


def _build_v(h, J1=0.0, J2=0.0, J3=0.0, Jr=None, nd=None):
    """Build the spin-orbital interaction matrix for a J1/J2/J3 (plus
    optional general Jr(r) function) neighbor-shell SzSz coupling,
    following exactly the same neighbor-shell/hopping-dict construction as
    Vinteraction, but with the +/-1/4 sign pattern of
    Sz_i Sz_j = 1/4 (n_iu-n_id)(n_ju-n_jd) in the four spin blocks instead
    of Vinteraction's uniform value. Same key set (bond directions) as
    Vinteraction's v for the same geometry, since that is fixed purely by
    the geometry's neighbor shells, independent of which J's are zero.

    nd: precomputed h.geometry.neighbor_distances(), if the caller already
    has one (e.g. Jinteraction/VJinteraction, which call this -- and
    _build_density_v -- several times for the same h and would otherwise
    recompute this O(n^2) geometry search from scratch on every call).
    Computed here if not given, so single-channel callers (SzSz) are
    unaffected."""
    if nd is None: nd = h.geometry.neighbor_distances() # distances to the neighbor shells
    mgenerator = specialhopping.distance_hopping_matrix(
            [J1/2., J2/2., J3/2.], nd[0:3])
    hv = h.geometry.get_hamiltonian(has_spin=False, is_multicell=True,
            mgenerator=mgenerator)
    if Jr is not None:
        hv1 = h.geometry.get_hamiltonian(has_spin=False, is_multicell=True,
                tij=Jr)
        hv = hv + hv1
    v = hv.get_hopping_dict()
    for d in v:
        m = v[d]
        n = m.shape[0]
        m1 = np.zeros((2*n, 2*n), dtype=np.complex128)
        for i in range(n):
            for j in range(n):
                m1[2*i, 2*j] = m[i, j]/4.       # up-up
                m1[2*i, 2*j+1] = -m[i, j]/4.    # up-down
                m1[2*i+1, 2*j] = -m[i, j]/4.    # down-up
                m1[2*i+1, 2*j+1] = m[i, j]/4.   # down-down
        v[d] = m1
    return v


def _callback_mf_constrains(h, constrains):
    if not constrains: return None
    from . import mfconstrains
    def callback_mf(mf):
        return mfconstrains.enforce_constrains(mf, h, constrains)
    return callback_mf


def _compose_callbacks(a, b):
    """Return a function applying `a` then `b`, skipping whichever of the
    two is None. Used to let an externally-supplied callback_mf (e.g. the
    lab-frame constrains wrapper built by _rotated_axis_exchange) compose
    with the one SzSz derives from its own `constrains` argument, instead
    of one silently overriding the other."""
    if a is None: return b
    if b is None: return a
    def combined(mf):
        return b(a(mf))
    return combined


def SzSz(h, J1=0.0, J2=0.0, J3=0.0, Jr=None, constrains=[], callback_mf=None,
        **kwargs):
    """Self-consistent Hartree-Fock mean field for a
    H = sum J1/J2/J3 (+ Jr(r)) Sz_i Sz_j
    spin-spin interaction (first/second/third neighbor shells, plus an
    optional general distance function Jr). J>0 is antiferromagnetic
    (Heisenberg-like sign convention), J<0 favors a ferromagnetic
    instability along z.

    Works for BdG (Nambu, h.has_eh=True) Hamiltonians too: densitydensity()
    already dispatches has_eh-aware Hartree-Fock+anomalous decoupling
    (selfconsistency.densitydensity.get_mf) generically for any v matrix,
    including SzSz's +/-1/4 one, with no changes needed here -- verified
    that a bare SzSz run (no pre-existing pairing) on a Nambu Hamiltonian
    converges to a purely magnetic state (zero anomalous/pairing mean
    field) that exactly matches the non-Nambu SzSz result in the electron
    sector."""
    from .densitydensity import densitydensity
    if not h.has_spin: return NotImplemented # only for spinful systems
    h = h.get_multicell().get_dense()
    v = _build_v(h, J1, J2, J3, Jr)
    constrain_cb = _callback_mf_constrains(h, constrains)
    callback_mf = _compose_callbacks(constrain_cb, callback_mf)
    return densitydensity(h, v=v, callback_mf=callback_mf, **kwargs)


_AXIS_ROTATION = {
        "x": dict(vector=[0., 1., 0.], angle=0.5),
        "y": dict(vector=[1., 0., 0.], angle=-0.5),
        }


def _rotate_mf_guess(h0, axis, mf, **fwd):
    """Turn a user-supplied `mf=` guess -- given in the lab frame, as a
    string mode name (e.g. "ferroX"), a matrix, a dict, or a Hamiltonian --
    into a plain hopping dict expressed in the internally-rotated frame
    that SzSz(hr,...) actually gets called with. Without this, an
    axis-appropriate guess like mf="ferroX" passed to SxSx would be handed
    unrotated to the internal (rotated) SzSz call, seeding a guess that has
    no overlap with the computational-z order parameter the rotated SCF
    loop actually looks for -- silently killing the intended instability."""
    if mf is None: return None
    from .mfconstrains import obj2mf
    if isinstance(mf, str):
        from ..meanfield import guess
        mf = guess(h0, mode=mf) # resolve the guess in the lab frame first
        if mf is None: return None # e.g. guess() modes that return nothing
    mf = obj2mf(mf) # normalize matrix/Hamiltonian/dict to a hopping dict
    return {d: _gsr(m, **fwd) for (d, m) in mf.items()}


def _rotated_constrains_callback(h0, constrains, fwd, bwd):
    """Build a callback_mf that enforces `constrains` in the LAB frame (as
    the user expects -- e.g. "no_offplane_magnetism" meaning the real z
    axis) even though the mean field iterate SzSz(hr,...) actually mixes
    lives in the internally-rotated frame where `axis` is the computational
    z axis. Passing `constrains` straight through to the inner SzSz call
    would enforce it against the ROTATED frame's z axis instead -- e.g.
    "no_offplane_magnetism" under SxSx would silently constrain the
    physical x component (computational z in that frame), not z."""
    if not constrains: return None
    from . import mfconstrains
    def callback_mf(mf_rot):
        mf_lab = _rotate_dict(mf_rot, **bwd) # mf lives in Hamiltonian convention
        mf_lab = mfconstrains.enforce_constrains(mf_lab, h0, constrains)
        return _rotate_dict(mf_lab, **fwd) # back into the rotated SCF's frame
    return callback_mf


def _rotated_axis_exchange(h, axis, J1, J2, J3, Jr, constrains, **kwargs):
    fwd = _AXIS_ROTATION[axis]
    bwd = dict(fwd); bwd["angle"] = -fwd["angle"]
    h0 = h.get_multicell().get_dense() # keep the original, unrotated reference
    hr = h0.copy()
    hr.global_spin_rotation(**fwd) # rotate so that `axis` becomes computational z
    mf0 = kwargs.pop("mf", None)
    mf_rot = _rotate_mf_guess(h0, axis, mf0, **fwd)
    callback_mf = _rotated_constrains_callback(h0, constrains, fwd, bwd)
    scf = SzSz(hr, J1, J2, J3, Jr, mf=mf_rot, callback_mf=callback_mf, **kwargs)
    if scf.hamiltonian is not None:
        scf.hamiltonian.global_spin_rotation(**bwd) # rotate the result back
    # scf.mf/scf.dm/scf.v are left expressed in the internally-rotated frame;
    # scf.hamiltonian (the user-facing result) and scf.hamiltonian0 are in
    # the original frame
    scf.hamiltonian0 = h0
    return scf


def SxSx(h, J1=0.0, J2=0.0, J3=0.0, Jr=None, constrains=[], **kwargs):
    """Self-consistent Hartree-Fock mean field for a
    H = sum J1/J2/J3 (+ Jr(r)) Sx_i Sx_j
    spin-spin interaction. Implemented by rotating the problem so that x
    becomes the computational z axis, running SzSz there, and rotating the
    converged Hamiltonian back -- see the module docstring. Works for BdG
    (Nambu, h.has_eh=True) Hamiltonians too: global_spin_rotation already
    handles the Nambu case correctly (see its docstring)."""
    if not h.has_spin: return NotImplemented
    return _rotated_axis_exchange(h, "x", J1, J2, J3, Jr, constrains, **kwargs)


def SySy(h, J1=0.0, J2=0.0, J3=0.0, Jr=None, constrains=[], **kwargs):
    """Self-consistent Hartree-Fock mean field for a
    H = sum J1/J2/J3 (+ Jr(r)) Sy_i Sy_j
    spin-spin interaction. Implemented by rotating the problem so that y
    becomes the computational z axis, running SzSz there, and rotating the
    converged Hamiltonian back -- see the module docstring. Works for BdG
    (Nambu, h.has_eh=True) Hamiltonians too: global_spin_rotation already
    handles the Nambu case correctly (see its docstring)."""
    if not h.has_spin: return NotImplemented
    return _rotated_axis_exchange(h, "y", J1, J2, J3, Jr, constrains, **kwargs)


def _rotate_dict(dd, vector, angle):
    """Rotate a dict of Hamiltonian-like (hopping/mean-field) matrices by a
    global spin rotation -- these live in the same convention as
    Hamiltonian.intra, for which R @ m @ R^dagger is the correct
    transformation (as used by Hamiltonian.global_spin_rotation, validated
    by SxSx/SySy)."""
    return {k: _gsr(m, vector=vector, angle=angle) for (k, m) in dd.items()}


def Jinteraction(h0, Jx1=0.0, Jx2=0.0, Jx3=0.0, Jy1=0.0, Jy2=0.0, Jy3=0.0,
        Jz1=0.0, Jz2=0.0, Jz3=0.0, Jxr=None, Jyr=None, Jzr=None,
        mf=None, filling=0.5, mu=None, mix=0.1, nk=8, maxerror=1e-5, maxite=None,
        T=1e-7, verbose=0, constrains=[]):
    """Self-consistent anisotropic exchange mean field,
    H = sum Jx Sx_i Sx_j + Jy Sy_i Sy_j + Jz Sz_i Sz_j,
    combining all three channels in a single SCF loop: at every iteration
    the z channel is decoupled directly (Hartree-Fock density-density in
    the lab/computational spin basis) and the x/y channels are decoupled
    by rotating the density matrix into the frame where that axis is the
    computational z axis, applying the same decoupling there, and rotating
    the resulting mean field back before summing the three contributions
    -- see SxSx/SySy for the single-axis version of the same trick.

    Superseded by VJinteraction (this module) for new code -- VJinteraction
    is a superset (also handles V/U density-density in the same SCF loop,
    isotropic J1/J2/J3 exchange, and is where the Tier 1-3 SCF performance
    work landed) and is the one Hamiltonian.get_mean_field_hamiltonian
    calls by default. Jinteraction is kept as-is mainly for the
    VJinteraction-reduces-to-Jinteraction-with-only-J equivalence tests
    (tests/scf/test_vjinteraction.py) -- don't extend or optimize it
    further, only touch it if a bug is found here specifically.

    Unlike SxSx/SySy, `mf` (a string mode name, matrix, dict or Hamiltonian
    guess) is used directly in the lab frame with no rotation: the mf
    iterate driving this SCF loop always lives in the lab frame -- only the
    per-iteration x/y mean-field *contributions* are computed by a
    temporary excursion into a rotated frame, rotated back before being
    summed in.

    Only integration="ed" and the plain-mixing solver are supported (unlike
    Vinteraction/SzSz/SxSx/SySy, which forward to the full
    generic_densitydensity solver zoo).

    Works for BdG (Nambu, h0.has_eh=True) Hamiltonians too, but decouples
    the exchange interaction in the normal (electron) sector only -- see
    _run_anisotropic_scf's docstring for why (in short: extending the x/y
    rotate-decouple-rotate-back trick to also generate anomalous/pairing
    mean field from Jx/Jy is a separate, unverified extension)."""
    if not h0.has_spin: return NotImplemented # only for spinful systems, same as SzSz/SxSx/SySy
    h1 = h0.get_multicell().get_dense()
    nd = h1.geometry.neighbor_distances() # shared by all three _build_v calls below
    vz = _build_v(h1, Jz1, Jz2, Jz3, Jzr, nd=nd)
    vx = _build_v(h1, Jx1, Jx2, Jx3, Jxr, nd=nd)
    vy = _build_v(h1, Jy1, Jy2, Jy3, Jyr, nd=nd)
    return _run_anisotropic_scf(h1, vx, vy, vz, mf, filling, mu, mix, nk,
            maxerror, maxite, T, verbose, constrains)


def _build_density_v(h, V1=0.0, V2=0.0, V3=0.0, U=0.0, Vr=None, nd=None):
    """Build the spin-orbital density-density interaction matrix -- uniform
    across all four spin blocks (V1/V2/V3 neighbor shells, optional general
    Vr(r) function), plus an onsite U between up/down -- exactly mirroring
    Vinteraction's own construction (selfconsistency/densitydensity.py).
    Kept as a small separate copy here (rather than refactoring Vinteraction
    to share it) to avoid touching that already-tested, widely-used code
    path; see _build_v's docstring for why the neighbor-shell key set this
    produces is independent of which of V1/V2/V3 happen to be zero.

    nd: see _build_v's docstring -- same precomputed-neighbor_distances
    reuse, since VJinteraction calls this alongside three _build_v calls
    for the same h."""
    from .. import specialhopping
    from .densitydensity import obj2geometryarray
    if nd is None: nd = h.geometry.neighbor_distances()
    mgenerator = specialhopping.distance_hopping_matrix(
            [V1/2., V2/2., V3/2.], nd[0:3])
    hv = h.geometry.get_hamiltonian(has_spin=False, is_multicell=True,
            mgenerator=mgenerator)
    if Vr is not None:
        hv1 = h.geometry.get_hamiltonian(has_spin=False, is_multicell=True,
                tij=Vr)
        hv = hv + hv1
    v = hv.get_hopping_dict()
    U = obj2geometryarray(U, h.geometry)
    for d in v:
        m = v[d]
        n = m.shape[0]
        m1 = np.zeros((2*n, 2*n), dtype=np.complex128)
        for i in range(n):
            for j in range(n):
                m1[2*i, 2*j] = m[i, j]
                m1[2*i+1, 2*j] = m[i, j]
                m1[2*i, 2*j+1] = m[i, j]
                m1[2*i+1, 2*j+1] = m[i, j]
        v[d] = m1
    for i in range(n):
        v[(0, 0, 0)][2*i, 2*i+1] += U[i]/2.
        v[(0, 0, 0)][2*i+1, 2*i] += U[i]/2.
    return v


def VJinteraction(h0, V1=0.0, V2=0.0, V3=0.0, U=0.0, Vr=None,
        J1=0.0, J2=0.0, J3=0.0, Jr=None, J1x=0.0, J1y=0.0, J1z=0.0,
        mf=None, filling=0.5, mu=None, mix=0.1, nk=8, maxerror=1e-5, maxite=None,
        T=1e-7, verbose=0, constrains=[],
        integration="ed", scale=None, npol=None, ne=None, cores=None):
    """Self-consistent mean field combining density-density interactions
    (U onsite Hubbard, V1/V2/V3/Vr neighbor-shell -- same convention as
    Vinteraction) with spin-spin exchange in a single SCF loop.

    J1/J2/J3 (+ Jr, a general distance-dependent function) are isotropic
    Heisenberg-like exchange, J*(Sx_i Sx_j + Sy_i Sy_j + Sz_i Sz_j), for the
    first/second/third neighbor shells -- the same first/second/third
    neighbor-shell convention as V1/V2/V3. J1x/J1y/J1z are an additional,
    optional anisotropic correction, added on top of J1 for the
    first-neighbor shell only (e.g. the effective first-neighbor Jz
    coupling is J1+J1z); second/third neighbors stay purely isotropic. All
    default to 0, i.e. plain density-density with no spin-spin exchange.

    This works by combining the two existing SCF modes rather than
    inventing new decoupling math: density-density interactions and
    Sa_i Sa_j are both already density-density interactions in the
    spin-orbital basis (Vinteraction's uniform sign pattern across the
    four spin blocks vs. SzSz's +/-1/4 one -- see the module docstring and
    _build_v), and Hartree-Fock decoupling (get_mf_normal) is linear in the
    interaction matrix, so for a normal-state (non-BdG) Hamiltonian the
    density-density contribution is simply added into the z-channel matrix
    before entering the shared SCF loop -- no separate channel, and no
    rotation, needed for it (unlike the x/y channels, which do need the
    rotate-decouple-rotate-back trick).

    For a BdG (Nambu, h0.has_eh=True) Hamiltonian, density-density and
    exchange are instead kept as two separate contributions summed each
    SCF iteration (see _run_anisotropic_scf's docstring): density-density
    keeps its existing full normal+anomalous treatment (identical to
    Vinteraction), while the exchange channels are decoupled in the normal
    (electron) sector only, i.e. J does not itself induce pairing here.

    See Vinteraction and Jinteraction for further background on the
    density-density and exchange conventions respectively; only the
    plain-mixing solver is supported (unlike Vinteraction/SzSz/SxSx/SySy),
    and integration is restricted to "ed" (default) or "kpm" (no "qtci").

    integration="ed" computes the per-iteration density matrix by exact
    diagonalization -- dense, or restricted to the sparse (direction,row,
    col) entries the mean field actually reads (see _build_sparse_pairs)
    for a normal-state (h0.has_eh=False) Hamiltonian. integration="kpm"
    instead gets those same sparse entries through
    kpmtk.densitymatrix_kpm's per-k Chebyshev-moment (Kernel Polynomial
    Method) engine -- the same one Vinteraction_kpm/densitydensity_kpm.py
    use -- never diagonalizing H(k), and finds the Fermi energy (when
    mu=None) the same diagonalization-free way via
    kpmtk.densitymatrix_kpm.get_fermi4filling_kpm.

    PERFORMANCE CAVEAT (measured 2026-07-27, not just a theoretical
    concern): despite kpmtk.densitymatrix_kpm._dm_kpm_from_needed batching
    its Chebyshev-moment-to-density-matrix reconstruction across every
    needed (row,col) pair at a given k (a ~3x speedup over the original
    per-pair implementation, verified against "ed" to ~1e-7), this backend
    is currently far SLOWER than integration="ed" at small/moderate system
    sizes, not faster -- measured ~50-60x slower per SCF iteration on a
    98-site (196-orbital) honeycomb Hubbard system (nk=4, npol=200): ED
    ~0.05-0.16s/iteration vs KPM ~7-9s/iteration. The reason is that ED's
    per-k diagonalization goes through highly-tuned dense LAPACK routines,
    which are extremely fast for a small-to-moderate matrix regardless of
    algorithmic complexity, while this KPM implementation still pays real
    per-element and per-orbital overhead that ED simply doesn't have: a
    separate Chebyshev VECTOR recursion per needed (row,col,k) triple
    (kpm_moments_vivj -- not yet batched across pairs sharing a starting
    vector/column, unlike the profile-reconstruction step, which is), and
    get_fermi4filling_kpm's own O(n_orb) separate per-orbital moment
    calculation (a deterministic trace, not a stochastic few-random-vector
    estimate) when mu=None, which was not touched by this round of fixes
    and is now a comparable-or-larger fraction of the per-iteration cost
    than the density-matrix step. Neither of those remaining costs scales
    down with system size the way ED's O(n^3) diagonalization eventually
    becomes the bottleneck at -- so while this backend should in principle
    win for a large enough (or sparse enough) system, that crossover was
    not reached in the sizes tested here (up to ~200 sites), and further
    batching work (grouping needed pairs by shared starting column,
    stochastic-trace Fermi search) would very likely be needed first to
    reach it in practice. Use integration="ed" unless you have confirmed
    "kpm" is actually faster for your specific system size/sparsity.
    Only supported for a normal-state Hamiltonian, for the
    same reason the sparse ED path is restricted that way (see
    _run_anisotropic_scf's docstring): a BdG/Nambu VJinteraction call
    keeps `vd` in a differently-indexed (Nambu-reordered) basis that the
    sparse-position machinery this KPM path reuses does not (yet) know how
    to translate -- passing integration="kpm" for a Nambu h0 raises
    NotImplementedError rather than silently computing the wrong thing.
    scale/npol/ne/cores are the same KPM tuning knobs as
    kpmtk.densitymatrix_kpm.get_dm_kpm (scale: KPM energy rescaling,
    estimated automatically when None; npol: number of Chebyshev moments,
    defaulting to kpmtk.densitymatrix_kpm.DEFAULT_NPOL when None; ne:
    number of energies sampled in the occupied window; cores: number of
    parallel workers across k-points); all four are unused when
    integration="ed".

    NOTE: unlike the "ed" path, scf.dm after convergence under
    integration="kpm" only holds the sparse subset of entries the SCF loop
    itself needed (see _build_sparse_pairs), not a complete dense density
    matrix -- recomputing a fully dense one via exact diagonalization after
    convergence would defeat the point of avoiding diagonalization in the
    first place, so (mirroring Vinteraction_kpm/densitydensity_kpm.py,
    whose own scf.dm has the same property) this path skips that step.

    integration="kpm" also never forces h0 into a dense representation: the
    other integration modes (and Jinteraction) unconditionally call
    .get_dense() here, which is fine for them (they end up diagonalizing
    dense matrices anyway), but would defeat the point of KPM for a large,
    genuinely sparse h0 (h0.is_sparse=True, e.g. a big 0D flake/island
    where the *unit cell itself* -- h.intra -- is too large to hold as a
    dense array). If h0.is_sparse, h1 (hence every per-iteration
    Hamiltonian derived from it inside _run_anisotropic_scf) stays sparse
    for the whole SCF loop -- see that function's docstring for how the
    mean-field update step, which would otherwise silently densify it
    (scipy sparse + dense ndarray returns a dense matrix), is kept from
    doing so.

    The total-energy computation after the SCF loop converges also never
    diagonalizes: it used to call h.get_total_energy(nk=h.nk)
    unconditionally (spectrum.total_energy's own `if nbands is None: h =
    h.get_dense()` densifies and diagonalizes regardless of h.is_sparse
    unless `nbands` is explicitly given, which that call never did --
    forcing a dense diagonalization right after an otherwise
    sparsity-preserving SCF loop, for a huge sparse h0 exactly the
    scenario this mode exists for). integration="kpm" instead calls
    kpmtk.densitymatrix_kpm.get_total_energy_kpm, which integrates the
    same KPM-reconstructed density of states get_fermi4filling_kpm already
    uses up to the Fermi energy instead of diagonalizing -- see that
    function's docstring (verified against exact diagonalization to
    ~0.1% relative on a frozen test Hamiltonian). Vinteraction_kpm/
    densitydensity_kpm.py's own total-energy tail still has the
    unconditional-densification version of this (out of scope for this
    pass, since it is a separate implementation this work did not
    touch)."""
    if not h0.has_spin: return NotImplemented # only for spinful systems, same as SzSz/SxSx/SySy
    h1 = h0.get_multicell()
    if integration != "kpm": h1 = h1.get_dense() # see docstring above
    nd = h1.geometry.neighbor_distances() # shared by all four _build_*_v calls below
    vz = _build_v(h1, J1+J1z, J2, J3, Jr, nd=nd)
    vd = _build_density_v(h1, V1, V2, V3, U, Vr, nd=nd)
    vx = _build_v(h1, J1+J1x, J2, J3, Jr, nd=nd)
    vy = _build_v(h1, J1+J1y, J2, J3, Jr, nd=nd)
    if not h1.has_eh: # normal-state: fold density-density directly into vz
        vz = (MultiHopping(vz) + MultiHopping(vd)).get_dict()
        vd = None
    return _run_anisotropic_scf(h1, vx, vy, vz, mf, filling, mu, mix, nk,
            maxerror, maxite, T, verbose, constrains, vd=vd,
            integration=integration, scale=scale, npol=npol, ne=ne,
            cores=cores)


def _channel_is_zero(v):
    """True if every matrix in this interaction channel's hopping dict is
    identically zero -- i.e. the J (or J1x/J1y) that built it via _build_v
    was 0. Used to skip an exchange channel's rotation and
    get_mf_normal/get_dc_energy calls entirely in _run_anisotropic_scf,
    since a zero interaction contributes exactly zero mean field regardless
    of the density matrix -- this is the difference between VJinteraction's
    pure density-density case (J1=J2=J3=J1x=J1y=0) and Vinteraction: without
    this check, VJinteraction always pays for three full exchange-channel
    passes (z, x, y) even when x and y are pure zero matrices."""
    return all(not np.any(m) for m in v.values())


def _build_sparse_pairs(vlist, keys, n):
    """For each direction, the union of (row,col) index pairs actually read
    downstream from the lab-frame density matrix, across the given
    interaction matrices (whichever of vz/vx/vy/vd are not None):

    - get_mf_normal's density-density (compute_dd) term only ever reads
      dm[(0,0,0)]'s DIAGONAL, for whichever sites participate in any bond
      of any channel -- in practice that is normally every site, so this
      always includes the full diagonal at (0,0,0) unconditionally, rather
      than trying to track which subset that is. Crucially this means the
      full 2x2 SPIN BLOCK at each site, not just its two purely-diagonal
      (up-up, down-down) entries: whenever vx/vy is active, _rot_dm rotates
      dme_lab[(0,0,0)] before compute_dd reads its (rotated) diagonal, and
      a spin rotation mixes a 2x2 block's diagonal and off-diagonal entries
      together (rot @ [[a,b],[c,d]] @ rot^dagger's [0,0]/[1,1] entries
      depend on b,c too) -- so leaving a site's up-down/down-up entries at
      zero when no channel happens to touch them silently corrupts the
      rotated diagonal too, not just the (unused) unrotated off-diagonal.
      Caught by test_jinteraction_random_direction_guess_gives_collinear_moment
      (isotropic exchange collapsing onto the z axis instead of the guess
      direction -- the x/y mean-field contribution was being silently
      zeroed by this).
    - get_mf_normal's cross term (compute_cross) reads dm[d2][j,i] (note
      the swapped indices) for every (i,j) where v[d][i,j] is nonzero, i.e.
      it needs the TRANSPOSE of v[d]'s nonzero pattern at the OPPOSITE
      direction d2=-d, not v[d2]'s own pattern -- so each v[d] contributes
      to two masks: its own pattern at d (used directly by get_dc_energy
      and as the rotation input for vx/vy), and its transpose at d2.
    - _rot_dm/_rot_dict (used for vx/vy) rotate whichever 2x2 spin
      sub-blocks are present as a unit (R is block-diagonal in the 2x2
      spin index, see build_rotation_matrix), so correctness requires
      each touched site-pair's full 2x2 block, not individual entries.
      _build_v/_build_density_v's own construction always populates a
      site-pair's 4 spin sub-entries together (or, for the onsite-U cross
      term specifically, only the 2 off-diagonal ones -- but those live at
      a diagonal site-pair (i,i), whose other 2 entries are already forced
      in by the always-include-the-diagonal rule above), so a plain
      per-entry union already comes out block-complete with no special
      handling needed here.

    A short-range neighbor-shell v is overwhelmingly zero -- measured on a
    98-site/196-orbital system, 0.02%-0.34% nonzero per off-diagonal
    direction key -- so the result is normally a tiny fraction of the full
    n^2 grid; see dmtk.fulldm.full_dm_batch_d_sparse and
    densitymatrix.full_dm_accumulate_sparse for what computing only these
    entries buys over the dense (n,n)@(n,n) per-direction matmul."""
    all_dirs = set(keys)
    for v in vlist:
        if v is None: continue
        for d in v:
            all_dirs.add(d)
            all_dirs.add((-d[0], -d[1], -d[2]))
    masks = {d: np.zeros((n, n), dtype=bool) for d in all_dirs}
    for v in vlist:
        if v is None: continue
        for d, m in v.items():
            nz = (m != 0)
            masks[d] |= nz               # own-direction uses (get_dc_energy, rotation input)
            d2 = (-d[0], -d[1], -d[2])
            masks[d2] |= nz.T            # cross-term uses dm[d2][j,i] for v[d][i,j] != 0
    masks[(0, 0, 0)] |= np.kron(np.eye(n // 2, dtype=bool), np.ones((2, 2), dtype=bool))
    pairs = dict()
    for d, mask in masks.items():
        rows, cols = np.nonzero(mask)
        pairs[d] = (rows.astype(np.int64), cols.astype(np.int64))
    return pairs


def _sparse_pairs_to_needed(pairs):
    """Convert _build_sparse_pairs' {direction: (rows, cols)} format into
    the (direction, row, col) triple set
    kpmtk.densitymatrix_kpm._dm_kpm_from_needed expects -- the exact same
    needed positions the ED sparse path (densitymatrix.full_dm_accumulate_
    sparse) already reads, just handed to the KPM engine instead of the
    diagonalization-based one (see _run_anisotropic_scf's integration="kpm"
    branch)."""
    needed = set()
    for d, (rows, cols) in pairs.items():
        d = tuple(d)
        for i, j in zip(rows.tolist(), cols.tolist()):
            needed.add((d, i, j))
    return needed


def _run_anisotropic_scf(h1, vx, vy, vz, mf, filling, mu, mix, nk,
        maxerror, maxite, T, verbose, constrains, vd=None,
        integration="ed", scale=None, npol=None, ne=None, cores=None):
    """Shared SCF core for Jinteraction/VJinteraction: decouples the
    z-channel matrix `vz` directly (Hartree-Fock density-density in the
    lab/computational spin basis) and the x/y-channel matrices `vx`/`vy`
    by rotating the density matrix into the frame where that axis is the
    computational z axis, applying the same decoupling there, and rotating
    the resulting mean field back before summing all three contributions
    -- see Jinteraction's docstring for the physics. `h1` must already be
    h0.get_multicell().get_dense().

    `vd`, if given, is an additional density-density interaction matrix
    (Vinteraction's convention) added to the mean field each iteration.
    For a normal-state h1, the caller should have already folded this into
    `vz` directly instead (Hartree-Fock decoupling is linear in the
    interaction, so this is equivalent and does not need a separate
    channel) and passed vd=None. `vd` as a genuinely separate argument only
    matters for a BdG (Nambu, h1.has_eh=True) h1: there, vx/vy/vz (the
    exchange channels) are decoupled in the normal (electron) sector only
    -- extracting it from the full Nambu density matrix, decoupling with
    get_mf_normal exactly as for a normal-state Hamiltonian (verified: the
    x/y-rotation trick's _rot_dm/_rot_dict logic, and
    rotate_spin.global_spin_rotation more generally, both already handle
    Nambu-doubled matrices correctly with no changes, since pyqula's Nambu
    convention (sctk/reorder.py) groups each site's electron pair and hole
    pair as separate, identically-transforming (up,down)-like 2-blocks),
    then embedded back into a full Nambu matrix with zero anomalous part --
    while `vd` gets the full has_eh-aware treatment (get_mf, both normal
    and anomalous/pairing), identical to how Vinteraction already handles
    it. In short: J does not itself induce superconducting pairing here,
    only V/U can (matching the existing Zeeman+attractive-V1 triplet-SC
    machinery) -- extending the x/y rotation trick to also rotate the
    anomalous sector is a separate, unverified piece of physics left for a
    future extension.

    vx/vy are skipped entirely (no rotation, no get_mf_normal/get_dc_energy
    call) when they are identically zero -- e.g. VJinteraction's pure
    density-density case (J1=J2=J3=J1x=J1y=0), where _build_v returns an
    all-zero matrix for every key regardless of geometry. vz and vd (when
    given) get the same treatment: vz can be identically zero for a Nambu
    VJinteraction call with only Jx/Jy set (J1=J2=J3=J1z=0), and vd for one
    with only J's and no V/U. A zero interaction contributes exactly zero
    mean field either way, so this changes no result, only the cost of
    computing it -- see _channel_is_zero.

    integration="kpm" (see VJinteraction's docstring) reuses exactly the
    same `sparse_pairs` positions the "ed" sparse path computes below
    (_build_sparse_pairs), just evaluating them via
    kpmtk.densitymatrix_kpm._dm_kpm_from_needed's per-k Chebyshev-moment
    engine instead of full_dm_accumulate_sparse's diagonalization, and
    finding the Fermi energy (mu=None) via
    kpmtk.densitymatrix_kpm.get_fermi4filling_kpm instead of
    Hamiltonian.get_fermi4filling -- so it is only available when
    use_sparse_dm is (i.e. has_eh=False); requesting it for a Nambu h1
    raises NotImplementedError rather than silently falling back to ED or
    misreading vd's differently-indexed Nambu basis."""
    from .densitydensity import (get_dm, get_mf_normal, get_mf, mix_mf,
            diff_mf, update_hamiltonian, set_hoppings, hamiltonian2dict,
            get_dc_energy, SCF, random_hermitian_guess)
    from .mfconstrains import obj2mf
    has_eh = h1.has_eh
    if has_eh: from .. import superconductivity
    if integration not in ("ed", "kpm"):
        raise ValueError("integration must be 'ed' or 'kpm', got %r" % (integration,))
    if integration == "kpm" and has_eh:
        raise NotImplementedError("VJinteraction's integration=\"kpm\" path "
                "only supports a normal-state (has_eh=False) Hamiltonian -- "
                "see _run_anisotropic_scf's docstring for why the Nambu "
                "case (vd in its own reordered basis) is out of scope here")
    h1.nk = nk
    # union of the three exchange channels' bond directions (+ vd's, if
    # given): in general the neighbor-shell hopping-dict builder could
    # prune a channel's key set differently depending on which of its J's
    # (or V's) are zero, so the lab-frame density matrix must be requested
    # at the union, not just vz's keys
    v_dirs = {d: None for d in (set(vz) | set(vx) | set(vy) |
            (set(vd) if vd is not None else set()))}
    # the x/y rotations are fixed for the whole SCF loop, so build the small
    # 2x2 spin rotation matrices once via build_rotation_matrix instead of
    # paying a fresh matrix exponential on every one of the many
    # _rotate_dict/_rotate_dm calls compute_mf makes each iteration; the
    # backward rotation is just the forward matrix's dagger (R(-angle) =
    # R(angle)^dagger), so only Rx/Ry need to be built.
    # build_rotation_matrix(1,...) returns exactly the small 2x2 spin
    # rotation with no reshaping needed (kron with a 1x1 identity is a
    # no-op) -- the full (2n_orb)x(2n_orb) kron'd matrix this used to build
    # is never actually needed: see _block_rotate for why applying it is an
    # O(n_orb^2) per-site contraction with this small matrix, not an
    # O(n_orb^3) dense matmul against the big one.
    vx_active = not _channel_is_zero(vx)
    vy_active = not _channel_is_zero(vy)
    vz_active = not _channel_is_zero(vz)
    vd_active = vd is not None and not _channel_is_zero(vd)
    Rx = Ry = Rxd = Ryd = None
    if vx_active or vy_active:
        from ..rotate_spin import build_rotation_matrix
        # daggers precomputed once here (rather than inside _rot_dict/_rot_dm,
        # which previously recomputed R.conj().T -- once for the forward
        # rotation, again explicitly at each of that channel's two call
        # sites in compute_mf/the total-energy tail -- on every one of the
        # maxite SCF iterations for a rotation that never changes)
        if vx_active:
            Rx = build_rotation_matrix(1, **_AXIS_ROTATION["x"]); Rxd = Rx.conj().T
        if vy_active:
            Ry = build_rotation_matrix(1, **_AXIS_ROTATION["y"]); Ryd = Ry.conj().T

    # sparse density-matrix path: only compute the (row,col) entries that
    # normal_term_ii/jj/ij/ji actually read, instead of the full (n,n)
    # matrix per direction (see _build_sparse_pairs/full_dm_accumulate_sparse).
    # has_eh=False only: vd's Nambu-basis interaction matrix lives in the
    # reordered Nambu convention (sctk/reorder.py), not the plain
    # spin-orbital one _build_sparse_pairs assumes, so the Nambu case keeps
    # the existing dense get_dm call below rather than risk misreading that
    # reordering here -- a possible follow-up, not attempted in this pass.
    use_sparse_dm = not has_eh
    if use_sparse_dm:
        n_dm = vz[(0, 0, 0)].shape[0]
        sparse_pairs = _build_sparse_pairs(
                [vz, vx, vy] + ([vd] if vd is not None else []), v_dirs, n_dm)

    def _get_dm(h):
        if use_sparse_dm:
            from ..densitymatrix import full_dm_accumulate_sparse
            delta = T if T != 0. else 1e-15 # see densitymatrix.full_dm's own T==0 guard
            return full_dm_accumulate_sparse(h, sparse_pairs, nk=nk, delta=delta)
        return get_dm(h, v_dirs, nk=nk, T=T, integration="ed")

    # KPM density-matrix path (integration="kpm", only reached when
    # use_sparse_dm is True -- see the NotImplementedError check above):
    # same `sparse_pairs` positions as the ED sparse path, evaluated
    # through kpmtk.densitymatrix_kpm's per-k Chebyshev-moment engine
    # instead of diagonalizing H(k) -- see VJinteraction's docstring.
    use_kpm = integration == "kpm"
    # h1 keeps whatever is_sparse it was built with when integration="kpm"
    # (see VJinteraction's docstring) -- keep_sparse gates the per-iteration
    # re-sparsification below that stops the mean-field update from
    # silently densifying it again.
    keep_sparse = use_kpm and h1.is_sparse
    if use_kpm:
        from ..kpmtk.densitymatrix_kpm import (DEFAULT_NPOL,
                get_fermi4filling_kpm, _dm_kpm_from_needed, get_total_energy_kpm)
        if npol is None: npol = DEFAULT_NPOL # same default as get_dm_kpm
        kpm_needed = _sparse_pairs_to_needed(sparse_pairs)

        def _get_dm_kpm(h):
            # scale=scale (the outer, possibly-None user override) is
            # deliberately re-estimated independently here rather than
            # shared with get_fermi4filling_kpm's own estimate below, even
            # though both call the identical _estimate_kpm_scale formula:
            # when mu=None, get_fermi4filling_kpm's estimate is necessarily
            # taken on h BEFORE the Fermi shift (the shift amount isn't
            # known until it returns), while this h is the POST-shift one
            # -- and shifting can move the bandwidth estimate by a
            # near-2x factor in practice (measured: 2.31 -> 4.09 on a
            # simple 1-site/spin chain), enough to push the pre-shift
            # scale below the post-shift spectrum's actual extent. Reusing
            # the pre-shift estimate here was tried and caused the
            # Chebyshev recursion to diverge to NaN within one SCF
            # iteration (rescaled H(k) then has eigenvalues outside
            # [-1,1]) -- a real, confirmed regression, not a theoretical
            # one, so this is intentionally NOT deduplicated with
            # get_fermi4filling_kpm's call despite the shared formula.
            dm = _dm_kpm_from_needed(h, kpm_needed, nk=nk, scale=scale,
                    npol=npol, ne=ne, cores=cores, T=T)
            for d in v_dirs: # every requested direction must have a key,
                if d not in dm: # even one contributing no needed entries
                    dm[d] = np.zeros((n_dm, n_dm), dtype=np.complex128)
            return dm

    def _block_rotate(m, rot):
        """rot @ m @ rot^dagger, applied to every site's own 2x2 spin
        sub-block of an (n,n) matrix independently, instead of a dense
        (n,n)@(n,n) matmul against the full R=kron(I_{n/2},rot) a global
        spin rotation used to be built as: R is block-diagonal with n/2
        IDENTICAL 2x2 blocks (rotate_spin.build_rotation_matrix), so it
        never mixes different sites -- only the 2 spin components within
        each one. Two small (n/2 * 4)-cost einsum contractions reproduce
        the same result as R @ m @ R^dagger at O(n^2) instead of O(n^3)."""
        n = m.shape[0]
        n_orb = n//2
        m4 = m.reshape(n_orb, 2, n_orb, 2)
        out = np.einsum('ab,xbyc->xayc', rot, m4, optimize=True)
        out = np.einsum('xayc,dc->xayd', out, rot.conj(), optimize=True)
        return out.reshape(n, n)

    def _rot_dict(dd, R):
        """Rotate a dict of Hamiltonian-like (hopping/mean-field) matrices:
        these live in the same convention as Hamiltonian.intra, for which
        R @ m @ R^dagger is the correct transformation (as used by
        Hamiltonian.global_spin_rotation, validated by SxSx/SySy)."""
        return {k: _block_rotate(m, R) for (k, m) in dd.items()}

    def _rot_dm(dd, R):
        """Rotate a dict of *density matrices*, which need a different
        (conjugate-sandwiched) transformation than _rot_dict.

        get_density_matrix's off-diagonal (spin-flip) entries are stored in
        a transposed convention relative to a Hamiltonian matrix --
        normal_term_ij (selfconsistency/densitydensity.py) deliberately
        reads dm[j,i] rather than dm[i,j] to reconstruct a
        physically-meaningful mean field from it. For a Hermitian matrix,
        transposing is the same as complex-conjugating, so a density matrix
        in this convention is the complex conjugate of the
        "Hamiltonian-convention" matrix at the same site/bond. Naively
        rotating it with R @ m @ R^dagger (_rot_dict) therefore silently
        flips the sign of the imaginary (y) Pauli component while leaving
        the real (x, z) ones untouched -- caught by checking that a
        random-direction initial guess converges to a moment collinear with
        it (only Jinteraction is affected: SzSz/SxSx/SySy rotate the whole
        Hamiltonian and run a native SCF in the rotated frame, never
        touching a raw density matrix directly, so they never hit this).
        Conjugating before and after the standard rotation corrects for it:
        if dm_stored = conj(dm_physical), then
        dm_stored' = conj(dm_physical') = conj(R @ conj(dm_stored) @ R^dagger)."""
        return {k: np.conj(_block_rotate(np.conj(m), R)) for (k, m) in dd.items()}

    callback_mf = _callback_mf_constrains(h1, constrains)

    def callback_h(hh):
        if mu is None:
            fermi = hh.get_fermi4filling(filling, nk=hh.nk)
            hh.fermi = fermi
            hh.shift_fermi(-fermi)
        else:
            hh.shift_fermi(-mu)
        return hh

    hop0 = hamiltonian2dict(h1)
    h0_ref = h1.copy() # reference Hamiltonian before the mean field is added
                        # (sparse-backed when keep_sparse is, like h1 itself)

    def electron_sector(dd):
        """Extract the normal (electron-electron) sector from a dict of
        (possibly Nambu-sized) density matrices; a no-op for normal-state
        h1."""
        if not has_eh: return dd
        return {k: superconductivity.get_eh_sector(m, i=0, j=0)
                for (k, m) in dd.items()}

    def embed_normal(mfe):
        """Embed an electron-sector-only mean field dict back into full
        Nambu form, with zero anomalous (pairing) part; a no-op for
        normal-state h1."""
        if not has_eh: return mfe
        return {k: superconductivity.build_nambu_matrix(m)
                for (k, m) in mfe.items()}

    def compute_mf(dm_lab):
        dme_lab = electron_sector(dm_lab) # exchange channels: normal sector only
        if vz_active:
            mf = get_mf_normal(vz, dme_lab)
        else: # vz identically zero -- skip the O(n^2) pass, same reasoning
              # as vx_active/vy_active (see _channel_is_zero)
            zero = dme_lab[(0, 0, 0)]*0.0
            mf = {d: zero.copy() for d in vz}
        if vx_active:
            dm_x = _rot_dm(dme_lab, Rx) # dm needs the conjugated rotation
            mf_x = _rot_dict(get_mf_normal(vx, dm_x), Rxd) # mf does not
            mf = (MultiHopping(mf) + MultiHopping(mf_x)).get_dict()
        if vy_active:
            dm_y = _rot_dm(dme_lab, Ry)
            mf_y = _rot_dict(get_mf_normal(vy, dm_y), Ryd)
            mf = (MultiHopping(mf) + MultiHopping(mf_y)).get_dict()
        mf = embed_normal(mf)
        if vd_active: # density-density: full normal+anomalous treatment
            mf_d = get_mf(vd, dm_lab, has_eh=has_eh)
            mf = (MultiHopping(mf) + MultiHopping(mf_d)).get_dict()
        return mf

    def f(mf):
        h = h1.copy()
        hop = update_hamiltonian(hop0, mf)
        if keep_sparse:
            # update_hamiltonian adds the (always dense, see get_mf_normal)
            # mean field mf on top of h1's own (sparse) hoppings; scipy's
            # sparse+dense addition returns a dense numpy.matrix, which
            # set_hoppings would otherwise install straight into h.intra/
            # h.hopping[i].m -- silently densifying h despite h.is_sparse
            # staying (now incorrectly) True, since set_dictionary never
            # touches that flag. Re-sparsifying here keeps h genuinely
            # sparse-backed for every downstream KPM call on it.
            from scipy.sparse import csr_matrix
            hop = {d: csr_matrix(m) for d, m in hop.items()}
        set_hoppings(h, hop)
        if use_kpm:
            # never diagonalize H(k): the Fermi energy (mu=None) comes from
            # get_fermi4filling_kpm's Chebyshev-moment DOS integral, and the
            # density matrix from _get_dm_kpm's per-k Chebyshev recursion --
            # see VJinteraction's docstring. The two calls do NOT share a
            # scale estimate even though scale=None makes both compute one
            # via the identical _estimate_kpm_scale formula -- see
            # _get_dm_kpm's docstring for why that's intentional (tried,
            # caused a confirmed NaN divergence).
            if mu is None:
                fermi = get_fermi4filling_kpm(h, filling, nk=nk, scale=scale,
                        npol=npol, ne=ne, cores=cores)
                h.fermi = fermi
                h.shift_fermi(-fermi)
            else:
                h.shift_fermi(-mu)
            dm_lab = _get_dm_kpm(h)
        elif use_sparse_dm and mu is None:
            # combined: diagonalize the unshifted h once, deriving both the
            # Fermi energy and the density matrix from the same
            # eigenvectors, instead of callback_h's get_fermi4filling
            # paying for an independent diagonalization sweep first -- see
            # densitymatrix.full_dm_accumulate_sparse_with_fermi's docstring
            from ..densitymatrix import full_dm_accumulate_sparse_with_fermi
            delta = T if T != 0. else 1e-15 # see densitymatrix.full_dm's own T==0 guard
            dm_lab, fermi = full_dm_accumulate_sparse_with_fermi(
                    h, sparse_pairs, filling, nk=nk, delta=delta)
            h.fermi = fermi
            h.shift_fermi(-fermi) # cheap (adds a constant to the diagonal);
                                   # the eigenvectors above are unaffected by it
        else:
            h = callback_h(h)
            dm_lab = _get_dm(h)
        mfnew = compute_mf(dm_lab)
        if callback_mf is not None: mfnew = callback_mf(mfnew)
        scf = SCF()
        scf.hamiltonian = h
        # z-channel matrix only (for a normal-state VJinteraction this
        # already includes the folded-in density-density vd, but never the
        # x/y channels, and never vd separately for a Nambu VJinteraction --
        # unlike Vinteraction/SzSz/SxSx/SySy, there is no single matrix
        # that fully captures this multi-channel interaction, so callers
        # relying on h.V for e.g. get_magnon_bands/get_rpa_kernel_poles
        # (which expect a single, onsite-only interaction matrix) should
        # not assume this represents the whole exchange+density-density mix
        scf.hamiltonian.V = vz
        scf.hamiltonian0 = h0_ref
        scf.mf = mfnew
        scf.dm = dm_lab
        scf.v = vz # see scf.hamiltonian.V above for what this does/doesn't capture
        scf.tol = maxerror
        return scf

    if mf is None:
        # seed over v_dirs (the vz/vx/vy key union), not just vz's own keys:
        # a channel with only its own neighbor shells (e.g. Jx1 nonzero,
        # Jz1=Jz2=Jz3=0) can have bond-direction keys absent from vz, and
        # diff_mf below only iterates the keys already present in this
        # initial guess, so seeding too few of them would make those
        # channels invisible to the very first convergence check.
        #
        # random_hermitian_guess (densitydensity.py) is reused here rather
        # than a separate inline copy so a future fix to this
        # safety-critical invariant (see its own docstring for why: a
        # non-Hermitian guess is harmless for "ed" but blows up
        # integration="kpm"'s Chebyshev recursion) automatically applies
        # to both -- this call site's own scale=1e-1 predates that helper
        # existing at all, kept as-is rather than switched to the helper's
        # own 1.0 default.
        mf = random_hermitian_guess(v_dirs, h1.intra.shape, scale=1e-1)
    else:
        if isinstance(mf, str):
            from ..meanfield import guess
            mf = guess(h0_ref, mode=mf)
        mf = obj2mf(mf)

    ite = 0
    while True:
        scf = f(mf)
        mfnew = scf.mf
        diff = diff_mf(mfnew, mf)
        mf = mix_mf(mfnew, mf, mix=mix)
        if callback_mf is not None: mf = callback_mf(mf)
        if verbose > 0: print("ERROR in the SCF cycle", ite, diff)
        if diff < maxerror:
            scf = f(mfnew) # last iteration, with the unmixed mean field
            scf.converged = True
            break
        if maxite is not None and ite >= maxite:
            scf.converged = False
            print("No convergence has been reached in", maxite, "iterations, stopping")
            break
        ite += 1

    if use_sparse_dm and not use_kpm:
        # scf.dm is a public field (Vinteraction/SzSz/SxSx/SySy's SCF
        # objects always expose a fully dense one, via densitydensity.py's
        # own get_dm), but the sparse path above only ever populated the
        # (row,col) entries the SCF loop itself needed (see
        # _build_sparse_pairs/full_dm_accumulate_sparse) -- leaving the
        # rest silently at zero would corrupt any external use of scf.dm
        # beyond the mean field itself (custom correlators, occupation
        # diagnostics, symmetry checks). Recompute it fully dense exactly
        # once here, for the converged (or not-converged-but-returned)
        # Hamiltonian only -- not once per SCF iteration -- reusing the
        # same dense get_dm the has_eh=True path below already relies on,
        # so this is one extra diagonalization for the whole call, not a
        # per-iteration cost.
        #
        # Skipped for integration="kpm": diagonalizing here would defeat
        # the point of avoiding diagonalization in the first place (the
        # whole reason to pick KPM), so scf.dm stays only the sparse subset
        # the SCF loop needed -- see VJinteraction's docstring, and
        # Vinteraction_kpm/densitydensity_kpm.py's scf.dm, which has the
        # same limitation for the same reason.
        scf.dm = get_dm(scf.hamiltonian, v_dirs, nk=nk, T=T, integration="ed")

    # total energy: sum of occupied energies plus the double-counting
    # correction for each of the three (independently-rotated) exchange
    # channels, plus vd's (if any) -- all electron-sector only. get_dc_energy
    # assumes dm's shape matches v's (2n, never Nambu-doubled), so it must
    # always be fed the extracted electron sector for a BdG h1, never the
    # full Nambu-sized scf.dm directly (verified: passing the full dm here,
    # matching how Vinteraction/densitydensity.py's own total-energy
    # computation does it for its single v, makes the reported total_energy
    # inconsistent between a primitive cell and a supercell of the same
    # system for a Nambu Hamiltonian -- a real, pre-existing bug in that
    # shared code, out of scope to fix here, but not one to reproduce for
    # vd just because it happens to match precedent)
    h = scf.hamiltonian
    if use_kpm:
        # never diagonalize H(k), even for this final, once-per-call step:
        # h is already shifted to its own fermi=0 (see f()'s use_kpm
        # branch), so this integrates the KPM-reconstructed DOS up to 0
        # instead of calling spectrum.total_energy, whose own nbands=None
        # default forces a dense diagonalization regardless of use_kpm --
        # see get_total_energy_kpm's docstring
        etot = get_total_energy_kpm(h, fermi=0.0, nk=nk, scale=scale,
                npol=npol, ne=ne, cores=cores)
    else:
        etot = h.get_total_energy(nk=h.nk)
    if mu is None:
        etot += h.fermi*h.intra.shape[0]*filling
    dme = electron_sector(scf.dm)
    if vz_active:
        etot += get_dc_energy(vz, dme)
    if vx_active:
        dm_x = _rot_dm(dme, Rx) # dm needs the conjugated rotation, see compute_mf
        etot += get_dc_energy(vx, dm_x)
    if vy_active:
        dm_y = _rot_dm(dme, Ry)
        etot += get_dc_energy(vy, dm_y)
    if vd_active:
        etot += get_dc_energy(vd, dme)
    scf.total_energy = etot.real
    return scf
