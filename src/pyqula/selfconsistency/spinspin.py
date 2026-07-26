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


def _build_v(h, J1=0.0, J2=0.0, J3=0.0, Jr=None):
    """Build the spin-orbital interaction matrix for a J1/J2/J3 (plus
    optional general Jr(r) function) neighbor-shell SzSz coupling,
    following exactly the same neighbor-shell/hopping-dict construction as
    Vinteraction, but with the +/-1/4 sign pattern of
    Sz_i Sz_j = 1/4 (n_iu-n_id)(n_ju-n_jd) in the four spin blocks instead
    of Vinteraction's uniform value. Same key set (bond directions) as
    Vinteraction's v for the same geometry, since that is fixed purely by
    the geometry's neighbor shells, independent of which J's are zero."""
    nd = h.geometry.neighbor_distances() # distances to the neighbor shells
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
    instability along z."""
    from .densitydensity import densitydensity
    if not h.has_spin: return NotImplemented # only for spinful systems
    if h.has_eh: return NotImplemented # not implemented for BdG Hamiltonians
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
    converged Hamiltonian back -- see the module docstring."""
    if not h.has_spin: return NotImplemented
    if h.has_eh: return NotImplemented
    return _rotated_axis_exchange(h, "x", J1, J2, J3, Jr, constrains, **kwargs)


def SySy(h, J1=0.0, J2=0.0, J3=0.0, Jr=None, constrains=[], **kwargs):
    """Self-consistent Hartree-Fock mean field for a
    H = sum J1/J2/J3 (+ Jr(r)) Sy_i Sy_j
    spin-spin interaction. Implemented by rotating the problem so that y
    becomes the computational z axis, running SzSz there, and rotating the
    converged Hamiltonian back -- see the module docstring."""
    if not h.has_spin: return NotImplemented
    if h.has_eh: return NotImplemented
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

    Unlike SxSx/SySy, `mf` (a string mode name, matrix, dict or Hamiltonian
    guess) is used directly in the lab frame with no rotation: the mf
    iterate driving this SCF loop always lives in the lab frame -- only the
    per-iteration x/y mean-field *contributions* are computed by a
    temporary excursion into a rotated frame, rotated back before being
    summed in.

    Only integration="ed" and the plain-mixing solver are supported (unlike
    Vinteraction/SzSz/SxSx/SySy, which forward to the full
    generic_densitydensity solver zoo)."""
    if not h0.has_spin: raise ValueError("Jinteraction needs a spinful Hamiltonian")
    if h0.has_eh: raise ValueError("Jinteraction is not implemented for BdG Hamiltonians")
    h1 = h0.get_multicell().get_dense()
    vz = _build_v(h1, Jz1, Jz2, Jz3, Jzr)
    vx = _build_v(h1, Jx1, Jx2, Jx3, Jxr)
    vy = _build_v(h1, Jy1, Jy2, Jy3, Jyr)
    return _run_anisotropic_scf(h1, vx, vy, vz, mf, filling, mu, mix, nk,
            maxerror, maxite, T, verbose, constrains)


def _build_density_v(h, V1=0.0, V2=0.0, V3=0.0, U=0.0, Vr=None):
    """Build the spin-orbital density-density interaction matrix -- uniform
    across all four spin blocks (V1/V2/V3 neighbor shells, optional general
    Vr(r) function), plus an onsite U between up/down -- exactly mirroring
    Vinteraction's own construction (selfconsistency/densitydensity.py).
    Kept as a small separate copy here (rather than refactoring Vinteraction
    to share it) to avoid touching that already-tested, widely-used code
    path; see _build_v's docstring for why the neighbor-shell key set this
    produces is independent of which of V1/V2/V3 happen to be zero."""
    from .. import specialhopping
    from .densitydensity import obj2geometryarray
    nd = h.geometry.neighbor_distances()
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
        Jx1=0.0, Jx2=0.0, Jx3=0.0, Jy1=0.0, Jy2=0.0, Jy3=0.0,
        Jz1=0.0, Jz2=0.0, Jz3=0.0, Jxr=None, Jyr=None, Jzr=None,
        mf=None, filling=0.5, mu=None, mix=0.1, nk=8, maxerror=1e-5, maxite=None,
        T=1e-7, verbose=0, constrains=[]):
    """Self-consistent mean field combining density-density interactions
    (U onsite Hubbard, V1/V2/V3/Vr neighbor-shell -- same convention as
    Vinteraction) with anisotropic spin-spin exchange (Jx/Jy/Jz Sa_i Sa_j
    -- same convention as Jinteraction) in a single SCF loop.

    This works by combining the two existing SCF modes rather than
    inventing new decoupling math: density-density interactions and
    Sz_i Sz_j are both already density-density interactions in the
    spin-orbital basis (Vinteraction's uniform sign pattern across the
    four spin blocks vs. SzSz's +/-1/4 one -- see the module docstring and
    _build_v), and Hartree-Fock decoupling (get_mf_normal) is linear in the
    interaction matrix, so the density-density contribution can simply be
    added into Jinteraction's z-channel matrix before entering its shared
    SCF loop -- no separate channel, and no rotation, needed for it (unlike
    Jx/Jy, which do need the rotate-decouple-rotate-back trick). The x/y
    channels are handled exactly as in Jinteraction.

    See Vinteraction and Jinteraction for the individual parameter
    conventions; only integration="ed" and the plain-mixing solver are
    supported (unlike Vinteraction/SzSz/SxSx/SySy)."""
    if not h0.has_spin: raise ValueError("VJinteraction needs a spinful Hamiltonian")
    if h0.has_eh: raise ValueError("VJinteraction is not implemented for BdG Hamiltonians")
    h1 = h0.get_multicell().get_dense()
    vz = _build_v(h1, Jz1, Jz2, Jz3, Jzr)
    vd = _build_density_v(h1, V1, V2, V3, U, Vr)
    vz = (MultiHopping(vz) + MultiHopping(vd)).get_dict()
    vx = _build_v(h1, Jx1, Jx2, Jx3, Jxr)
    vy = _build_v(h1, Jy1, Jy2, Jy3, Jyr)
    return _run_anisotropic_scf(h1, vx, vy, vz, mf, filling, mu, mix, nk,
            maxerror, maxite, T, verbose, constrains)


def _run_anisotropic_scf(h1, vx, vy, vz, mf, filling, mu, mix, nk,
        maxerror, maxite, T, verbose, constrains):
    """Shared SCF core for Jinteraction/VJinteraction: decouples the
    z-channel matrix `vz` directly (Hartree-Fock density-density in the
    lab/computational spin basis) and the x/y-channel matrices `vx`/`vy`
    by rotating the density matrix into the frame where that axis is the
    computational z axis, applying the same decoupling there, and rotating
    the resulting mean field back before summing all three contributions
    -- see Jinteraction's docstring for the physics. `h1` must already be
    h0.get_multicell().get_dense()."""
    from .densitydensity import (get_dm, get_mf_normal, mix_mf, diff_mf,
            update_hamiltonian, set_hoppings, hamiltonian2dict,
            get_dc_energy, SCF)
    from .mfconstrains import obj2mf
    h1.nk = nk
    # union of the three channels' bond directions: in general the
    # neighbor-shell hopping-dict builder could prune a channel's key set
    # differently depending on which of its J's are zero, so the lab-frame
    # density matrix must be requested at the union, not just vz's keys
    v_dirs = {d: None for d in (set(vz) | set(vx) | set(vy))}
    # the x/y rotations are fixed for the whole SCF loop, so build the
    # (small, 2x2-block) rotation matrices once via build_rotation_matrix
    # instead of paying a fresh matrix exponential on every one of the many
    # _rotate_dict/_rotate_dm calls compute_mf makes each iteration; the
    # backward rotation is just the forward matrix's dagger (R(-angle) =
    # R(angle)^dagger), so only Rx/Ry need to be built
    from ..rotate_spin import build_rotation_matrix
    n_orb = h1.intra.shape[0]//2
    Rx = build_rotation_matrix(n_orb, **_AXIS_ROTATION["x"])
    Ry = build_rotation_matrix(n_orb, **_AXIS_ROTATION["y"])

    def _rot_dict(dd, R):
        """Rotate a dict of Hamiltonian-like (hopping/mean-field) matrices:
        these live in the same convention as Hamiltonian.intra, for which
        R @ m @ R^dagger is the correct transformation (as used by
        Hamiltonian.global_spin_rotation, validated by SxSx/SySy)."""
        Rd = R.conj().T
        return {k: R @ m @ Rd for (k, m) in dd.items()}

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
        Rd = R.conj().T
        return {k: np.conj(R @ np.conj(m) @ Rd) for (k, m) in dd.items()}

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
    h0_dense = h1.copy() # reference Hamiltonian before the mean field is added

    def compute_mf(dm_lab):
        mf = get_mf_normal(vz, dm_lab)
        dm_x = _rot_dm(dm_lab, Rx) # dm needs the conjugated rotation
        mf_x = _rot_dict(get_mf_normal(vx, dm_x), Rx.conj().T) # mf does not
        mf = (MultiHopping(mf) + MultiHopping(mf_x)).get_dict()
        dm_y = _rot_dm(dm_lab, Ry)
        mf_y = _rot_dict(get_mf_normal(vy, dm_y), Ry.conj().T)
        mf = (MultiHopping(mf) + MultiHopping(mf_y)).get_dict()
        return mf

    def f(mf):
        h = h1.copy()
        hop = update_hamiltonian(hop0, mf)
        set_hoppings(h, hop)
        h = callback_h(h)
        dm_lab = get_dm(h, v_dirs, nk=nk, T=T, integration="ed")
        mfnew = compute_mf(dm_lab)
        if callback_mf is not None: mfnew = callback_mf(mfnew)
        scf = SCF()
        scf.hamiltonian = h
        scf.hamiltonian0 = h0_dense
        scf.mf = mfnew
        scf.dm = dm_lab
        scf.v = vz # for identify_symmetry_breaking's tolerance bookkeeping
        scf.tol = maxerror
        return scf

    if mf is None:
        mf = dict()
        # seed over v_dirs (the vz/vx/vy key union), not just vz's own keys:
        # a channel with only its own neighbor shells (e.g. Jx1 nonzero,
        # Jz1=Jz2=Jz3=0) can have bond-direction keys absent from vz, and
        # diff_mf below only iterates the keys already present in this
        # initial guess, so seeding too few of them would make those
        # channels invisible to the very first convergence check
        for d in v_dirs: mf[d] = np.exp(1j*np.random.random(h1.intra.shape))*1e-1
        mf[(0, 0, 0)] = mf[(0, 0, 0)] + mf[(0, 0, 0)].T.conjugate()
    else:
        if isinstance(mf, str):
            from ..meanfield import guess
            mf = guess(h0_dense, mode=mf)
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

    # total energy: sum of occupied energies plus the double-counting
    # correction for each of the three (independently-rotated) channels
    h = scf.hamiltonian
    etot = h.get_total_energy(nk=h.nk)
    if mu is None:
        etot += h.fermi*h.intra.shape[0]*filling
    etot += get_dc_energy(vz, scf.dm)
    dm_x = _rot_dm(scf.dm, Rx) # dm needs the conjugated rotation, see compute_mf
    etot += get_dc_energy(vx, dm_x)
    dm_y = _rot_dm(scf.dm, Ry)
    etot += get_dc_energy(vy, dm_y)
    scf.total_energy = etot.real
    return scf
