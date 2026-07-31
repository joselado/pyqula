# JAX-derivative-based counterpart of selfconsistency/spinspin.py's
# VJinteraction.
#
# Same physical model (combined V/U density-density + J1/J2/J3/J1x/J1y/J1z
# spin-spin exchange mean field, normal-state only) as the numpy engine in
# spinspin.py's _run_anisotropic_scf, but the one-SCF-step map
#     mf_vector -> mf_vector_new
# is a pure, differentiable JAX function. That lets solver="newton" (and
# "newton_krylov"/"fsolve") solve the fixed point x = step(x) with a genuine
# root-finder driven by JAX-computed derivatives of step, instead of only
# linear mixing -- the same idea selfconsistency/densitydensity_jax.py
# already implements for Vinteraction (V/U-only, no exchange). This module
# reuses that machinery almost entirely: densitydensity_jax.solve_scf (the
# shared dispatch across newton_solve/fsolve_solve/newton_krylov_solve/
# fixed_point_solve/lbfgs_solve) is generic in the step function it is
# handed and needs no changes at all; only the step function itself needs
# generalizing, from a single density-density channel to VJinteraction's
# three channels (vz direct, vx/vy via the rotate-decouple-rotate-back
# trick spinspin._run_anisotropic_scf uses for anisotropic exchange).
#
# Deliberately narrower in scope than spinspin.VJinteraction, mirroring
# exactly how densitydensity_jax.py scopes down relative to densitydensity.py:
#  - normal-state (has_eh=False) mean field only, no anomalous/BdG part
#  - dense exact diagonalization only (jax.numpy.linalg.eigh), no KPM/sparse
#  - no constrains/callback_mf (needs concrete numpy arrays each iteration,
#    incompatible with jax tracing -- same restriction densitydensity_jax.py's
#    solver="newton"/"fsolve"/"newton_krylov" already have for Vinteraction)
#  - a target filling is supported the same way densitydensity_jax.py does:
#    mu is resolved *inside* the trace each step as the midpoint between the
#    n_occ_total-th and (n_occ_total+1)-th eigenvalue of the full (sorted)
#    spectrum, rather than a numpy root-find outside the trace
#  - occupations always use a finite smearing temperature T (default 1e-4,
#    see densitydensity_jax.default_T_jax) since jnp.linalg.eigh's eigenvector
#    gradient is only well defined away from exact degeneracies
#
# solver="error_gradient" (dispatched internally as densitydensity_jax's
# solver="levenberg_marquardt"): minimizes ||step(x)-x||^2 -- not by
# root-finding step(x)=x the way newton/fsolve/newton_krylov do, but as a
# genuine nonlinear-least-squares problem, via matrix-free Levenberg-
# Marquardt (densitydensity_jax.levenberg_marquardt_solve): jax.jvp/jax.vjp
# give Jacobian-vector and Jacobian-transpose-vector products of the
# residual r(x)=step(x)-x, and each LM step solves the damped subproblem
# with scipy.sparse.linalg.lsqr, entirely matrix-free (no O(norb^2)
# Jacobian, same motivation as newton_krylov's GMRES). Originally
# (2026-07-30) this dispatched to densitydensity_jax's scipy-L-BFGS-B-based
# solver="lbfgs" instead (driven only by jax.grad of the scalar loss) --
# still available directly via generic_densitydensity_jax's own
# solver="lbfgs" for Vinteraction, but replaced here after it was found
# (2026-07-31) to stall on larger systems; see levenberg_marquardt_solve's
# own docstring and the "Scaling" section below for the measured failure
# and fix.
#
# What was tried first and abandoned: minimizing the actual physical
# free-energy/grand-potential functional directly, rather than the SCF
# residual. This is a real, well-defined functional -- writing
#   Omega_trial(x) = (1/nk) sum_{k,n} [-T*softplus(-(e_kn(x)-mu)/T)]
# (the grand potential of the fictitious non-interacting trial Hamiltonian
# H(x)=H0+x) and correcting it for double-counting the interaction,
#   Omega_phys(x) = Omega_trial(x) - Re Tr[x.rho(x)] - sum_channel
#                   get_dc_energy(v_channel, dm_channel(x))
#   [+ mu_eff(x)*n_occ_total/nk for a filling target, the same Legendre
#    add-back scf.total_energy itself already applies there],
# gives dOmega_phys/dx = (Delta[rho(x)]-x).(response operator), which is
# exactly zero (for any response operator) iff Delta[rho(x)]=x, i.e. iff x
# is an SCF fixed point -- confirmed numerically to jax.grad(Omega_phys)(x)
# ~ 1e-16 at Newton-converged solutions, for both fixed-mu and filling-
# target cases, V/U-only and combined-exchange cases. Both the -Re Tr[x rho]
# term and the filling-target add-back were confirmed REQUIRED (not
# redundant with the dc-energy correction): omitting either leaves the
# functional's *value* unchanged at the SCF solution but its *gradient*
# nonzero there (measured ~0.7 and ~0.5 respectively, vs ~1e-16 with them).
#
# Despite the gradient being exactly correct, minimizing Omega_phys with
# L-BFGS-B empirically failed: the FULL Hessian of Omega_phys at a Newton-
# converged SCF solution (a small bichain+U dimer, checked by explicit
# eigendecomposition) is INDEFINITE -- 6 negative eigenvalues, 13 exactly
# zero, 13 positive, out of 32 -- i.e. the physical SCF solution is
# generically a SADDLE POINT of Omega_phys, not a minimum. This is not a
# sign/factor bug (the gradient check above rules that out) and not
# evidence that the SCF solution itself is a poor/metastable physical
# state either -- perturbing x along the most unstable eigenvector and
# re-running Newton from the perturbed point converges right back to the
# exact same solution both directions (agreement to ~1e-15), so it is a
# locally isolated, unique SCF fixed point. The negative curvature instead
# reflects that Omega_phys, evaluated OFF the self-consistency surface
# (at x with step(x)!=x), is an essentially arbitrary off-shell extension
# of the physical energy with no guarantee of convexity -- a known,
# general phenomenon for Hartree-Fock/mean-field energy functionals (most
# production electronic-structure codes use Newton/DIIS-style SCF
# acceleration rather than naive energy minimization for exactly this
# reason). In practice, L-BFGS-B on Omega_phys reliably ran downhill AWAY
# from the true solution into unrelated, non-self-consistent points with a
# deceptively small gradient (residual ~1.8-14 vs the ~1e-6 needed) even
# starting only 0.05 away from the exact answer. A Lagrange multiplier does
# not fix this: mu is already exactly that (for the filling constraint),
# and the saddle behavior appears even with no filling constraint at all;
# adding a multiplier to enforce Delta[rho(x)]=x exactly and seeking a
# saddle point of the resulting Lagrangian is mathematically equivalent to
# solving Delta[rho(x)]=x directly -- i.e. just solver="newton" again with
# extra steps, not a new capability.
#
# ||step(x)-x||^2 sidesteps all of this: it is a plain sum of squares, so
# its global minimum (value 0) sits exactly at every SCF fixed point by
# construction, with no separate derivation or saddle-point risk. Measured
# behavior (see tests/scf/test_vjinteraction_jax.py): matches solver=
# "newton"/"newton_krylov" closely on V/U-only, combined V+anisotropic-J,
# and filling-target bichain systems (all to <1e-5 in mf/total_energy, and
# to ~1e-8 against newton_krylov on the 30-site chain in the "Scaling"
# section below). It is still only a LOCAL method, so it is not immune to
# getting stuck in a nonzero-residual local minimum of the least-squares
# landscape on a sufficiently hard problem -- always check scf.converged
# rather than assuming a returned Hamiltonian is self-consistent, the same
# caveat every other solver here documents for its own marginal cases.
#
# History: the original (2026-07-30) scipy-L-BFGS-B-based implementation of
# this solver had exactly this problem, but severely -- on a 30-site
# (60-orbital) biased Hubbard chain (nk=20) it did not reach maxerror=1e-6
# even after 3000 L-BFGS-B iterations (77s), residual stuck around 5e-3 to
# 1.5e-2, INCLUDING when seeded with 5-40 linear_mixing warm-up iterations
# first (the fix that worked for solver="broyden_mixing"'s own analogous
# stall, see broydenmixing.py -- it did not transfer here). The root cause:
# L-BFGS-B only ever sees the scalar gradient of the squared residual and
# discards the residual VECTOR's own structure, whereas newton_krylov's
# jax.jvp-driven GMRES uses that structure directly and converged the same
# system in 10 iterations/9.0s. levenberg_marquardt_solve (2026-07-31) fixes
# this by using that same residual structure (jvp AND vjp, i.e. an actual
# nonlinear-least-squares method, not a generic scalar-loss optimizer)
# while keeping the "why not just use newton_krylov" advantage this solver
# was always meant to have: LM's Tikhonov damping keeps its subproblem
# well-posed even when the Jacobian is singular/near-singular (e.g. the
# unbroken-continuous-spin-symmetry marginal direction warned about above
# newton_krylov_solve in densitydensity_jax.py), where GMRES on a literally
# singular operator can simply fail to find any improving step at all.
#
# Scaling (measured, not assumed -- a biased chain+U, nk=20, fixed mu=0,
# wall time includes one-time jax/XLA compilation overhead so treat these
# as rough orders of magnitude, not a precise benchmark):
#   n=8 orbitals:  error_gradient 2.0s/9 iters,  linear_mixing 0.16s/136 iters,
#                  newton_krylov 0.3s/4 iters
#   n=24 orbitals: error_gradient 2.3s/8 iters,  linear_mixing 0.2s/126 iters,
#                  newton_krylov 103s/4 iters (GMRES apparently needing many
#                  more inner iterations per outer Newton step at this size)
#   n=60 orbitals (30 sites): error_gradient 9.2s/9 iters (matches
#                  newton_krylov's answer to dE~1.3e-8, d(mf)~1.8e-8),
#                  newton_krylov 6.1s/8 iters, linear_mixing 2.0s/175 iters
#                  (converges, but to a total_energy ~1.1e-5 off from the
#                  other two -- linear_mixing's own maxerror=1e-6 is on the
#                  mean-field vector, not energy, so this is expected slop,
#                  not a bug)
# i.e. error_gradient's outer-iteration count now stays flat/small (~6-9)
# across this whole size range rather than blowing up, matching the
# motivating idea (matrix-free, no O(norb^2) Jacobian) -- and at n=60,
# where the old L-BFGS-B version failed outright, it now succeeds and
# closely tracks newton_krylov, at comparable (not dramatically better)
# wall time; a proper scaling study to larger sizes, and a case exercising
# the singular-Jacobian robustness levenberg_marquardt_solve is specifically
# meant to have over newton_krylov, are natural follow-ups, not done here.
from __future__ import annotations
import functools
import warnings
import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from .densitydensity import (SCF, set_hoppings, hamiltonian2dict,
        get_dc_energy, random_hermitian_guess)
from .densitydensity_jax import (flatten_mf, unflatten_mf, make_bloch_stack,
        get_mf_normal_jax, default_T_jax, solve_scf)
from .spinspin import _build_v, _build_density_v, _channel_is_zero, _AXIS_ROTATION
from .mfconstrains import obj2mf
from ..multihopping import MultiHopping
from ..rotate_spin import build_rotation_matrix


def _block_rotate_jax(m, rot):
    """JAX port of spinspin._block_rotate: rot @ m @ rot^dagger, applied to
    every site's own 2x2 spin sub-block of an (n,n) matrix independently --
    see spinspin._block_rotate's docstring for why this is equivalent to,
    but much cheaper than, a dense (n,n)@(n,n) matmul against the full
    kron(I_{n/2},rot)."""
    n = m.shape[0]
    n_orb = n // 2
    m4 = m.reshape(n_orb, 2, n_orb, 2)
    out = jnp.einsum('ab,xbyc->xayc', rot, m4)
    out = jnp.einsum('xayc,dc->xayd', out, jnp.conj(rot))
    return out.reshape(n, n)


def _rot_dict_jax(dd, R):
    """JAX port of spinspin._rot_dict (Hamiltonian-like matrices: R @ m @ R^dagger)"""
    return {k: _block_rotate_jax(m, R) for (k, m) in dd.items()}


def _rot_dm_jax(dd, R):
    """JAX port of spinspin._rot_dm (density matrices need the conjugate-
    sandwiched transformation -- see that function's docstring for why)."""
    return {k: jnp.conj(_block_rotate_jax(jnp.conj(m), R)) for (k, m) in dd.items()}


def _rot_dm_np(dd, R):
    """Used only for the one-time (post-optimization) total-energy tail
    below -- reuses _rot_dm_jax/_block_rotate_jax directly (jnp runs fine on
    plain numpy/complex inputs outside any jit/grad trace) rather than
    duplicating the rotation formula a third time; spinspin._block_rotate/
    _rot_dm are nested closures inside _run_anisotropic_scf, not importable
    at module level, so calling into this module's own jax version -- kept
    in sync with the step function by construction -- is the reuse path
    that avoids that."""
    dd_j = {k: jnp.asarray(m) for k, m in dd.items()}
    R_j = jnp.asarray(R, dtype=jnp.complex128)
    return {k: np.asarray(m) for k, m in _rot_dm_jax(dd_j, R_j).items()}


@functools.lru_cache(maxsize=32)
def _get_step_core_vj(dirs, dirs_all, n, vz_active, vx_active, vy_active,
        has_filling_target):
    """VJinteraction's analogue of densitydensity_jax._get_step_core -- see
    that function's docstring for why this exists: build_step_function_vj
    used to close over hop0/vz/vx/vy/ks/T as Python constants baked into the
    traced program, so every top-level call (even with an identical problem
    shape but different Hamiltonian/interaction values, e.g. a parameter
    sweep or several random mf seeds on the same system) paid a fresh XLA
    recompile. Here the actual per-iteration computation is jitted exactly
    once per distinct STATIC/structural key (which directions exist, system
    size, which of the z/x/y channels are active, fixed-mu vs
    filling-target mode) and reused via jax's own per-shape compiled-
    executable cache for every call that shares it -- build_step_function_vj
    now just supplies the concrete numeric arrays (ms0, vz/vx/vy, ks, T,
    Rx/Ry) as jit trace arguments instead of baking them in."""
    ds_arr = jnp.array([list(d) for d in dirs_all], dtype=jnp.float64)
    dir_phase = jnp.array([list(d) for d in dirs], dtype=jnp.float64)  # (nt,3)

    def step_core(x, mu, ms0, vz_j, vx_j, vy_j, ks, T, n_occ_total, Rx_j, Ry_j):
        mf = unflatten_mf(x, dirs, n)
        mats = [ms0[i] + mf[d] if d in mf else ms0[i]
                for i, d in enumerate(dirs_all)]
        ms = jnp.stack(mats)

        def hk(k):
            phases = jnp.exp(1j * 2 * jnp.pi * (ds_arr @ k))
            return jnp.einsum('nij,n->ij', ms, phases)

        hks = jax.vmap(hk)(ks)                      # (nk,n,n)
        es, vs = jnp.linalg.eigh(hks)                # (nk,n), (nk,n,n)
        nk = ks.shape[0]
        if has_filling_target:
            es_sorted = jnp.sort(es.reshape(-1))
            mu_eff = 0.5 * (es_sorted[n_occ_total - 1] + es_sorted[n_occ_total])
        else:
            mu_eff = mu
        occ = jax.nn.sigmoid(-(es - mu_eff) / T)     # (nk,n)
        kd = ks @ dir_phase.T                        # (nk,nt)
        phase = jnp.exp(1j * 2 * jnp.pi * kd)         # (nk,nt)
        dm_all = jnp.einsum('kt,kie,ke,kje->tij', phase,
                jnp.conj(vs), occ, vs) / nk           # (nt,n,n)
        dm = {d: dm_all[i] for i, d in enumerate(dirs)}

        zero = dm[(0, 0, 0)] * 0.0
        if vz_active:
            mfnew = get_mf_normal_jax(vz_j, dm, dirs)
        else:
            mfnew = {d: zero for d in dirs}
        if vx_active:
            dm_x = _rot_dm_jax(dm, Rx_j)  # dm needs the conjugated rotation
            mf_x = _rot_dict_jax(get_mf_normal_jax(vx_j, dm_x, dirs),
                    jnp.conj(Rx_j).T)     # mf does not
            mfnew = {d: mfnew[d] + mf_x[d] for d in dirs}
        if vy_active:
            dm_y = _rot_dm_jax(dm, Ry_j)
            mf_y = _rot_dict_jax(get_mf_normal_jax(vy_j, dm_y, dirs),
                    jnp.conj(Ry_j).T)
            mfnew = {d: mfnew[d] + mf_y[d] for d in dirs}

        xnew = flatten_mf(mfnew, dirs)
        return xnew, dm, es, occ, mu_eff

    return jax.jit(step_core)


def build_step_function_vj(hop0, vz, vx, vy, ks, dirs, dirs_all, T,
        Rx, Ry, vz_active, vx_active, vy_active, n_occ_total=None):
    """Return step(x,mu) -> (xnew, dm, es, occ, mu_eff), the pure-JAX one
    SCF step combining the z/x/y channels -- structurally
    densitydensity_jax.build_step_function (same Bloch-stack build,
    vmap(eigh), mu-for-filling jnp.sort trick, per-direction dm
    reconstruction via einsum) generalized from a single get_mf_normal_jax
    call to VJinteraction's three-channel combination, mirroring
    spinspin._run_anisotropic_scf.compute_mf restricted to the normal-state
    case (no has_eh/vd-separate/embed_normal branches -- vd is folded into
    vz by the caller before this point, exactly as the numpy VJinteraction
    already does whenever has_eh=False).

    vz/vx/vy must already be padded to have an entry for every key in dirs
    (zero matrices where a channel does not itself reach that direction) --
    get_mf_normal_jax indexes v[d] directly for every d in dirs, unlike the
    numpy get_mf_normal which only loops over v's own (possibly smaller) key
    set.

    The heavy computation lives in a cached, once-jitted core (see
    _get_step_core_vj) shared across every call with the same structural
    shape -- the returned step() is a thin, NOT separately jitted, wrapper
    that just supplies the concrete numeric arrays as arguments to it;
    callers should not wrap it in another jax.jit (see _get_step_core_vj's
    docstring for why that would defeat the point)."""
    n = hop0[(0, 0, 0)].shape[0]
    ms0 = make_bloch_stack(hop0, dirs_all, n)
    vz_j = {d: jnp.asarray(vz[d], dtype=jnp.complex128) for d in vz}
    vx_j = {d: jnp.asarray(vx[d], dtype=jnp.complex128) for d in vx}
    vy_j = {d: jnp.asarray(vy[d], dtype=jnp.complex128) for d in vy}
    Rx_j = jnp.asarray(Rx, dtype=jnp.complex128) if Rx is not None else None
    Ry_j = jnp.asarray(Ry, dtype=jnp.complex128) if Ry is not None else None
    T_arr = jnp.asarray(T, dtype=jnp.float64)
    has_filling_target = n_occ_total is not None
    n_occ_arr = jnp.asarray(n_occ_total if has_filling_target else 0,
            dtype=jnp.int64)
    core = _get_step_core_vj(tuple(dirs), tuple(dirs_all), n, vz_active,
            vx_active, vy_active, has_filling_target)

    def step(x, mu):
        return core(x, mu, ms0, vz_j, vx_j, vy_j, ks, T_arr, n_occ_arr,
                Rx_j, Ry_j)

    return step


# VJinteraction/VJinteraction_jax's own public solver= names differ from
# the internal dispatch names solve_scf (shared with Vinteraction/
# densitydensity_jax.py, not renamed here) expects: "error_gradient"
# describes what the solver does (minimizing the SCF residual, currently via
# levenberg_marquardt_solve) rather than naming the specific algorithm
# behind it, so that name can keep meaning the same thing even if the
# algorithm behind it changes again (as it already has once, from
# scipy L-BFGS-B to matrix-free Levenberg-Marquardt -- see the module
# docstring's "solver='error_gradient'" section), and "linear_mixing" names
# the algorithm itself rather than its role as a
# baseline/comparison point ("fixed_point"). Translated once at the entry
# point below so solve_scf's own dispatch (and its "lbfgs"/"fixed_point"
# checks, shared verbatim with Vinteraction) never needs to know about the
# VJinteraction-only names.
_PUBLIC_SOLVER_NAMES = {"error_gradient": "levenberg_marquardt",
        "linear_mixing": "fixed_point"}


def generic_vjinteraction_jax(h0, vz, vx, vy, mf=None, nk=8, mu=0.0,
        filling=None, T=None, mix=0.1, maxerror=1e-5, maxite=2000,
        solver="newton", verbose=0, gmres_tol=1e-6, gmres_restart=20):
    """JAX-differentiable analogue of spinspin._run_anisotropic_scf,
    restricted to the normal-state case -- see the module docstring for the
    full scope restriction. vz/vx/vy are the (unpadded, possibly
    smaller-than-`dirs`) numpy interaction matrices built by
    spinspin._build_v/_build_density_v; vd (density-density) must already be
    folded into vz by the caller, exactly as VJinteraction itself does for
    has_eh=False."""
    if solver != "linear_mixing" and mix != 0.1:
        # mix only controls solver="linear_mixing"'s linear-mixing step --
        # newton/fsolve/newton_krylov/error_gradient all use their own
        # backtracking/damping (Levenberg-Marquardt's own lam for
        # error_gradient), so a
        # caller-tuned mix (e.g. carried over from the numpy engine, where
        # it always matters) would otherwise be silently ignored with no
        # signal at all
        warnings.warn("mix=%r has no effect for solver=%r (only "
                "solver=\"linear_mixing\" uses linear mixing)"
                % (mix, solver), stacklevel=2)
    dispatch_solver = _PUBLIC_SOLVER_NAMES.get(solver, solver)
    if T is None:
        T = default_T_jax
    elif T <= 0:
        raise ValueError("T=%r is not usable with use_jax=True: occupations "
                "are occ=sigmoid(-(e-mu)/T), so T<=0 (including exactly 0) "
                "divides by a non-positive number and produces NaN/Inf, "
                "unlike the numpy engine's T=0 hard Fermi step "
                "(delta = T if T != 0 else 1e-15) -- pass a small positive "
                "T instead (e.g. the jax engine's own default, "
                "densitydensity_jax.default_T_jax=%r)" % (T, default_T_jax))
    h1 = h0.copy()
    h1 = h1.get_dense()
    h1.nk = nk
    hop0 = hamiltonian2dict(h1)  # numpy dict, bare hoppings
    n = hop0[(0, 0, 0)].shape[0]

    vz_active = not _channel_is_zero(vz)
    vx_active = not _channel_is_zero(vx)
    vy_active = not _channel_is_zero(vy)

    # the x/y rotations are fixed for the whole SCF loop -- build the small
    # 2x2 spin rotation matrices once, same reasoning as
    # _run_anisotropic_scf's own Rx/Ry precomputation
    Rx = Ry = None
    if vx_active:
        Rx = build_rotation_matrix(1, **_AXIS_ROTATION["x"])
    if vy_active:
        Ry = build_rotation_matrix(1, **_AXIS_ROTATION["y"])

    dirs = sorted(set(vz) | set(vx) | set(vy))
    if (0, 0, 0) not in dirs:
        dirs = [(0, 0, 0)] + dirs
    dirs_all = sorted(set(hop0.keys()) | set(dirs))
    ks = jnp.asarray(np.array(h1.geometry.get_kmesh(nk=nk)), dtype=jnp.float64)

    # pad each channel's interaction matrix to the full `dirs` union: a
    # channel with a smaller key set than another (e.g. Jx1 nonzero, Jz set
    # to a different neighbor range) must still supply a (zero) matrix for
    # every direction get_mf_normal_jax is asked about -- see
    # build_step_function_vj's docstring. An inactive channel (vx_active/
    # vy_active False) is padded to {} instead: step()/free-energy code
    # never reads vx_j/vy_j when the channel is inactive (guarded by the
    # same vx_active/vy_active check), so padding+jnp-converting a full set
    # of all-zero (n,n) complex128 matrices for it would be pure wasted
    # host-to-device transfer -- e.g. ~12.8MB for n=200, len(dirs)~10.
    zero_np = np.zeros((n, n), dtype=np.complex128)
    def _pad(v):
        return {d: (v[d] if d in v else zero_np) for d in dirs}
    vz_full = _pad(vz)
    vx_full = _pad(vx) if vx_active else {}
    vy_full = _pad(vy) if vy_active else {}

    if mf is None:
        mf = random_hermitian_guess({d: None for d in dirs}, h1.intra.shape,
                scale=1e-1)  # same scale as VJinteraction's own default guess
    elif isinstance(mf, str):
        from ..meanfield import guess
        mf = guess(h0, mode=mf)
    mf = obj2mf(mf)
    zero_n = jnp.zeros((n, n), dtype=jnp.complex128)
    x0 = flatten_mf({d: jnp.asarray(mf[d], dtype=jnp.complex128)
        if d in mf else zero_n for d in dirs}, dirs)

    n_occ_total = None
    if filling is not None:
        n_tot = n * ks.shape[0]
        n_occ_total = int(round(filling * n_tot))
        n_occ_total = min(max(n_occ_total, 1), n_tot - 1)

    # not wrapped in an extra jax.jit: build_step_function_vj's returned
    # closure already dispatches into a cached, once-jitted core shared
    # across every call with this same structural shape -- see
    # build_step_function_vj/_get_step_core_vj's docstrings
    step_jit = build_step_function_vj(hop0, vz_full, vx_full, vy_full, ks,
            dirs, dirs_all, T, Rx, Ry, vz_active, vx_active, vy_active,
            n_occ_total=n_occ_total)

    # solve_scf (selfconsistency.densitydensity_jax) is the same solver
    # dispatch densitydensity_jax.generic_densitydensity_jax's own
    # use_jax=True path uses for Vinteraction -- see its docstring for the
    # solver="lbfgs" residual-minimization rationale (this module's own
    # docstring "solver='error_gradient'" section) and why the single
    # trailing step_jit(x, mu) call it makes is exactly correct here too.
    # dispatch_solver (not the public `solver` name) is what solve_scf sees
    # -- see _PUBLIC_SOLVER_NAMES above.
    # callback_mf is never passed (VJinteraction's use_jax=True path has no
    # such hook at all), so solve_scf's NotImplementedError branch for it
    # never triggers here.
    x, final_mu, ite, converged, dm, es, occ = solve_scf(step_jit, x0, mu,
            dirs, n, dispatch_solver, maxite, maxerror, mix, verbose, gmres_tol,
            gmres_restart)
    mf_final = unflatten_mf(x, dirs, n)
    dm_np = {d: np.asarray(dm[d]) for d in dirs}
    mf_np = {d: np.asarray(mf_final[d]) for d in dirs}

    hop_final = dict()
    for d in dirs_all:
        m = np.asarray(hop0[d]) if d in hop0 else np.zeros((n, n), dtype=complex)
        if d in mf_np:
            m = m + mf_np[d]
        hop_final[d] = m
    h_final = h1.copy()
    set_hoppings(h_final, hop_final)
    # Mirror spinspin._run_anisotropic_scf's callback_h *exactly*, including
    # which branch sets .fermi: for a filling target (mu is None upstream,
    # i.e. n_occ_total is not None here) it sets h.fermi=fermi and shifts by
    # -fermi; for a fixed mu it only shifts by -mu and never assigns .fermi
    # at all (so scf.hamiltonian.fermi raises AttributeError there, same as
    # the numpy engine -- code checking hasattr(h,'fermi') to detect a
    # filling-target run must see the same answer regardless of use_jax).
    # This step's own dm/es/occ above were computed directly against mu_eff
    # without needing any shift, but the *returned* h_final must still carry
    # it, and total_energy must be computed via h_final.get_total_energy
    # (the exact same call the numpy engine's own total-energy tail makes on
    # its identically-shifted h) rather than a hand-derived formula from the
    # raw (unshifted) es/occ above: for a fixed, nonzero mu the numpy engine
    # sums the SHIFTED eigenvalues with no compensating add-back (only the
    # filling-target branch adds one, since sum(e_n-fermi,occ)+fermi*N*filling
    # telescopes back to sum(e_n,occ) when N_occ==N*filling); reusing
    # sum(occ*es)/nk unconditionally here matched only the filling-target
    # case and silently gave the wrong energy (off by mu*N_occ) whenever a
    # caller passed an explicit nonzero mu -- caught by
    # test_vjinteraction_jax_fixed_nonzero_mu_matches_numpy_engine.
    if n_occ_total is not None:
        h_final.fermi = final_mu
    h_final.shift_fermi(-final_mu)

    # total energy: same h_final.get_total_energy() call (plus the identical
    # filling-target add-back) the numpy engine's own total-energy tail
    # makes, so this is defined identically to it by construction -- then
    # the double-counting correction for each active channel, rotated into
    # that channel's own frame first for vx/vy, exactly
    # spinspin._run_anisotropic_scf's total-energy tail (reusing the same
    # numpy get_dc_energy/_rot_dm_np, not re-derived)
    etot = h_final.get_total_energy(nk=nk)
    if n_occ_total is not None:
        etot += h_final.fermi * n * filling
    etot = float(etot)
    if vz_active:
        etot += get_dc_energy(vz, dm_np)
    if vx_active:
        dm_x = _rot_dm_np(dm_np, Rx)
        etot += get_dc_energy(vx, dm_x)
    if vy_active:
        dm_y = _rot_dm_np(dm_np, Ry)
        etot += get_dc_energy(vy, dm_y)

    scf = SCF()
    scf.hamiltonian = h_final
    scf.hamiltonian.V = vz
    scf.hamiltonian0 = h0
    scf.mf = mf_np
    scf.dm = dm_np
    scf.v = vz
    scf.tol = maxerror
    scf.converged = bool(converged)
    if not scf.converged:
        # unconditional (not gated on verbose), matching the numpy engine's
        # own "No convergence has been reached..." print -- maxite here is
        # 2000 by default under use_jax=True even when the caller left the
        # numpy engine's maxite=None (unbounded) default, so this is the
        # only signal such a caller gets that the jax engine gave up early
        print("No convergence has been reached in", ite,
                "iterations (solver=%r), stopping" % (solver,))
    scf.total_energy = etot.real
    scf.mu = final_mu
    scf.iterations = ite
    if verbose > 1:
        print("##################")
        print("Total energy", scf.total_energy)
        print("Converged", scf.converged, "in", ite, "iterations")
        print("##################")
    return scf


def VJinteraction_jax(h0, V1=0.0, V2=0.0, V3=0.0, U=0.0, Vr=None,
        J1=0.0, J2=0.0, J3=0.0, Jr=None, J1x=0.0, J1y=0.0, J1z=0.0,
        mf=None, filling=0.5, mu=None, nk=8, maxerror=1e-5, maxite=2000,
        T=None, mix=0.1, verbose=0, solver="newton",
        gmres_tol=1e-6, gmres_restart=20):
    """JAX drop-in for spinspin.VJinteraction (use_jax=True path) -- see the
    module docstring for the scope restriction relative to the full numpy
    engine (normal-state only, dense ED, no constrains). Builds the same
    vz/vx/vy interaction matrices as the numpy VJinteraction
    (spinspin._build_v/_build_density_v, vd folded into vz for has_eh=False)
    and solves the resulting SCF fixed point with a JAX-derivative-based
    root-finder (solver="newton"/"fsolve"/"newton_krylov", or "linear_mixing"
    for plain linear mixing through the same machinery) instead of
    VJinteraction's own hardcoded plain-mixing loop. solver="error_gradient"
    instead minimizes ||step(x)-x||^2 via matrix-free Levenberg-Marquardt
    (jax.jvp/jax.vjp + scipy's lsqr) -- see the module docstring's
    "solver='error_gradient'" section for why this (residual-norm
    minimization), and not minimizing the physical free energy directly, is
    what it does."""
    if not h0.has_spin:
        return NotImplemented  # only for spinful systems, same as VJinteraction
    if h0.has_eh:
        raise NotImplementedError("VJinteraction's use_jax=True does not "
                "support the anomalous/BdG mean field yet; use the default "
                "(numpy) engine")
    if mu is None and filling is None:
        raise ValueError("VJinteraction's use_jax=True needs either mu "
                "(a fixed chemical potential) or filling (a target filling, "
                "resolved to a mu internally each SCF step) -- got both "
                "None, which would silently run a mu=0.0 calculation "
                "instead of raising, if not caught here")
    h1 = h0.get_multicell().get_dense()
    nd = h1.geometry.neighbor_distances()  # shared by all three _build_*_v calls
    vz = _build_v(h1, J1 + J1z, J2, J3, Jr, nd=nd)
    vd = _build_density_v(h1, V1, V2, V3, U, Vr, nd=nd)
    vx = _build_v(h1, J1 + J1x, J2, J3, Jr, nd=nd)
    vy = _build_v(h1, J1 + J1y, J2, J3, Jr, nd=nd)
    vz = (MultiHopping(vz) + MultiHopping(vd)).get_dict()  # fold density-density in

    kwargs = dict(mf=mf, nk=nk, T=T, mix=mix, maxerror=maxerror, maxite=maxite,
            solver=solver, verbose=verbose, gmres_tol=gmres_tol,
            gmres_restart=gmres_restart)
    if mu is not None:
        return generic_vjinteraction_jax(h1, vz, vx, vy, mu=mu, filling=None,
                **kwargs)
    else:
        return generic_vjinteraction_jax(h1, vz, vx, vy, mu=0.0,
                filling=filling, **kwargs)
