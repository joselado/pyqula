# JAX-differentiable counterpart of scftk/densitydensity.py
#
# Same physical model (density-density mean field) as the numpy/numba
# engine in densitydensity.py, but the one-SCF-step map
#     mf_vector -> mf_vector_new
# is a pure, differentiable JAX function. That lets a genuine Newton
# solver (jax.jacfwd of the fixed-point residual) drive the self
# consistency condition to zero, instead of only linear mixing.
#
# Deliberately narrower in scope than densitydensity.py:
#  - normal (density-density) mean field only, no anomalous/BdG (has_eh) part
#  - no krylov/anderson/broyden1/linear scipy.optimize solvers
#  - no callback_h/callback_dm/callback_mf hooks
#  - a target filling IS supported by solver="newton": rather than resolving
#    mu(filling) with a numpy sort/root-find outside the trace (which would
#    break jax.jacfwd), mu is computed *inside* the trace each step as the
#    midpoint between the n_occ_total-th and (n_occ_total+1)-th eigenvalue
#    of the full (sorted, via jnp.sort) spectrum - jnp.sort's gradient is
#    well defined away from ties, so this stays differentiable
#  - occupations always use a finite smearing temperature T (default 1e-4)
#    because jnp.linalg.eigh's eigenvector gradient is only well defined
#    away from exact degeneracies; T=0/None silently falls back to the
#    default rather than erroring
#
# solver="newton"/"fsolve" do NOT scale to large systems: the mean field is
# parameterized as a dense vector of every entry of every mf[direction]
# matrix (no attempt to exploit physical sparsity), so for norb orbitals
# the Jacobian is O(norb^2) x O(norb^2), and jax.jacfwd builds it with
# O(norb^2) forward passes each doing an O(norb^3) batched eigh - a steep
# polynomial blowup. Measured on a 1D chain (V1 interaction, nk=8): 16
# orbitals took ~7s for 3 Newton iterations; 32 orbitals did not finish 3
# iterations in 280s. solver="fixed_point" has no such issue (it's a plain
# jitted forward pass per iteration, same asymptotic cost per iteration as
# the numpy engine) - e.g. it converged a 100-orbital, mu-fixed chain in
# 140 iterations / 3.7s, about 2x faster than the numpy engine's 7.2s for
# the same problem. Use solver="newton"/"fsolve" only for small systems
# (roughly dimer-to-tens-of-orbitals); for anything larger, use
# solver="fixed_point" regardless of whether the differentiability of
# Newton/fsolve would otherwise be preferred.
#
# solver="fsolve" wraps scipy.optimize.fsolve (MINPACK hybrj) with the same
# jax.jacfwd Jacobian as fprime, as an alternative globalization strategy to
# the hand-rolled backtracking Newton above - see fsolve_solve's docstring
# for how to check (via infodict['njev'] vs ['nfev']) whether it reuses
# Broyden updates of the Jacobian instead of rebuilding it every iteration.
#
# It does: on a 4-orbital problem njev=1 for nfev=16 (a single Jacobian,
# Broyden-updated for the rest) - but this does NOT fix the scaling problem,
# because the cost of building the Jacobian even once is already what's
# expensive (see above), and Broyden reuse only reduces the *count* of
# rebuilds, not their individual cost. Measured on the same 1D chain: 20
# orbitals converged in 53s with njev=4 (~13s/build); 32 orbitals did not
# finish even a couple of builds in 200s. So solver="fsolve" has roughly the
# same practical ceiling as solver="newton" (tens of orbitals, not ~100) -
# reaching ~100 orbitals with any Jacobian-based solver here would need a
# matrix-free approach (e.g. jax.jvp Jacobian-vector products feeding
# scipy.optimize.newton_krylov, never forming the dense O(norb^2)x O(norb^2)
# Jacobian at all).
#
# solver="newton_krylov" is exactly that matrix-free approach: the same
# damped-Newton outer loop as solver="newton", but each linear solve uses
# GMRES (scipy.sparse.linalg.gmres) with only Jacobian-VECTOR products
# from jax.jvp, never materializing the dense Jacobian. A jvp costs about
# one extra step_vec evaluation (O(norb^3), one more batched eigh), not
# O(norb^2) of those - this is the fix for the scaling problem above, as
# long as GMRES converges in a modest number of Krylov iterations. See
# newton_krylov_solve's docstring.
#
# WARNING: for solver in {"newton","fsolve","newton_krylov"}, an unbiased
# spinful Hamiltonian with an unbroken continuous spin-rotation symmetry
# leaves the Jacobian singular along that marginal direction, and the
# outer Newton loop can give up after zero completed iterations (the
# backtracking line search finds no improving step even on the very first
# try - see newton_solve's/newton_krylov_solve's "no backtracked step
# improved the residual" branch). scf.converged is False in that case, but
# scf.total_energy is still populated with whatever the unconverged state
# evaluates to (essentially the untouched initial guess) - this is easy to
# miss since no exception is raised. See the WARNING in
# densitydensity.Vinteraction's docstring; the fix used throughout this
# module's own tests is to bias the Hamiltonian itself (not just the mean
# field guess) along an arbitrary direction, e.g. h.add_exchange(0.8*v),
# or use a Hamiltonian that already breaks the symmetry physically (SOC,
# an external field, a spinless interaction).
from __future__ import annotations
import functools
import warnings
import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from .densitydensity import (SCF, set_hoppings, hamiltonian2dict,
        get_dc_energy, obj2geometryarray)
from .mfconstrains import obj2mf
from ..multihopping import MultiHopping

default_T_jax = 1e-4


def normal_term_ii_jax(v, dm):
    return jnp.diag(v @ jnp.diag(dm))


def normal_term_jj_jax(v, dm):
    return jnp.diag(v.T @ jnp.diag(dm))


def normal_term_ij_jax(v, dm):
    return -v * dm.T


def get_mf_normal_jax(v, dm, dirs, compute_dd=True, compute_cross=True,
        add_dagger=True):
    """JAX version of densitydensity.get_mf_normal (normal part only)"""
    zero = dm[(0, 0, 0)] * 0.0
    mf = {d: zero for d in dirs}
    for d in dirs:
        d2 = (-d[0], -d[1], -d[2])
        if compute_cross:
            m = normal_term_ij_jax(v[d], dm[d2])
            mf[d] = mf[d] + m
            if add_dagger:
                mf[d2] = mf[d2] + jnp.conj(m).T
        if compute_dd:
            m = normal_term_ii_jax(v[d], dm[(0, 0, 0)])
            mf[(0, 0, 0)] = mf[(0, 0, 0)] + m
            m = normal_term_jj_jax(v[d2], dm[(0, 0, 0)])
            mf[(0, 0, 0)] = mf[(0, 0, 0)] + m
    return mf


def flatten_mf(mf, dirs):
    """Mean field dict -> real vector (real/imag parts concatenated)"""
    parts = [jnp.real(mf[d]).reshape(-1) for d in dirs]
    parts += [jnp.imag(mf[d]).reshape(-1) for d in dirs]
    return jnp.concatenate(parts)


def unflatten_mf(x, dirs, n):
    """Real vector -> mean field dict"""
    nt = len(dirs)
    chunk = n * n
    mf = dict()
    for i, d in enumerate(dirs):
        re = x[i * chunk:(i + 1) * chunk].reshape(n, n)
        im = x[(nt + i) * chunk:(nt + i + 1) * chunk].reshape(n, n)
        mf[d] = re + 1j * im
    return mf


def make_bloch_stack(hop0, dirs_all, n):
    """Stack the bare hopping matrices in dirs_all order (zero if absent)"""
    zero = jnp.zeros((n, n), dtype=jnp.complex128)
    return jnp.stack([jnp.asarray(hop0[d], dtype=jnp.complex128)
        if d in hop0 else zero for d in dirs_all])


@functools.lru_cache(maxsize=32)
def _get_step_core(dirs, dirs_all, n, compute_dd, compute_cross, add_dagger,
        has_filling_target):
    """Build (once per distinct STATIC/structural key) and cache the actual
    jitted SCF-step computation. build_step_function used to close over the
    concrete Hamiltonian/interaction arrays (hop0, v, ks, T) as Python
    constants, baking their VALUES into the traced program -- so two calls
    with identical shapes but different numeric values (e.g. a parameter
    sweep, or the same system solved from several random mf seeds) produced
    structurally different jaxpr/HLO and could not share a compiled
    executable: every top-level SCF call paid a fresh ~0.6-1.2s XLA
    recompile (measured on an 8-60 orbital chain), dwarfing the actual
    per-iteration compute cost (sub-ms to ~13ms in the same range).

    Here those arrays are genuine jax.jit trace ARGUMENTS of step_core
    instead (see build_step_function, which now just supplies them each
    call), and step_core itself is jitted exactly once per distinct
    (dirs, dirs_all, n, compute_dd, compute_cross, add_dagger,
    has_filling_target) key -- the only things that actually change the
    SHAPE of the computation or its Python-level control flow (e.g. the
    n_occ_total-is-not-None filling-target branch). Repeat calls sharing a
    key reuse the same jax.jit-wrapped Python object, so jax's own
    per-shape compilation cache (keyed on argument abstract shapes/dtypes,
    not on which Python closure invoked it) takes over from there:
    identical-shape calls with different concrete hop0/v/ks/T values hit an
    already-compiled executable instead of recompiling."""
    ds_arr = jnp.array([list(d) for d in dirs_all], dtype=jnp.float64)
    dir_phase = jnp.array([list(d) for d in dirs], dtype=jnp.float64)  # (nt,3)

    def step_core(x, mu, ms0, v_jnp, ks, T, n_occ_total):
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
        mfnew = get_mf_normal_jax(v_jnp, dm, dirs, compute_dd=compute_dd,
                compute_cross=compute_cross, add_dagger=add_dagger)
        xnew = flatten_mf(mfnew, dirs)
        return xnew, dm, es, occ, mu_eff

    return jax.jit(step_core)


def build_step_function(hop0, v, ks, dirs, dirs_all, T,
        compute_dd, compute_cross, add_dagger, n_occ_total=None):
    """Return step(x,mu) -> (xnew, dm, es, occ, mu_eff), the pure-JAX one
    SCF step. If n_occ_total is given (a fixed number of occupied states
    out of the nk*norb total, i.e. a filling target), mu is IGNORED and
    instead computed inside the trace as the midpoint between the
    n_occ_total-th and (n_occ_total+1)-th eigenvalue in the whole (sorted)
    spectrum, via jnp.sort - unlike resolving mu(filling) with a numpy
    sort/root-find outside the trace, this stays fully differentiable
    (jnp.sort has a well-defined gradient away from ties) so solver="newton"
    can handle a fixed filling directly, not just a fixed mu.

    The heavy computation itself lives in a cached, once-jitted core (see
    _get_step_core) shared across every call with the same structural shape
    -- the returned step() is a thin, NOT separately jitted, wrapper that
    just supplies the concrete numeric arrays as arguments to it; callers
    should not wrap it in another jax.jit (that would reintroduce a
    fresh-compile-per-call cost for no benefit, since the actual physics
    computation is already compiled and cached inside step_core)."""
    n = hop0[(0, 0, 0)].shape[0]
    ms0 = make_bloch_stack(hop0, dirs_all, n)
    v_jnp = {d: jnp.asarray(v[d], dtype=jnp.complex128) for d in v}
    T_arr = jnp.asarray(T, dtype=jnp.float64)
    has_filling_target = n_occ_total is not None
    n_occ_arr = jnp.asarray(n_occ_total if has_filling_target else 0,
            dtype=jnp.int64)
    core = _get_step_core(tuple(dirs), tuple(dirs_all), n, compute_dd,
            compute_cross, add_dagger, has_filling_target)

    def step(x, mu):
        return core(x, mu, ms0, v_jnp, ks, T_arr, n_occ_arr)

    return step


def diff_mf_vec(x0, x1):
    return float(jnp.mean(jnp.abs(x0 - x1)))


def newton_solve(step_vec, x0, maxite=50, tol=1e-10, damping=1.0, verbose=0,
        max_backtrack=30):
    """Solve x = step_vec(x) with Newton's method on r(x) = step_vec(x) - x,
    using the exact JAX Jacobian (jax.jacfwd). A full undamped step can
    overshoot into a region where jnp.linalg.eigh's gradient is numerically
    ill-conditioned (near-degenerate eigenvalues) and blow up to NaN, so each
    step is backtracked (halved) until it actually decreases the residual;
    since any comparison against NaN is False in Python, a NaN'd trial step
    is automatically rejected. The backtracking merit function is the smooth
    sum-of-squares norm, not max(|r|): max-norm is only piecewise smooth
    (its argmax component can switch between vector entries between
    iterations), which was observed to stall the line search - accepting a
    step even though a smaller one would keep decreasing it, because the max
    stops going down while the overall residual is still shrinking."""
    def merit(r):
        return float(jnp.sum(jnp.abs(r) ** 2))
    jac_fn = jax.jacfwd(step_vec)
    x = x0
    n = x0.shape[0]
    eye = jnp.eye(n, dtype=x0.dtype)
    fx = step_vec(x)
    r = fx - x
    err = float(jnp.max(jnp.abs(r)))
    m = merit(r)
    for ite in range(maxite):
        if verbose > 0:
            print("Newton iteration", ite, "error", err)
        if err < tol:
            return x, ite, True
        J = jac_fn(x) - eye
        # least-squares (pseudo-inverse) rather than a plain solve: a weak
        # symmetry-breaking bias leaves J close to singular along the
        # near-marginal direction, and lstsq degrades gracefully there
        # instead of returning a huge, numerically meaningless step
        dx = jnp.linalg.lstsq(J, -r, rcond=1e-8)[0]
        step = damping
        x_try, err_try, m_try = x, err, m
        for _ in range(max_backtrack):
            x_try = x + step * dx
            fx_try = step_vec(x_try)
            r_try = fx_try - x_try
            m_try = merit(r_try)
            if m_try < m:
                err_try = float(jnp.max(jnp.abs(r_try)))
                break
            step *= 0.5
        else:
            # no backtracked step improved the residual: stuck, stop early
            return x, ite, err < tol
        x, fx, r, err, m = x_try, fx_try, r_try, err_try, m_try
    return x, maxite, err < tol


def newton_krylov_solve(step_vec, x0, maxite=50, tol=1e-10, damping=1.0,
        verbose=0, max_backtrack=30, gmres_tol=1e-6, gmres_restart=20,
        gmres_maxiter=None):
    """Matrix-free (Jacobian-free) Newton-Krylov: same damped-Newton outer
    loop and backtracking as newton_solve, but the linear system
    (J_step(x) - I) dx = -r(x) at each step is solved with GMRES using only
    Jacobian-VECTOR products from jax.jvp, never forming the dense
    O(norb^2) x O(norb^2) Jacobian that makes newton_solve/fsolve_solve
    expensive at scale. A jvp costs about the same as one extra evaluation
    of step_vec (one more batched eigh), i.e. O(norb^3), not O(norb^2) of
    those - the whole point of this solver. This is the classic JFNK
    (Jacobian-free Newton-Krylov) method, except the Jacobian-vector
    products are exact (via jax.jvp / forward-mode autodiff) rather than
    the usual finite-difference approximation."""
    from scipy.sparse.linalg import gmres, LinearOperator
    n = x0.shape[0]
    # jitted once; reused for every GMRES matvec call across all outer
    # Newton iterations (x becomes a traced argument, not baked in)
    jvp_fn = jax.jit(lambda x, v: jax.jvp(step_vec, (x,), (v,))[1] - v)

    def merit(r):
        return float(jnp.sum(jnp.abs(r) ** 2))

    def gmres_solve(x_cur, rhs_np):
        def matvec(v_np):
            # np.array(..., copy=True) rather than np.asarray: a numpy view
            # of a jax array's buffer can come back read-only, and scipy's
            # gmres does in-place updates on the vectors matvec returns
            return np.array(jvp_fn(x_cur, jnp.asarray(v_np)), copy=True)
        Jop = LinearOperator((n, n), matvec=matvec, dtype=np.float64)
        rhs_np = np.array(rhs_np, copy=True)
        try:
            dx_np, info = gmres(Jop, rhs_np, rtol=gmres_tol,
                    restart=gmres_restart, maxiter=gmres_maxiter)
        except TypeError:  # older scipy: "tol" instead of "rtol"
            dx_np, info = gmres(Jop, rhs_np, tol=gmres_tol,
                    restart=gmres_restart, maxiter=gmres_maxiter)
        return jnp.asarray(dx_np)

    x = x0
    fx = step_vec(x)
    r = fx - x
    err = float(jnp.max(jnp.abs(r)))
    m = merit(r)
    for ite in range(maxite):
        if verbose > 0:
            print("Newton-Krylov iteration", ite, "error", err)
        if err < tol:
            return x, ite, True
        dx = gmres_solve(x, -np.asarray(r))
        step = damping
        x_try, err_try, m_try = x, err, m
        for _ in range(max_backtrack):
            x_try = x + step * dx
            fx_try = step_vec(x_try)
            r_try = fx_try - x_try
            m_try = merit(r_try)
            if m_try < m:
                err_try = float(jnp.max(jnp.abs(r_try)))
                break
            step *= 0.5
        else:
            return x, ite, err < tol
        x, fx, r, err, m = x_try, fx_try, r_try, err_try, m_try
    return x, maxite, err < tol


def levenberg_marquardt_solve(step_vec, x0, maxite=200, tol=1e-8, verbose=0,
        lam0=1e-3, lam_factor=3.0, max_inner_tries=15, lsqr_iter_lim=20):
    """Matrix-free Levenberg-Marquardt on the residual r(x) = step_vec(x) - x,
    i.e. a proper nonlinear-least-squares solver for min ||r(x)||^2 - unlike
    lbfgs_solve (scipy's generic L-BFGS-B on the same objective, driven only
    by jax.grad of the scalar loss), this uses jax.jvp/jax.vjp to get actual
    Jacobian-vector and Jacobian-transpose-vector products of r itself, and
    solves the LM subproblem with scipy.sparse.linalg.lsqr's damped
    least-squares (min ||J dx + r||^2 + lam*||dx||^2, LinearOperator built
    from those jvp/vjp matvecs - never the dense O(norb^2) Jacobian). This is
    the fix for a measured failure mode of lbfgs_solve/solver="error_gradient":
    on a 30-site (60-orbital) biased Hubbard chain it did not reach
    maxerror=1e-6 even after 3000 L-BFGS-B iterations (residual stuck around
    5e-3 to 1.5e-2, including with various amounts of linear-mixing warm start
    first) while this solver converges in a handful of outer iterations, like
    solver="newton_krylov" already does on the same case - unsurprising, since
    L-BFGS-B only ever sees the scalar gradient of the squared residual and
    discards the residual VECTOR's own structure, whereas both newton_krylov
    and this solver use that structure directly (jvp of r, not of a scalar).

    Differs from newton_krylov_solve (plain Newton + GMRES on the SQUARE
    system (J-I)dx=-r) in exactly the way LM differs from Newton generally:
    the damping term lam*I (equivalently, lsqr's Tikhonov `damp`) keeps the
    subproblem well-posed even when J is singular/near-singular - e.g. the
    unbroken-continuous-spin-symmetry marginal direction documented in this
    module's own WARNING, where newton_krylov_solve's GMRES on a literally
    singular operator can fail to find any improving step at all. lsqr also
    works directly on J (not the squared, worse-conditioned J^T J a
    hand-rolled CG-on-normal-equations version would form), which is the
    standard numerically-preferred way to solve a damped least-squares
    subproblem matrix-free.

    Levenberg-Marquardt's classic adaptive-damping accept/reject loop: try a
    step with the current lam; if it decreases ||r||^2, accept it and relax
    lam (trust the linear model more); if not, grow lam (fall back toward
    steepest-descent/more-regularized behavior) and retry the SAME point
    without advancing the outer iteration count. max_inner_tries caps that
    retry loop the same way newton_solve/newton_krylov_solve's max_backtrack
    caps their own step-halving retries.

    lsqr_iter_lim caps each inner lsqr solve's own iteration count, the same
    role gmres_restart plays for newton_krylov_solve's GMRES - an exact
    solve of the LM subproblem is not needed, only a good search direction
    (standard "inexact/truncated Newton" practice), and leaving it unbounded
    is expensive: measured on a 30-site (60-orbital) biased Hubbard chain,
    lsqr_iter_lim=None took 29s for 8 outer iterations vs. 8.4s for the same
    8 iterations at lsqr_iter_lim=20 - same outer-iteration convergence,
    ~3.5x less wall time, because each lsqr call was doing far more inner
    work than the resulting step direction actually needed."""
    from scipy.sparse.linalg import lsqr, LinearOperator
    n = x0.shape[0]
    r_fn = jax.jit(lambda x: step_vec(x) - x)
    jvp_fn = jax.jit(lambda x, v: jax.jvp(r_fn, (x,), (v,))[1])

    def vjp_fn(x, u):
        _, vjp = jax.vjp(r_fn, x)
        return vjp(u)[0]
    vjp_fn = jax.jit(vjp_fn)

    x = x0
    r = r_fn(x)
    err = float(jnp.max(jnp.abs(r)))
    m = float(jnp.sum(jnp.abs(r) ** 2))
    lam = lam0
    # jax.vjp's reverse-mode pass through step_vec's complex intermediates
    # (Hamiltonian/eigh/density-matrix arithmetic) onto a real cotangent
    # triggers numpy's ComplexWarning deep in jax's own VJP machinery on
    # every rmatvec call -- benign for the same reason documented in
    # lbfgs_solve's docstring (the discarded part is the expected zero
    # imaginary component of a real-input/real-output map's cotangent), so
    # it is suppressed here the same way, by exact message text rather than
    # a bare category filter
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore",
                message=".*Casting complex values to real.*")
        for ite in range(maxite):
            if verbose > 0:
                print("Levenberg-Marquardt iteration", ite, "error", err,
                        "lambda", lam)
            if err < tol:
                return x, ite, True

            def matvec(v_np):
                return np.array(jvp_fn(x, jnp.asarray(v_np)), copy=True)

            def rmatvec(u_np):
                return np.array(vjp_fn(x, jnp.asarray(u_np)), copy=True)

            Jop = LinearOperator((n, n), matvec=matvec, rmatvec=rmatvec,
                    dtype=np.float64)
            r_np = np.array(r, copy=True)
            accepted = False
            for _ in range(max_inner_tries):
                dx_np = lsqr(Jop, -r_np, damp=np.sqrt(lam),
                        iter_lim=lsqr_iter_lim)[0]
                dx = jnp.asarray(dx_np)
                x_try = x + dx
                r_try = r_fn(x_try)
                m_try = float(jnp.sum(jnp.abs(r_try) ** 2))
                if m_try < m:
                    x, r, m = x_try, r_try, m_try
                    err = float(jnp.max(jnp.abs(r)))
                    lam = max(lam / lam_factor, 1e-12)
                    accepted = True
                    break
                lam *= lam_factor
            if not accepted:
                # no damping level tried improved the residual: stuck, stop early
                return x, ite, err < tol
        return x, maxite, err < tol


def fsolve_solve(step_vec, x0, maxite=2000, tol=1e-8, verbose=0):
    """Solve x = step_vec(x) with scipy.optimize.fsolve (MINPACK hybrj),
    using the exact JAX Jacobian (jax.jacfwd) as fprime. Unlike
    newton_solve's hand-rolled backtracking, MINPACK's Powell hybrid dogleg
    method is a mature trust-region implementation, and may reuse Broyden
    rank-1 updates of the Jacobian between full recomputations instead of
    rebuilding the O(norb^2) x O(norb^2) Jacobian every iteration - compare
    infodict['njev'] to infodict['nfev'] to see whether that is actually
    happening for a given problem size (njev << nfev means yes)."""
    from scipy.optimize import fsolve
    jac_fn = jax.jacfwd(step_vec)
    n = x0.shape[0]
    eye = jnp.eye(n, dtype=x0.dtype)

    def func(x_np):
        return np.asarray(step_vec(jnp.asarray(x_np)) - jnp.asarray(x_np))

    def jac(x_np):
        return np.asarray(jac_fn(jnp.asarray(x_np)) - eye)

    x_sol, infodict, ier, mesg = fsolve(func, np.asarray(x0), fprime=jac,
            full_output=True, maxfev=maxite, xtol=tol)
    if verbose > 0:
        print("fsolve: nfev", infodict["nfev"], "njev",
                infodict.get("njev"), "ier", ier, mesg)
    return jnp.asarray(x_sol), infodict["nfev"], ier == 1


def fixed_point_solve(step_fn, x0, mu, dirs, n, mix=0.1, maxite=2000, tol=1e-5,
        verbose=0, callback_mf=None):
    """Linear-mixing fixed point, mirrors densitydensity.generic_densitydensity
    with solver="plain". mu is ignored by step_fn (in favor of an internally
    computed, filling-derived value) when step_fn was built with
    n_occ_total set - see build_step_function. callback_mf, if given, is
    applied on concrete numpy arrays each iteration (e.g.
    mfconstrains.enforce_constrains) - it cannot be used inside a
    jax.jacfwd trace, which is why solver="newton" rejects it."""
    x = x0
    cur_mu = mu
    for ite in range(maxite):
        xnew, dm, es, occ, cur_mu = step_fn(x, cur_mu)
        if callback_mf is not None:
            mfnew_np = {d: np.asarray(m) for d, m in
                    unflatten_mf(xnew, dirs, n).items()}
            mfnew_np = callback_mf(mfnew_np)
            xnew = flatten_mf({d: jnp.asarray(mfnew_np[d], dtype=jnp.complex128)
                for d in dirs}, dirs)
        diff = diff_mf_vec(xnew, x)
        x = (1 - mix) * x + mix * xnew
        if verbose > 0:
            print("ERROR in the SCF cycle", ite, diff)
        if diff < tol:
            return x, cur_mu, ite, True
    return x, cur_mu, maxite, False


def lbfgs_solve(loss_fn, x0, maxite=2000, tol=1e-5, verbose=0, gtol=None):
    """Minimize loss_fn (any JAX-differentiable scalar function of x) with
    scipy.optimize.minimize's L-BFGS-B, using jax.grad (via
    jax.value_and_grad) for the exact gradient.

    Reachable directly via generic_densitydensity_jax's own solver="lbfgs"
    (Vinteraction's use_jax=True path) with loss_fn(x) =
    sum((step_vec(x)-x)**2), the squared SCF residual -- NOT a physical
    free-energy functional. vjinteraction_jax's solver="error_gradient" used
    to dispatch here too but now uses levenberg_marquardt_solve instead
    (see that function's docstring and vjinteraction_jax's module docstring
    for why: this scalar-loss-only approach was found to stall on larger
    systems, where levenberg_marquardt_solve's use of the residual's actual
    Jacobian-vector products, not just the scalar loss gradient, does not).
    See vjinteraction_jax's module docstring for why minimizing the actual
    mean-field
    free energy directly (via jax.grad of a grand-potential functional) was
    tried first and abandoned after empirically finding the physical SCF
    solution is generically a *saddle point* of that functional, not a
    minimum -- L-BFGS-B reliably converged to spurious, non-self-consistent
    points instead, even from very close to the true solution. Minimizing
    the squared residual instead has no such issue, since it is a sum of
    squares whose global minimum (value 0) sits exactly at every SCF fixed
    point, by construction -- any x this converges to with a near-zero loss
    IS (to that tolerance) self-consistent, not just a stationary point of
    an unrelated functional.

    This still gets the intended scaling benefit over newton_solve/
    fsolve_solve: a jax.grad of a scalar costs about one extra pass through
    step_vec's own eigh-based computation (structurally like
    newton_krylov_solve's jax.jvp), not O(norb^2) forward passes to build a
    dense Jacobian -- so this should scale per-iteration like
    fixed_point_solve/newton_krylov_solve.

    L-BFGS-B's own gtol/success criteria measure stationarity of loss_fn
    (gradient norm), which is a necessary but not sufficient proxy for the
    SCF-residual sense of "converged" every other solver in this file uses
    (max(|step(x)-x|) < tol) -- e.g. a nonlinear least-squares loss like
    this can in principle have its own local minima with loss>0. The caller
    (solve_scf's solver="lbfgs" branch) must recompute the actual residual
    from the returned x itself and derive scf.converged from that, exactly
    as it already computes final_mu that way.

    gtol defaults to tol (the same value the caller passes as its own
    maxerror), a reasonable default tying the two together without a
    dedicated tuning knob -- add a separate gtol= passthrough later only if
    that default proves insufficient in practice."""
    from scipy.optimize import minimize
    if gtol is None:
        gtol = tol
    val_and_grad = jax.jit(jax.value_and_grad(loss_fn))

    def func(x_np):
        v, g = val_and_grad(jnp.asarray(x_np))
        # np.array(..., copy=True) rather than np.asarray: a numpy view of a
        # jax array's buffer can come back read-only, and scipy's L-BFGS-B
        # is not contractually guaranteed never to write into the gradient
        # array it receives in place (see newton_krylov_solve.gmres_solve's
        # matvec, which hits this exact issue with scipy's gmres)
        return float(v), np.array(g, dtype=np.float64, copy=True)

    # ftol=0 (scipy's default is a loose ~2.22e-9 relative-function-reduction
    # criterion) disables L-BFGS-B's OWN early-stopping-on-plateau check, so
    # it keeps iterating down to gtol -- with the default ftol, a residual
    # loss already small in absolute terms (e.g. ~1e-10 for a ~1e-5 residual)
    # can plateau in *relative* terms well before gtol is reached, capping
    # the achievable residual short of the caller's requested maxerror
    #
    # reverse-mode jax.grad through loss_fn's complex intermediates (the
    # underlying step_vec's Hamiltonian/eigh/density-matrix arithmetic) onto
    # a real scalar triggers numpy's ComplexWarning ("Casting complex values
    # to real discards the imaginary part") deep in jax's own VJP machinery
    # on every func() call above -- benign (the discarded part is the
    # expected zero imaginary component of a real-input/real-output map's
    # cotangent; confirmed by tests/scf/test_vjinteraction_jax.py's
    # solver="lbfgs" tests matching solver="newton" to <1e-6), so it is
    # suppressed once here around the whole optimization rather than
    # per-call, and by exact message text (not a bare category filter) so an
    # unrelated ComplexWarning from a genuine bug elsewhere would still show
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore",
                message=".*Casting complex values to real.*")
        res = minimize(func, np.asarray(x0), jac=True, method="L-BFGS-B",
                options=dict(maxiter=maxite, gtol=gtol, ftol=0.0))
    if verbose > 0:
        print("L-BFGS-B:", res.message, "nit", res.nit, "nfev", res.nfev)
    return jnp.asarray(res.x), int(res.nit)


def solve_scf(step_jit, x0, mu, dirs, n, solver, maxite, maxerror, mix,
        verbose, gmres_tol, gmres_restart, callback_mf=None):
    """Shared solver dispatch for generic_densitydensity_jax's and
    vjinteraction_jax.generic_vjinteraction_jax's use_jax=True paths --
    drives x0 to a fixed point of step_jit via whichever solver= was
    requested ("newton"/"fsolve"/"newton_krylov"/"fixed_point"/"lbfgs"/
    "levenberg_marquardt"/"broyden_mixing"),
    then evaluates step_jit exactly ONCE more at the converged x to get
    everything callers need: final_mu, xfinal/dm/es/occ, and (solver=
    "lbfgs" only, which has no residual-based convergence notion of its
    own) the SCF-residual convergence check. Previously each of those was
    computed via its own separate step_jit call in each of the two
    call sites (up to 3 redundant full Bloch-build+batched-eigh passes for
    solver="lbfgs" alone) -- see this function's git history for the
    per-branch version this replaced.

    Passing the ORIGINAL mu (not each solver's own possibly-different
    "final" mu) into that single trailing step_jit call is exactly correct:
    when a filling target is active (step_jit was built with n_occ_total
    set) step()'s mu_eff ignores its mu argument entirely and resolves it
    from n_occ_total instead, and for a fixed mu every solver's converged x
    already has mu_eff == mu by construction (fixed_point_solve's own
    tracked mu never drifts from the constant mu it started with in that
    case either).

    Returns (x, final_mu, ite, converged, dm, es, occ). callback_mf (only
    meaningful for solver="fixed_point"; applied on concrete numpy arrays
    each iteration) raises NotImplementedError for every other solver, which
    need x to stay a pure jax-traced value throughout."""
    if solver in ("newton", "fsolve", "newton_krylov", "lbfgs",
            "levenberg_marquardt", "broyden_mixing"):
        if callback_mf is not None:
            raise NotImplementedError("solver=%r cannot apply "
                    "callback_mf/constrains (they need concrete numpy "
                    "arrays each iteration, incompatible with jax tracing); "
                    "use solver=\"fixed_point\" instead" % (solver,))
        # not wrapped in jax.jit here: step_jit already dispatches into the
        # cached, once-jitted core built by build_step_function/
        # _get_step_core (see there) -- an extra jax.jit around this thin
        # closure would just add its own fresh-per-call compile for no
        # benefit, since the actual physics computation is already
        # compiled and shared. jax.jacfwd/jax.jvp/jax.vjp/jax.grad (used
        # by the solvers below) all work fine tracing through a plain
        # Python function that calls an already-jitted one.
        step_vec = lambda x: step_jit(x, mu)[0]
    if solver == "newton":
        x, ite, converged = newton_solve(step_vec, x0, maxite=maxite,
                tol=maxerror, verbose=verbose)
    elif solver == "fsolve":
        x, ite, converged = fsolve_solve(step_vec, x0, maxite=maxite,
                tol=maxerror, verbose=verbose)
    elif solver == "newton_krylov":
        x, ite, converged = newton_krylov_solve(step_vec, x0, maxite=maxite,
                tol=maxerror, verbose=verbose, gmres_tol=gmres_tol,
                gmres_restart=gmres_restart)
    elif solver == "fixed_point":
        # the mu fixed_point_solve itself tracks/returns is superseded by
        # the fresh step_jit(x, mu) call below (see this function's
        # docstring), so it is not needed here
        x, _, ite, converged = fixed_point_solve(step_jit, x0, mu, dirs, n,
                mix=mix, maxite=maxite, tol=maxerror, verbose=verbose,
                callback_mf=callback_mf)
    elif solver == "lbfgs":
        residual_loss = jax.jit(lambda x: jnp.sum((step_vec(x) - x) ** 2))
        x, ite = lbfgs_solve(residual_loss, x0, maxite=maxite, tol=maxerror,
                verbose=verbose)
        converged = None  # resolved below, once xfinal is available
    elif solver == "levenberg_marquardt":
        x, ite, converged = levenberg_marquardt_solve(step_vec, x0,
                maxite=maxite, tol=maxerror, verbose=verbose)
    elif solver == "broyden_mixing":
        # unlike newton/fsolve/newton_krylov/lbfgs, this solver never needs
        # x to stay a traced jax value between iterations (no jacfwd/jvp/grad
        # of step_vec involved -- it only ever calls step_vec(x) as a black
        # box), so grouping it with those above (for the callback_mf check)
        # is a simplicity choice, not a technical requirement the way it is
        # for them; step_vec still works fine here since a jax.jit function
        # accepts plain numpy input and converts internally
        from .broydenmixing import broyden_mixing_solve
        x, ite, converged = broyden_mixing_solve(step_vec, x0, maxite=maxite,
                tol=maxerror, verbose=verbose)
        x = jnp.asarray(x)
    else:
        raise ValueError("unrecognised solver for use_jax=True: %r" % (solver,))

    xfinal, dm, es, occ, final_mu = step_jit(x, mu)
    final_mu = float(final_mu)
    if solver == "lbfgs":
        # scf.converged still means the same thing here as for every other
        # solver -- the actual SCF residual, not L-BFGS-B's own gradient-norm
        # stopping criterion (see lbfgs_solve's docstring)
        converged = bool(jnp.max(jnp.abs(xfinal - x)) < maxerror)
    return x, final_mu, ite, bool(converged), dm, es, occ


def generic_densitydensity_jax(h0, mf=None, v=None, nk=8, mu=0.0,
        filling=None, T=None, mix=0.1, maxerror=1e-5, maxite=2000,
        solver="newton", compute_dd=True, compute_cross=True,
        add_dagger=True, verbose=0, callback_mf=None,
        gmres_tol=1e-6, gmres_restart=20, **kwargs):
    """JAX-differentiable analogue of densitydensity.generic_densitydensity.
    maxite defaults to 2000 (vs. the numpy engine's unbounded default) since
    plain linear mixing from a cold/random start can need many hundreds of
    iterations at tight tolerance - see the "fixed_point" cases in the
    benchmark. solver="newton" converges in a handful of iterations when it
    converges at all, so this default is generous there too, never a
    bottleneck. solver="lbfgs" minimizes ||step(x)-x||^2 with jax.grad +
    scipy's L-BFGS-B instead of root-finding step(x)=x -- see
    vjinteraction_jax's module docstring for the "solver='lbfgs'" section
    (written for VJinteraction, but solve_scf/lbfgs_solve are the same
    generic machinery used here). solver="broyden_mixing" is a black-box
    mixing scheme (regularized, limited-memory multisecant Broyden mixing,
    arXiv:0801.3098) rather than a root-finder/gradient method -- see
    broydenmixing.py's module docstring."""
    if h0.has_eh:
        raise NotImplementedError("use_jax=True does not support the "
                "anomalous/BdG mean field yet; use the default (numpy) engine")
    if solver != "fixed_point" and mix != 0.1:
        # mix only controls solver="fixed_point"'s linear-mixing step -- see
        # vjinteraction_jax.generic_vjinteraction_jax's identical check
        warnings.warn("mix=%r has no effect for solver=%r (only "
                "solver=\"fixed_point\" uses linear mixing)"
                % (mix, solver), stacklevel=2)
    if T is None:
        T = default_T_jax
    elif T <= 0:
        raise ValueError("T=%r is not usable with use_jax=True: occupations "
                "are occ=sigmoid(-(e-mu)/T), so T<=0 (including exactly 0) "
                "divides by a non-positive number and produces NaN/Inf, "
                "unlike the numpy engine's T=0 hard Fermi step -- pass a "
                "small positive T instead (e.g. this module's own default, "
                "default_T_jax=%r)" % (T, default_T_jax))
    h1 = h0.copy()
    h1 = h1.get_dense()
    h1.nk = nk
    hop0 = hamiltonian2dict(h1)  # numpy dict, bare hoppings
    n = hop0[(0, 0, 0)].shape[0]
    dirs = sorted(v.keys())
    if (0, 0, 0) not in dirs:
        dirs = [(0, 0, 0)] + dirs
    dirs_all = sorted(set(hop0.keys()) | set(dirs))
    ks = jnp.asarray(np.array(h1.geometry.get_kmesh(nk=nk)), dtype=jnp.float64)
    if mf is None:
        rng = np.random.default_rng()
        mf0 = dict()
        for d in dirs:
            mf0[d] = np.exp(1j * rng.random((n, n)))
        mf0[(0, 0, 0)] = mf0[(0, 0, 0)] + mf0[(0, 0, 0)].T.conjugate()
        mf = mf0
    elif isinstance(mf, str):
        from ..meanfield import guess
        mf = guess(h0, mode=mf)
    mf = obj2mf(mf)
    # mf need not cover every direction in dirs (e.g. a nearest-neighbor-only
    # guess like mode="kekule" combined with a longer-range V1+V2
    # interaction): missing directions start at zero, matching the old
    # engine's implicit behavior (MultiHopping addition treats an absent
    # key as a zero contribution)
    zero_n = jnp.zeros((n, n), dtype=jnp.complex128)
    x0 = flatten_mf({d: jnp.asarray(mf[d], dtype=jnp.complex128)
        if d in mf else zero_n for d in dirs}, dirs)
    n_occ_total = None
    if filling is not None:
        n_tot = n * ks.shape[0]
        n_occ_total = int(round(filling * n_tot))
        n_occ_total = min(max(n_occ_total, 1), n_tot - 1)
    # not wrapped in an extra jax.jit: build_step_function's returned
    # closure already dispatches into a cached, once-jitted core shared
    # across every call with this same structural shape -- see
    # build_step_function/_get_step_core's docstrings
    step_jit = build_step_function(hop0, v, ks, dirs, dirs_all, T,
            compute_dd, compute_cross, add_dagger, n_occ_total=n_occ_total)
    x, final_mu, ite, converged, dm, es, occ = solve_scf(step_jit, x0, mu,
            dirs, n, solver, maxite, maxerror, mix, verbose, gmres_tol,
            gmres_restart, callback_mf=callback_mf)
    mf_final = unflatten_mf(x, dirs, n)
    dm_np = {d: np.asarray(dm[d]) for d in dirs}
    mf_np = {d: np.asarray(mf_final[d]) for d in dirs}
    v_np = {d: np.asarray(v[d]) for d in v}
    hop_final = dict()
    for d in dirs_all:
        m = np.asarray(hop0[d]) if d in hop0 else np.zeros((n, n), dtype=complex)
        if d in mf_np:
            m = m + mf_np[d]
        hop_final[d] = m
    h_final = h1.copy()
    set_hoppings(h_final, hop_final)
    etot_band = float(jnp.sum(occ * es) / ks.shape[0])
    etot = etot_band + get_dc_energy(v_np, dm_np)
    scf = SCF()
    scf.hamiltonian = h_final
    scf.hamiltonian.V = v
    scf.hamiltonian0 = h0
    scf.mf = mf_np
    scf.dm = dm_np
    scf.v = v
    scf.tol = maxerror
    scf.converged = bool(converged)
    if not scf.converged:
        # unconditional (not gated on verbose), matching the numpy engine's
        # own "No convergence has been reached..." print
        print("No convergence has been reached in", ite,
                "iterations (solver=%r), stopping" % (solver,))
    scf.total_energy = etot
    scf.mu = final_mu
    scf.iterations = ite
    if verbose > 1:
        print("##################")
        print("Total energy", etot)
        print("Converged", scf.converged, "in", ite, "iterations")
        print("##################")
    return scf


def densitydensity_jax(h, filling=0.5, mu=None, verbose=0, **kwargs):
    """JAX drop-in for densitydensity.densitydensity"""
    if h.has_eh:
        raise NotImplementedError("use_jax=True does not support the "
                "anomalous/BdG mean field yet; use the default (numpy) engine")
    h = h.get_multicell()
    h = h.get_dense()
    if mu is not None:
        return generic_densitydensity_jax(h, mu=mu, filling=None,
                verbose=verbose, **kwargs)
    else:
        return generic_densitydensity_jax(h, mu=0.0, filling=filling,
                verbose=verbose, solver=kwargs.pop("solver", "fixed_point"),
                **kwargs)
