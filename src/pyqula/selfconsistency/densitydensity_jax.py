# JAX-differentiable counterpart of selfconsistency/densitydensity.py
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
#  - the Newton solver requires a fixed chemical potential mu (grand
#    canonical); a target filling is only supported by solver="fixed_point"
#    since resolving mu(filling) requires a non-differentiable sort/root-find
#  - occupations always use a finite smearing temperature T (default 1e-4)
#    because jnp.linalg.eigh's eigenvector gradient is only well defined
#    away from exact degeneracies; T=0/None silently falls back to the
#    default rather than erroring
from __future__ import annotations
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


def build_step_function(hop0, v, ks, dirs, dirs_all, T,
        compute_dd, compute_cross, add_dagger):
    """Return step(x,mu) -> (xnew, dm, es, occ), the pure-JAX one SCF step"""
    n = hop0[(0, 0, 0)].shape[0]
    ds_arr = jnp.array([list(d) for d in dirs_all], dtype=jnp.float64)
    ms0 = make_bloch_stack(hop0, dirs_all, n)
    v_jnp = {d: jnp.asarray(v[d], dtype=jnp.complex128) for d in v}
    nk = ks.shape[0]
    dir_phase = jnp.array([list(d) for d in dirs], dtype=jnp.float64)  # (nt,3)

    def step(x, mu):
        mf = unflatten_mf(x, dirs, n)
        mats = [ms0[i] + mf[d] if d in mf else ms0[i]
                for i, d in enumerate(dirs_all)]
        ms = jnp.stack(mats)

        def hk(k):
            phases = jnp.exp(1j * 2 * jnp.pi * (ds_arr @ k))
            return jnp.einsum('nij,n->ij', ms, phases)

        hks = jax.vmap(hk)(ks)                      # (nk,n,n)
        es, vs = jnp.linalg.eigh(hks)                # (nk,n), (nk,n,n)
        occ = jax.nn.sigmoid(-(es - mu) / T)         # (nk,n)
        kd = ks @ dir_phase.T                        # (nk,nt)
        phase = jnp.exp(1j * 2 * jnp.pi * kd)         # (nk,nt)
        dm_all = jnp.einsum('kt,kie,ke,kje->tij', phase,
                jnp.conj(vs), occ, vs) / nk           # (nt,n,n)
        dm = {d: dm_all[i] for i, d in enumerate(dirs)}
        mfnew = get_mf_normal_jax(v_jnp, dm, dirs, compute_dd=compute_dd,
                compute_cross=compute_cross, add_dagger=add_dagger)
        xnew = flatten_mf(mfnew, dirs)
        return xnew, dm, es, occ

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


def fixed_point_solve(step_fn, x0, mu, dirs, n, mix=0.1, maxite=2000, tol=1e-5,
        verbose=0, resolve_mu=None, callback_mf=None):
    """Linear-mixing fixed point, mirrors densitydensity.generic_densitydensity
    with solver="plain". resolve_mu(x), if given, recomputes mu each
    iteration (used for a target filling instead of a fixed mu).
    callback_mf, if given, is applied on concrete numpy arrays each
    iteration (e.g. mfconstrains.enforce_constrains) - it cannot be used
    inside a jax.jacfwd trace, which is why solver="newton" rejects it."""
    x = x0
    cur_mu = mu
    for ite in range(maxite):
        if resolve_mu is not None:
            cur_mu = resolve_mu(x)
        xnew, dm, es, occ = step_fn(x, cur_mu)
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


def _numpy_fermi_for_filling(h1, hop0, dirs_all, mf_concrete, filling, nk):
    """Resolve the chemical potential for a target filling. Non-differentiable
    (uses the existing numpy get_fermi4filling path), only usable outside
    a jax trace, i.e. only for solver="fixed_point"."""
    hop = dict()
    for d in dirs_all:
        m = np.asarray(hop0[d]) if d in hop0 else 0.0
        if d in mf_concrete:
            m = m + np.asarray(mf_concrete[d])
        hop[d] = m
    htmp = h1.copy()
    set_hoppings(htmp, hop)
    return htmp.get_fermi4filling(filling, nk=nk)


def generic_densitydensity_jax(h0, mf=None, v=None, nk=8, mu=0.0,
        filling=None, T=None, mix=0.1, maxerror=1e-5, maxite=2000,
        solver="newton", compute_dd=True, compute_cross=True,
        add_dagger=True, verbose=0, callback_mf=None, **kwargs):
    """JAX-differentiable analogue of densitydensity.generic_densitydensity.
    maxite defaults to 2000 (vs. the numpy engine's unbounded default) since
    plain linear mixing from a cold/random start can need many hundreds of
    iterations at tight tolerance - see the "fixed_point" cases in the
    benchmark. solver="newton" converges in a handful of iterations when it
    converges at all, so this default is generous there too, never a
    bottleneck."""
    if h0.has_eh:
        raise NotImplementedError("use_jax=True does not support the "
                "anomalous/BdG mean field yet; use the default (numpy) engine")
    if T is None or T <= 0:
        T = default_T_jax
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
    x0 = flatten_mf({d: jnp.asarray(mf[d], dtype=jnp.complex128)
        for d in dirs}, dirs)
    step = build_step_function(hop0, v, ks, dirs, dirs_all, T,
            compute_dd, compute_cross, add_dagger)
    step_jit = jax.jit(step)
    if solver == "newton":
        if filling is not None:
            raise NotImplementedError("solver=\"newton\" requires a fixed "
                    "mu (grand canonical); pass mu=... instead of "
                    "filling=..., or use solver=\"fixed_point\"")
        if callback_mf is not None:
            raise NotImplementedError("solver=\"newton\" cannot apply "
                    "callback_mf/constrains (they need concrete numpy "
                    "arrays, incompatible with jax.jacfwd tracing); use "
                    "solver=\"fixed_point\" instead")
        step_vec = jax.jit(lambda x: step_jit(x, mu)[0])
        x, ite, converged = newton_solve(step_vec, x0, maxite=maxite,
                tol=maxerror, verbose=verbose)
        final_mu = mu
    elif solver == "fixed_point":
        resolve_mu = None
        if filling is not None:
            def resolve_mu(xc):
                mfc = unflatten_mf(xc, dirs, n)
                return _numpy_fermi_for_filling(h1, hop0, dirs_all, mfc,
                        filling, nk)
        x, final_mu, ite, converged = fixed_point_solve(step_jit, x0, mu,
                dirs, n, mix=mix, maxite=maxite, tol=maxerror,
                verbose=verbose, resolve_mu=resolve_mu,
                callback_mf=callback_mf)
    else:
        raise ValueError("unrecognised solver for use_jax=True: %s" % solver)
    xfinal, dm, es, occ = step_jit(x, final_mu)
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
