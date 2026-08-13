# Regularized, preconditioned, limited-memory multisecant Broyden mixing,
# following Marks & Luke, "Robust Mixing for Ab-Initio Quantum Mechanical
# Calculations", arXiv:0801.3098 (the boxed "Algorithm" in that paper's
# Section III.E is the exact recipe implemented below).
#
# This is NOT the same thing as scipy's broyden1/broyden2 (already wired in
# elsewhere in this package, e.g. densitydensity.py's solver="broyden1",
# meanfield.py's broyden_solver): those are single-secant, full-memory,
# unregularized root-finders on x-F(x)=0. This module instead treats the
# last `m` SCF steps as SIMULTANEOUS secant conditions (a multisecant form of
# Broyden's second/"bad" method, MSBB), Tikhonov-regularizes the resulting
# least-squares solve, normalizes columns so the regularization parameter is
# scale-independent, and adaptively bounds the step length -- the combination
# the paper credits for converging on cases (charge sloshing, near-singular
# Jacobians) that defeat plain Broyden/Pulay mixing.
#
# Scope relative to the paper: the two-block preconditioner Omega (eq. 45-47
# there) rescales the step based on a physical split specific to LAPW codes
# (interstitial vs. muffin-tin electrons) that has no analog in pyqula's
# mean-field vectors, and is deliberately NOT implemented here -- everything
# else (Tikhonov regularization alpha, column-normalization Psi, and the
# adaptive step-size control sigma_n) is.
#
# DEVIATION FROM THE PAPER'S LITERAL ALGORITHM BOX, found necessary by
# benchmarking (not a stylistic choice): the paper seeds its main multisecant
# loop with a single damped "Pratt step" (eq. \rho_1 = \rho_0 +
# lam*(F(\rho_0)-\rho_0)) starting from a cold, possibly-far-from-solution
# guess. On pyqula's small (few-hundred-to-few-thousand-unknown) mean-field
# SCF problems -- much lower-dimensional than the paper's ~10^4-unknown DFT
# densities -- starting the multisecant phase that far out was measured to
# regularly fail: the adaptive step-size control sigma_n can lock into an
# overly conservative value early on (the predicted-step norm |p_n| fluctuates
# erratically iteration-to-iteration on these smaller/less-DFT-like problems,
# collapsing sigma_n's cap R*|p_n|/|g_n|), and its only recovery mechanism
# (bounded to 2x growth per iteration, eq. 51) is too slow to escape once a
# tiny step barely moves the residual -- a self-reinforcing stall. Measured
# on 5 finite flakes (5-13 atoms, both a spinless charge-order and a biased
# spinful Hubbard problem, 3 random guesses each): starting the multisecant
# phase directly on a cold guess failed to converge in 1500 iterations on
# several of these (e.g. 0/3 seeds for one Hubbard case), while simply
# running plain linear mixing (mix=lam, the same damping factor as the
# paper's own Pratt step, just repeated instead of applied once) until the
# residual drops below `warmup_tol` before switching to the multisecant phase
# converged on EVERY case tested, and did so in 2-10x fewer step_vec
# evaluations than plain linear mixing alone at any fixed mix. This is a
# standard practical pattern in the SCF-mixing literature (start with a few
# safe linear-mixing steps before turning on Broyden/Pulay/DIIS acceleration)
# -- broyden_mixing_solve below does it automatically, using the existing
# `lam` parameter as the warm-up's mixing factor (so no new "how aggressive"
# knob is introduced) and switching once the residual falls below
# `warmup_tol` (default 1e-2).
#
# Only needs black-box evaluations of step_vec(x) -> F(x) (no Jacobian, no
# autodiff), so one pure-numpy implementation serves both of this package's
# SCF solver entry points: densitydensity_jax.py's solve_scf dispatcher
# (jnp arrays converted to/from numpy at that call site) and
# densitydensity.py's plain-numpy solver dispatch (via its existing
# fsol(x)=x-F(x) residual convention). x must already be a flat REAL vector,
# exactly like every other array-based solver wired into those two
# dispatchers (densitydensity_jax.flatten_mf splits real/imag into a real
# vector before any solver sees x; densitydensity.get_mf2array does the same
# for the plain-numpy path) -- this module has no complex-array handling of
# its own.
from __future__ import annotations
import numpy as np


def broyden_mixing_solve(step_vec, x0, maxite=500, tol=1e-8, m=8,
        alpha=1e-4, R=0.1, sigma_bar=0.15, sigma0=None, lam=0.1,
        warmup_tol=1e-2, verbose=0):
    """Solve x = step_vec(x) with the regularized, limited-memory multisecant
    Broyden mixing of Marks & Luke (arXiv:0801.3098), Omega dropped, preceded
    by a plain-linear-mixing warm-up phase -- see this module's docstring for
    the scope note and the benchmark data behind the warm-up phase (a
    deviation from the paper's own literal algorithm, needed for reliability
    on pyqula-sized problems). Returns (x, iterations, converged), matching
    the (x, ite, converged) convention of every other solver in
    densitydensity_jax.py; `iterations` counts step_vec evaluations across
    BOTH phases (warm-up + multisecant), sharing a single `maxite` budget.

    step_vec(x) -> x_new is treated as the SCF map F; internally the residual
    is g(x) = step_vec(x) - x. x0 must be a flat real numpy array (or
    anything np.asarray converts to one, e.g. a jax array from the
    jax-engine call site -- see the module docstring).

    warmup_tol: residual norm threshold below which the algorithm switches
    from the warm-up (plain linear mixing, x_new = x + lam*g) to the
    multisecant phase. m: number of previous SCF steps kept as simultaneous
    secant pairs (eq. "m=min(n,8)" in the paper's Algorithm box; limited
    memory, so cost per iteration is a small m-by-m linear solve, independent
    of len(x0)). alpha: Tikhonov regularization strength for the
    (Y^T Y + alpha*I) solve, paper's suggested range 1e-6 to 1e-4. R: caps
    the unpredicted step relative to the predicted one
    (sigma_n <= R*|p_n|/|g_n|), paper's suggested range 0.05 (hard problems)
    to 0.15 (easy ones). sigma_bar: hard upper bound on the step-size scale
    sigma_n, paper's suggested range 0.1-0.2. sigma0/lam: initial step-size
    scale and damping factor -- lam doubles as both the warm-up phase's
    linear-mixing factor and the damping of the single extra "Pratt step"
    (eq. \\rho_1 = \\rho_0 + lam*(F(\\rho_0)-\\rho_0)) that seeds the
    multisecant phase's secant memory once warm-up ends. sigma0 defaults to
    0.2*sigma_bar (a conservative first guess -- the paper's own eq. for
    sigma_0 is tailored to an LAPW-specific diagnostic (dQ/dPW/dRMT) that has
    no analog here).
    """
    x = np.asarray(x0, dtype=float)
    if sigma0 is None:
        sigma0 = 0.2 * sigma_bar
    g = np.asarray(step_vec(x), dtype=float) - x
    gnorm = float(np.linalg.norm(g))
    if verbose:
        print("broyden_mixing iteration 0 (initial guess), residual", gnorm)
    if gnorm < tol:
        return x, 0, True

    # warm-up phase: plain linear mixing until the residual is small enough
    # that the multisecant phase below starts in a well-behaved regime --
    # see the module docstring for why this was added (measured necessary
    # for reliable convergence on pyqula-sized problems)
    ite = 0
    while gnorm > warmup_tol and ite < maxite:
        x = x + lam * g
        ite += 1
        g = np.asarray(step_vec(x), dtype=float) - x
        gnorm = float(np.linalg.norm(g))
        if verbose:
            print("broyden_mixing warmup iteration", ite, "residual", gnorm)
        if gnorm < tol:
            return x, ite, True
    if ite >= maxite:
        return x, ite, gnorm < tol

    # step 0 of the paper's Algorithm box: a single damped Pratt step,
    # seeding one (x,g) pair into the secant buffer before the main loop
    buf_x = [x]
    buf_g = [g]
    x = x + lam * g
    ite += 1
    sigma = sigma0
    gprev_norm = gnorm

    while ite < maxite:
        g = np.asarray(step_vec(x), dtype=float) - x
        gnorm = float(np.linalg.norm(g))
        ite += 1
        if verbose:
            print("broyden_mixing iteration", ite, "residual", gnorm,
                    "sigma", sigma)
        if gnorm < tol:
            return x, ite, True

        # centered on the CURRENT point (eq. 32 in the paper -- preferred
        # over centering on the previous step, eq. 20), recomputed fresh
        # each iteration since buf_x/buf_g store the raw historical points
        S = np.stack([xb - x for xb in buf_x], axis=1)
        Y = np.stack([gb - g for gb in buf_g], axis=1)
        # drop near-stagnant columns (||y_j|| ~ 0): they carry no secant
        # information and would blow up the column-normalization Psi below
        ynorm = np.linalg.norm(Y, axis=0)
        keep = ynorm > 1e-12 * max(gnorm, 1.0)
        S, Y, ynorm = S[:, keep], Y[:, keep], ynorm[keep]

        if Y.shape[1] > 0:
            psi = 1.0 / ynorm                      # eq. 39 (Psi, column-normalization)
            d = Y.T @ g
            YtY = Y.T @ Y
            # A_n g_n = Psi (Psi Y^T Y Psi + alpha*I)^-1 Psi Y^T g_n, solved
            # as a small m-by-m system (eq. 41 specialized to MSBB, Omega
            # dropped) rather than by forming/inverting A_n explicitly
            W = (psi[:, None] * psi[None, :]) * YtY + alpha * np.eye(Y.shape[1])
            t = np.linalg.solve(W, psi * d)
            c = psi * t                            # = A_n g_n
            p = -S @ c                              # predicted step, eq. 33-35
            unpred_dir = g - Y @ c                  # (I - Y A_n) g_n
            sigma_cap = R * float(np.linalg.norm(p)) / gnorm
        else:
            # no usable secant data yet: fall back to a plain scaled
            # residual step (equivalent to H_0=sigma*I with no memory) --
            # not itself a case the paper's numbering reaches (its main loop
            # always has >=1 secant pair from the seeded Pratt step), but a
            # defensive fallback if every buffered column happens to stagnate
            p = np.zeros_like(x)
            unpred_dir = g
            sigma_cap = sigma_bar

        # adaptive step-size control (eq. 51 and the paper's Algorithm box
        # eq. for sigma_n): bounded relative to the previous scale, capped by
        # both the predicted/unpredicted balance and a hard ceiling
        sigma_tilde = sigma * np.clip(gprev_norm / gnorm, 0.5, 2.0)
        sigma = min(sigma_tilde, sigma_cap, sigma_bar)
        # defensive floor: sigma_cap can legitimately be very small (a tiny
        # predicted step is a real signal to be cautious), but not exactly
        # zero without stalling the iteration entirely
        sigma = max(sigma, 1e-3 * sigma_bar)

        u = -sigma * unpred_dir
        x_new = x + u + p

        buf_x.append(x)
        buf_g.append(g)
        if len(buf_x) > m:
            buf_x.pop(0)
            buf_g.pop(0)
        gprev_norm = gnorm
        x = x_new

    g = np.asarray(step_vec(x), dtype=float) - x
    return x, ite, float(np.linalg.norm(g)) < tol
