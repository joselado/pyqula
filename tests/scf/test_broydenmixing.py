import numpy as np

from pyqula.selfconsistency.broydenmixing import broyden_mixing_solve


def _sloshing_problem():
    """A 2-variable affine fixed-point map F(x) = x - A(x-x*) with a badly
    scaled A (eigenvalues 2.5 and 0.001), the textbook "charge sloshing"
    scenario the paper (arXiv:0801.3098) targets: no single fixed linear-
    mixing factor handles both channels -- mix=1 (plain fixed-point
    iteration) is unstable on the fast channel (Jacobian eigenvalue
    1-2.5=-1.5, |.|>1) and diverges, while any mix small enough to stabilize
    that channel makes the slow channel's convergence (Jacobian eigenvalue
    close to 1) impractically slow."""
    xstar = np.array([0.37, -1.2])
    A = np.diag([2.5, 0.001])

    def F(x):
        return x - A @ (x - xstar)

    x0 = xstar + np.array([1.0, 1.0])
    return F, x0, xstar


def test_plain_full_step_diverges_on_sloshing_problem():
    """Sanity check on the synthetic problem itself: plain fixed-point
    iteration (x_{n+1}=F(x_n), i.e. mix=1) diverges."""
    F, x0, xstar = _sloshing_problem()
    x = x0.copy()
    for _ in range(50):
        x = F(x)
    assert np.linalg.norm(x) > 1e6


def test_moderate_linear_mixing_is_impractically_slow_on_sloshing_problem():
    """Sanity check: a mix stable for the fast channel (mix=0.5) does not
    reach a tight tolerance within a generous iteration budget, because it
    is simultaneously far too aggressive/too timid for the two very
    differently-scaled channels at once."""
    F, x0, xstar = _sloshing_problem()
    x = x0.copy()
    mix = 0.5
    converged = False
    for _ in range(300):
        g = F(x) - x
        if np.linalg.norm(g) < 1e-8:
            converged = True
            break
        x = x + mix * g
    assert not converged


def test_broyden_mixing_converges_on_sloshing_problem():
    """The paper's core claim: robust multisecant Broyden mixing converges
    on a problem that defeats any single fixed linear-mixing factor, and
    does so in a small number of iterations (no per-channel tuning)."""
    F, x0, xstar = _sloshing_problem()
    x, ite, converged = broyden_mixing_solve(F, x0, maxite=100, tol=1e-8)
    assert converged
    assert ite < 50
    assert np.max(np.abs(x - xstar)) < 1e-6


def test_broyden_mixing_converges_on_well_conditioned_problem():
    """A well-conditioned linear map (single scale, no sloshing) should also
    converge cleanly -- not just the pathological case above."""
    rng = np.random.default_rng(0)
    n = 5
    xstar = rng.random(n) - 0.5
    A = np.diag(rng.uniform(0.3, 1.2, size=n))

    def F(x):
        return x - A @ (x - xstar)

    x0 = xstar + rng.uniform(-1, 1, size=n)
    x, ite, converged = broyden_mixing_solve(F, x0, maxite=100, tol=1e-10)
    assert converged
    assert np.max(np.abs(x - xstar)) < 1e-8


def test_broyden_mixing_reports_converged_false_within_maxite():
    """A deliberately tiny maxite must stop early and report converged=False,
    not raise or silently return a bogus success flag."""
    F, x0, xstar = _sloshing_problem()
    x, ite, converged = broyden_mixing_solve(F, x0, maxite=2, tol=1e-12)
    assert not converged
    assert ite == 2


def test_broyden_mixing_returns_immediately_for_converged_initial_guess():
    """If x0 is already a fixed point, no Pratt step or iteration is needed."""
    F, x0, xstar = _sloshing_problem()
    x, ite, converged = broyden_mixing_solve(F, xstar.copy(), maxite=100, tol=1e-8)
    assert converged
    assert ite == 0
    assert np.max(np.abs(x - xstar)) < 1e-12


def test_broyden_mixing_converges_from_cold_start_on_small_hubbard_flake():
    """Regression test for the linear-mixing warm-up phase (see this
    module's own module docstring for the benchmark behind it): this exact
    real (non-synthetic) problem -- an 8-atom Lieb flake, biased spinful
    Hubbard U, cold random initial guess -- reliably failed to converge
    within 1500 iterations when the multisecant phase started directly from
    the cold guess (the paper's literal algorithm); with the warm-up phase
    it must now converge reliably."""
    from pyqula import islands
    from pyqula.selfconsistency.densitydensity import Vinteraction

    g = islands.get_geometry(name="lieb", n=1.5)
    h0 = g.get_hamiltonian(has_spin=True)
    rng = np.random.default_rng(123)
    v = rng.random(3) - 0.5
    v = 0.3 * v / np.linalg.norm(v)
    h0.add_exchange(v)

    for seed in range(3):
        np.random.seed(seed)
        scf = Vinteraction(h0.copy(), U=3.0, filling=0.5, nk=1, mf=None,
                load_mf=False, solver="broyden_mixing", maxerror=1e-6,
                maxite=1500, verbose=0)
        assert scf.converged, f"seed={seed} did not converge"
