import numpy as np
import pytest

jax = pytest.importorskip("jax")

from pyqula import geometry
from pyqula.meanfield import Vinteraction
from testutils import SCF_MAXERROR


def _biased_hamiltonian_and_guess(h0, seed, bias=.4):
    """Bias the *Hamiltonian itself* (not just the mean-field guess) along a
    random direction, exactly as test_rotational_symmetry.py does. A raw
    random mf/mu seed leaves the SU(2) spin-rotation symmetry of the Hubbard
    mean field unbroken, so the fixed point is a whole continuous manifold
    (marginal direction) and linear mixing (and Newton, whose Jacobian is
    then singular along that direction) converges only very slowly or not at
    all. Biasing the Hamiltonian picks an isolated fixed point. Newton needs
    a firmer bias than plain mixing does (0.8 vs 0.4) since it explicitly
    inverts the Jacobian, which is only mildly conditioned near a
    weakly-broken continuous symmetry even with the lstsq fallback."""
    rng = np.random.default_rng(seed)
    v = rng.random(3) - .5
    v = 2 * v / np.sqrt(v.dot(v))
    mf = h0.copy()
    mf.add_exchange([v, -v])  # initial guess
    h1 = h0.copy()
    h1.add_exchange(bias * v)  # bias, breaks the marginal direction
    return h1, mf


def test_densitydensity_jax_fixed_point_matches_numpy_engine():
    """With the same starting mean field, same mixing, and same smearing
    temperature, the jax fixed-point engine (solver="fixed_point") runs the
    same linear-mixing math as the numpy engine, so both must converge to
    the same total energy and mean field. mix=0.8 is used because at very
    tight tolerance plain linear mixing is slow (matching the numpy engine's
    own behaviour: e.g. the default mix=0.1 needs O(1e3) iterations to reach
    1e-8 on this system) - a larger mix converges both engines in a handful
    of iterations without changing which physics is being solved."""
    g = geometry.bichain()
    h0 = g.get_hamiltonian()
    T = 1e-4  # match explicitly: the numpy engine defaults to T=1e-7
    h1, mf0 = _biased_hamiltonian_and_guess(h0, seed=0)

    scf_old = Vinteraction(h1.copy(), nk=20, U=2., mf=mf0.copy(),
            maxerror=1e-6, mix=0.8, T=T, verbose=0)
    scf_new = Vinteraction(h1.copy(), nk=20, U=2., mf=mf0.copy(),
            maxerror=1e-6, mix=0.8, T=T, verbose=0,
            use_jax=True, solver="fixed_point")

    assert scf_old.converged and scf_new.converged
    assert abs(scf_old.total_energy - scf_new.total_energy) < 1e-4
    diff = np.abs(scf_old.mf[(0, 0, 0)] - scf_new.mf[(0, 0, 0)])
    assert np.max(diff) < 1e-3


def _total_energy_newton_random_direction(h0, seed):
    h1, mf = _biased_hamiltonian_and_guess(h0, seed, bias=.8)
    scf = Vinteraction(h1, nk=20, mu=0.0, U=2., mf=mf,
            maxerror=1e-8, verbose=0, use_jax=True, solver="newton")
    assert scf.converged
    return scf.total_energy


def test_densitydensity_jax_newton_is_rotationally_invariant():
    """Same physical invariant as test_rotational_symmetry.py, but exercised
    through the new jax Newton solver: the converged total energy must not
    depend on the (arbitrary) direction of the initial exchange field."""
    g = geometry.bichain()
    h0 = g.get_hamiltonian()
    etots = np.array([_total_energy_newton_random_direction(h0, seed)
        for seed in range(4)])
    diff = etots - np.mean(etots)
    assert np.max(np.abs(diff)) < 1e-6, \
        f"jax Newton SCF total energy is not rotationally invariant: {diff}"


def test_densitydensity_jax_fsolve_matches_newton():
    """solver="fsolve" (scipy.optimize.fsolve/MINPACK hybrj, using the same
    jax.jacfwd Jacobian as fprime) is an alternative globalization strategy
    to the hand-rolled backtracking Newton solver - it must converge to the
    same physics."""
    g = geometry.bichain()
    h0 = g.get_hamiltonian()
    h1, mf = _biased_hamiltonian_and_guess(h0, seed=0, bias=.8)

    scf_newton = Vinteraction(h1.copy(), nk=20, mu=0.0, U=2., mf=mf.copy(),
            maxerror=1e-8, verbose=0, use_jax=True, solver="newton")
    scf_fsolve = Vinteraction(h1.copy(), nk=20, mu=0.0, U=2., mf=mf.copy(),
            maxerror=1e-8, verbose=0, use_jax=True, solver="fsolve")

    assert scf_newton.converged and scf_fsolve.converged
    assert abs(scf_newton.total_energy - scf_fsolve.total_energy) < 1e-6


def test_densitydensity_jax_newton_krylov_matches_newton():
    """solver="newton_krylov" solves the same Newton step (J_step - I) dx =
    -r with matrix-free GMRES (jax.jvp Jacobian-vector products) instead of
    forming the dense jax.jacfwd Jacobian - the whole point is that it
    scales to much larger systems, but it must still converge to the same
    physics as solver="newton" on a case both can handle.

    bias=1.2, not the 0.8 used elsewhere in this file: at 0.8 the GMRES
    trajectory (unlike solver="newton"'s dense-Jacobian one, which takes a
    different path from the same start) can land exactly on the SU(2)
    marginal direction after a couple of accepted steps, where jax.jvp of
    jnp.linalg.eigh is undefined (NaN for every probe direction, not just
    an unlucky Krylov one - verified directly) and GMRES breaks down; see
    the WARNING in this module's own header docstring for the general
    phenomenon. 1.2 was checked to converge both solvers cleanly (energies
    agreeing to ~1e-15) across 20 random seeds, vs. several failures at
    0.8-1.0."""
    g = geometry.bichain()
    h0 = g.get_hamiltonian()
    h1, mf = _biased_hamiltonian_and_guess(h0, seed=0, bias=1.2)

    scf_newton = Vinteraction(h1.copy(), nk=20, mu=0.0, U=2., mf=mf.copy(),
            maxerror=1e-8, verbose=0, use_jax=True, solver="newton")
    scf_nk = Vinteraction(h1.copy(), nk=20, mu=0.0, U=2., mf=mf.copy(),
            maxerror=1e-8, verbose=0, use_jax=True, solver="newton_krylov")

    assert scf_newton.converged and scf_nk.converged
    assert abs(scf_newton.total_energy - scf_nk.total_energy) < 1e-6


def test_densitydensity_jax_lbfgs_matches_newton():
    """solver="lbfgs" (minimizes ||step(x)-x||^2 with jax.grad + scipy's
    L-BFGS-B, via the same densitydensity_jax.solve_scf/lbfgs_solve
    machinery vjinteraction_jax.py's VJinteraction uses -- see that
    module's docstring for why residual-norm minimization, not the physical
    free energy, is what it does) must be reachable for the plain V/U-only
    engine too, not just VJinteraction, and converge to the same physics as
    solver="newton"."""
    g = geometry.bichain()
    h0 = g.get_hamiltonian()
    h1, mf = _biased_hamiltonian_and_guess(h0, seed=0, bias=.8)

    scf_newton = Vinteraction(h1.copy(), nk=20, mu=0.0, U=2., mf=mf.copy(),
            maxerror=1e-6, verbose=0, use_jax=True, solver="newton")
    scf_lbfgs = Vinteraction(h1.copy(), nk=20, mu=0.0, U=2., mf=mf.copy(),
            maxerror=1e-6, verbose=0, use_jax=True, solver="lbfgs")

    assert scf_newton.converged and scf_lbfgs.converged
    assert abs(scf_newton.total_energy - scf_lbfgs.total_energy) < 1e-4


def test_densitydensity_jax_levenberg_marquardt_matches_newton():
    """solver="levenberg_marquardt" (matrix-free Levenberg-Marquardt on the
    SCF residual via jax.jvp/jax.vjp + scipy's lsqr -- see
    densitydensity_jax.levenberg_marquardt_solve's docstring) is what
    vjinteraction_jax's VJinteraction dispatches solver="error_gradient" to,
    but exercised there only through _get_step_core_vj (the vz/vx/vy
    three-channel step). This exercises it directly through the plain
    V/U-only single-channel step (_get_step_core/build_step_function) that
    Vinteraction/generic_densitydensity_jax uses, which is a structurally
    different code path and was previously untested."""
    g = geometry.bichain()
    h0 = g.get_hamiltonian()
    h1, mf = _biased_hamiltonian_and_guess(h0, seed=0, bias=.8)

    scf_newton = Vinteraction(h1.copy(), nk=20, mu=0.0, U=2., mf=mf.copy(),
            maxerror=1e-6, verbose=0, use_jax=True, solver="newton")
    scf_lm = Vinteraction(h1.copy(), nk=20, mu=0.0, U=2., mf=mf.copy(),
            maxerror=1e-6, verbose=0, use_jax=True, solver="levenberg_marquardt")

    assert scf_newton.converged and scf_lm.converged
    assert abs(scf_newton.total_energy - scf_lm.total_energy) < 1e-4


def test_densitydensity_jax_handles_mismatched_guess_directions():
    """A regression test: an initial mean-field guess that only covers a
    subset of the interaction's directions (e.g. a nearest-neighbor-only
    guess like mode="kekule" combined with a longer-range V1+V2
    interaction, as in examples/2d/kekule_honeycomb_scf) must not crash -
    missing directions should default to zero, matching what the numpy
    engine does implicitly via MultiHopping addition."""
    g = geometry.honeycomb_lattice().get_supercell(3)
    h = g.get_hamiltonian(has_spin=False)
    scf = Vinteraction(h, V1=6.0, mf="kekule", V2=4.0, nk=4, filling=0.5,
            mix=0.3, maxerror=1e-4, maxite=50, verbose=0,
            use_jax=True, solver="fixed_point")
    assert np.isfinite(scf.total_energy)


def test_densitydensity_jax_documents_unsupported_configurations():
    """Configurations intentionally not carried over to the jax engine must
    fail loudly (NotImplementedError), never silently ignore the request."""
    g = geometry.dimer()
    h = g.get_hamiltonian()
    with pytest.raises(NotImplementedError):
        Vinteraction(h.copy(), filling=0.5, U=2.0, mf="random",
                use_jax=True, solver="newton",
                constrains=["no_charge"])  # newton can't run callback_mf
    h_nambu = h.copy()
    h_nambu.turn_nambu()
    with pytest.raises(NotImplementedError):
        Vinteraction(h_nambu, mu=0.0, U=2.0, use_jax=True)  # no BdG support yet


@pytest.mark.parametrize("solver", ["fixed_point", "newton"])
@pytest.mark.parametrize("n_occ", [2, 4])
def test_densitydensity_jax_filling_holds_the_requested_electron_count(solver,
        n_occ):
    """With a filling target the density matrix must hold exactly the
    requested number of electrons, Tr dm(0,0,0)/n = filling. On the bare
    honeycomb lattice with nk=6 the second level is a six-fold star of
    k-points, so filling=2/72 puts the cut inside it. mu used to be the
    midpoint of the two levels either side of the cut, which lands on the
    level and half-fills all six, giving twice the requested count. The
    numpy engine inverts the finite-T count, and so must this one."""
    g = geometry.honeycomb_lattice()
    h = g.get_hamiltonian(has_spin=False)
    filling = n_occ / 72
    mf0 = {(0, 0, 0): np.zeros((2, 2), dtype=complex)}
    scf = Vinteraction(h.copy(), V1=1e-10, nk=6, filling=filling, mf=mf0,
            maxerror=1e-12, maxite=3, T=1e-4, use_jax=True, solver=solver)
    count = np.trace(scf.dm[(0, 0, 0)]).real / 2
    assert abs(count - filling) < 1e-8


def test_mu_for_filling_derivative_matches_finite_differences():
    """mu_for_filling solves the count inside the jax trace, and its
    derivative is what Newton's Jacobian sees. Check it against central
    finite differences on a spectrum where the cut falls in a multiplet
    (the count branch) and on a gapped one (the midpoint branch)."""
    from pyqula.scftk.densitydensity_jax import mu_for_filling
    import jax.numpy as jnp
    rng = np.random.default_rng(3)
    T = 1e-2
    degenerate = np.array([-1., 0., 0., 0., 0.003, 1., 2.])
    gapped = np.array([-1., -0.8, 0.5, 1., 2.])
    for es, n_occ in [(degenerate, 3), (gapped, 2)]:
        es = jnp.asarray(es)
        de = jnp.asarray(rng.random(es.shape) - 0.5)
        mu = lambda t: mu_for_filling(es + t * de, n_occ, T)
        _, dmu = jax.jvp(mu, (0.,), (1.,))
        eps = 1e-6
        fd = (mu(eps) - mu(-eps)) / (2 * eps)
        assert abs(float(dmu) - float(fd)) < 1e-6
        # and the count itself is right
        count = jnp.sum(jax.nn.sigmoid(-(es - mu(0.)) / T))
        assert abs(float(count) - n_occ) < 1e-9


@pytest.mark.parametrize("target", [dict(mu=0.5), dict(filling=0.5)])
def test_densitydensity_jax_returned_hamiltonian_matches_numpy_engine(target):
    """The same public call with use_jax=False and use_jax=True must return
    the same Hamiltonian and total energy. The numpy engine returns h
    measured from the Fermi level (shifted by -mu, or by -fermi with .fermi
    set for a filling target); the jax engine used to return it unshifted
    and summed the unshifted eigenvalues, so at a fixed nonzero mu its
    total energy was off by mu*N. A spinless CDW on a two-site chain, which
    is gapped, so both engines put the Fermi level at the same place."""
    g = geometry.chain().get_supercell(2)
    h = g.get_hamiltonian(has_spin=False)
    mf = {(0, 0, 0): np.diag([0.5, -0.5]).astype(complex)}
    k = [0.13, 0., 0.]
    out = []
    for extra in [dict(mix=0.5), dict(use_jax=True, solver="newton")]:
        hh, e = h.get_mean_field_hamiltonian(V1=3., nk=10, T=1e-4,
                maxerror=1e-10, mf=mf, return_total_energy=True,
                **target, **extra)
        out.append((hh, np.linalg.eigvalsh(hh.get_hk_gen()(k)), e))
    (h_np, b_np, e_np), (h_jax, b_jax, e_jax) = out
    assert np.max(np.abs(b_np - b_jax)) < 1e-7
    assert abs(e_np - e_jax) < 1e-7
    assert hasattr(h_np, "fermi") == hasattr(h_jax, "fermi")
    if "filling" in target:
        assert abs(h_np.fermi - h_jax.fermi) < 1e-7


def test_jax_solver_names_come_from_one_registry_on_both_routes():
    """solver= on the use_jax=True engine is a string-selected option, so an
    unknown name must be refused with the accepted ones listed, and the
    list must be the registry's (get_jax_solver_names), the same on the
    spinless route (Vinteraction) and the spinful one (VJinteraction). The
    two routes used to accept different sets: the spinless one refused
    "linear_mixing" and "error_gradient", the names VJinteraction
    documents, and its error listed no names at all."""
    from pyqula.scftk.densitydensity_jax import (get_jax_solver_names,
            resolve_jax_solver)
    from pyqula.scftk.spinspin import VJinteraction
    names = get_jax_solver_names()
    assert "linear_mixing" in names and "error_gradient" in names
    assert resolve_jax_solver("linear_mixing") == "fixed_point"
    assert resolve_jax_solver("error_gradient") == "levenberg_marquardt"
    g = geometry.chain().get_supercell(2)
    hs = g.get_hamiltonian(has_spin=False)
    hf = g.get_hamiltonian()
    for call in [lambda: Vinteraction(hs.copy(), V1=2., mu=0., nk=4,
                    use_jax=True, solver="bogus"),
            lambda: VJinteraction(hf.copy(), U=2., mu=0., nk=4,
                    use_jax=True, solver="bogus")]:
        with pytest.raises(ValueError) as err:
            call()
        for name in names:
            assert repr(name) in str(err.value)
    # the aliases now work on the spinless route too, and give the same
    # answer as the names they stand for
    mf = {(0, 0, 0): np.diag([0.5, -0.5]).astype(complex)}
    e = {}
    for solver in ["linear_mixing", "fixed_point"]:
        scf = Vinteraction(hs.copy(), V1=3., mu=0., nk=10, mf=mf, T=1e-4,
                mix=0.5, maxerror=1e-8, use_jax=True, solver=solver)
        assert scf.converged
        e[solver] = scf.total_energy
    assert abs(e["linear_mixing"] - e["fixed_point"]) < 1e-10


def test_densitydensity_jax_fixed_point_converged_means_residual_below_maxerror(
        monkeypatch):
    """scf.converged means max|step(x)-x| < maxerror for every use_jax=True
    solver. fixed_point used to stop on the mean of |step(x)-x| instead and
    returned the mixed x rather than the one it had measured, so it
    reported converged=True with a residual ~8x maxerror. The residual is
    recomputed here at the returned x."""
    from pyqula.scftk import densitydensity_jax as ddj
    residual = {}
    solve_scf = ddj.solve_scf

    def wrapped(step_jit, x0, mu, *args, **kwargs):
        out = solve_scf(step_jit, x0, mu, *args, **kwargs)
        x = out[0]
        residual["max"] = float(np.max(np.abs(step_jit(x, mu)[0] - x)))
        return out
    monkeypatch.setattr(ddj, "solve_scf", wrapped)
    g = geometry.bichain()
    h0 = g.get_hamiltonian()
    h1, mf = _biased_hamiltonian_and_guess(h0, seed=0, bias=0.8)
    for maxerror in [1e-2, 1e-6]:
        scf = Vinteraction(h1.copy(), nk=20, U=2., mf=mf.copy(), mu=0.0,
                maxerror=maxerror, T=1e-4, use_jax=True, solver="fixed_point")
        assert scf.converged
        assert residual["max"] < maxerror
