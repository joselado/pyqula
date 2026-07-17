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


def test_densitydensity_jax_newton_handles_filling():
    """solver="newton" also supports a target filling: mu is resolved
    *inside* the jax trace each step as the midpoint between the N-th and
    (N+1)-th eigenvalue of the sorted spectrum (via jnp.sort), rather than
    with a numpy root-find outside the trace - so it must converge to the
    same physics as the numpy engine for the same filling target. Compared
    against the numpy engine (not solver="fixed_point"): from a raw random,
    unbiased seed this dimer has the same SU(2)-marginal-direction slow tail
    as test_densitydensity_jax_newton_is_rotationally_invariant works around
    with a bias, and plain mixing (both engines) needs very many iterations
    to fully resolve it - not something either implementation's Newton
    solver needs to (and shouldn't be graded on)."""
    g = geometry.dimer()
    h = g.get_hamiltonian()
    U = 2.0
    rng = np.random.default_rng(3)
    n = h.intra.shape[0]
    m = rng.random((n, n)) - 0.5 + 1j * (rng.random((n, n)) - 0.5)
    m = m + m.T.conjugate()
    mf0 = {(0, 0, 0): m}

    scf_old = Vinteraction(h.copy(), filling=0.5, U=U,
            mf={k: v.copy() for k, v in mf0.items()},
            maxerror=1e-6, verbose=0)
    scf_newton = Vinteraction(h.copy(), filling=0.5, U=U,
            mf={k: v.copy() for k, v in mf0.items()},
            maxerror=1e-7, verbose=0, use_jax=True, solver="newton")

    assert scf_old.converged and scf_newton.converged
    assert abs(scf_old.total_energy - scf_newton.total_energy) < 1e-4


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
    physics as solver="newton" on a case both can handle."""
    g = geometry.bichain()
    h0 = g.get_hamiltonian()
    h1, mf = _biased_hamiltonian_and_guess(h0, seed=0, bias=.8)

    scf_newton = Vinteraction(h1.copy(), nk=20, mu=0.0, U=2., mf=mf.copy(),
            maxerror=1e-8, verbose=0, use_jax=True, solver="newton")
    scf_nk = Vinteraction(h1.copy(), nk=20, mu=0.0, U=2., mf=mf.copy(),
            maxerror=1e-8, verbose=0, use_jax=True, solver="newton_krylov")

    assert scf_newton.converged and scf_nk.converged
    assert abs(scf_newton.total_energy - scf_nk.total_energy) < 1e-6


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
