import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp

from pyqula import geometry
from pyqula.selfconsistency.spinspin import VJinteraction


def _biased_hamiltonian_and_guess(h0, seed, bias=.8):
    """Same construction as test_densitydensity_jax.py's helper: bias the
    Hamiltonian itself (not just the mean-field guess) along a random
    direction, so the SU(2) spin-rotation symmetry of the exchange/Hubbard
    mean field is broken and the fixed point is an isolated point rather
    than a whole marginal manifold -- otherwise Newton's Jacobian is
    singular along that direction and convergence is slow or stalls."""
    rng = np.random.default_rng(seed)
    v = rng.random(3) - .5
    v = 2 * v / np.sqrt(v.dot(v))
    mf = h0.copy()
    mf.add_exchange([v, -v])  # initial guess
    h1 = h0.copy()
    h1.add_exchange(bias * v)  # bias, breaks the marginal direction
    return h1, mf


def test_vjinteraction_jax_vu_only_matches_numpy_engine():
    """With no exchange (J1=J2=J3=J1x=J1y=J1z=0), VJinteraction's use_jax=True
    path must converge to the same physics as its own plain-mixing numpy
    engine for a plain U interaction."""
    g = geometry.bichain()
    h0 = g.get_hamiltonian()
    h1, mf0 = _biased_hamiltonian_and_guess(h0, seed=0)

    scf_np = VJinteraction(h1.copy(), nk=20, mu=0.0, U=2.0, mf=mf0.copy(),
            maxerror=1e-6, verbose=0)
    scf_jax = VJinteraction(h1.copy(), nk=20, mu=0.0, U=2.0, mf=mf0.copy(),
            maxerror=1e-6, verbose=0, use_jax=True, solver="newton")

    assert scf_np.converged and scf_jax.converged
    assert abs(scf_np.total_energy - scf_jax.total_energy) < 1e-4
    diff = np.max(np.abs(scf_np.mf[(0, 0, 0)] - scf_jax.mf[(0, 0, 0)]))
    assert diff < 1e-3


def test_vjinteraction_jax_fixed_nonzero_mu_matches_numpy_engine():
    """Regression test for a real bug found during development: with an
    explicit, nonzero mu, generic_vjinteraction_jax's total_energy formula
    summed the raw (unshifted) band energies unconditionally -- correct only
    for a filling target, where the numpy engine's own shift-then-add-back
    telescopes back to the same thing, but silently wrong (off by mu*N_occ)
    for a fixed mu, where the numpy engine sums the *shifted* eigenvalues
    with no compensating add-back. scf.hamiltonian.fermi must also raise
    AttributeError here, matching the numpy engine, which only ever assigns
    .fermi on the filling-target branch of its own callback_h."""
    g = geometry.bichain()
    h0 = g.get_hamiltonian()
    h1, mf0 = _biased_hamiltonian_and_guess(h0, seed=0)

    scf_np = VJinteraction(h1.copy(), nk=20, mu=0.3, U=2.0, mf=mf0.copy(),
            maxerror=1e-6, verbose=0)
    scf_jax = VJinteraction(h1.copy(), nk=20, mu=0.3, U=2.0, mf=mf0.copy(),
            maxerror=1e-6, verbose=0, use_jax=True, solver="newton")

    assert scf_np.converged and scf_jax.converged
    assert abs(scf_np.total_energy - scf_jax.total_energy) < 1e-4
    diff_intra = np.max(np.abs(scf_np.hamiltonian.intra - scf_jax.hamiltonian.intra))
    assert diff_intra < 1e-3
    with pytest.raises(AttributeError):
        scf_np.hamiltonian.fermi
    with pytest.raises(AttributeError):
        scf_jax.hamiltonian.fermi


def test_vjinteraction_jax_isotropic_exchange_matches_numpy_engine():
    """Pure isotropic J1 exchange (no V/U) must also match the numpy engine
    -- checked on scf.mf directly (not just total_energy), since this is the
    case that most exercises the x/y rotate-decouple-rotate-back trick
    (vz==vx==vy here, as J1x=J1y=J1z=0): a rotation-formula bug that happens
    to preserve total energy (plausible, since an isotropic interaction's
    energy is itself rotation-invariant) would slip past a total_energy-only
    check but not this one."""
    g = geometry.bichain()
    h0 = g.get_hamiltonian()
    h1, mf0 = _biased_hamiltonian_and_guess(h0, seed=1)

    scf_np = VJinteraction(h1.copy(), nk=20, mu=0.0, J1=-1.0, mf=mf0.copy(),
            maxerror=1e-6, verbose=0)
    scf_jax = VJinteraction(h1.copy(), nk=20, mu=0.0, J1=-1.0, mf=mf0.copy(),
            maxerror=1e-6, verbose=0, use_jax=True, solver="newton")

    assert scf_np.converged and scf_jax.converged
    assert abs(scf_np.total_energy - scf_jax.total_energy) < 1e-4
    diff = np.max(np.abs(scf_np.mf[(0, 0, 0)] - scf_jax.mf[(0, 0, 0)]))
    assert diff < 1e-3


def _total_energy_isotropic_jax(h0, seed):
    h1, mf = _biased_hamiltonian_and_guess(h0, seed)
    scf = VJinteraction(h1, nk=20, mu=0.0, J1=-1.0, mf=mf, maxerror=1e-7,
            verbose=0, use_jax=True, solver="newton")
    assert scf.converged
    return scf.total_energy


def test_vjinteraction_jax_isotropic_exchange_is_rotationally_invariant():
    """Same physical invariant as test_rotational_symmetry.py /
    test_densitydensity_jax.py's Newton test: with a purely isotropic
    exchange, the converged total energy must not depend on the (arbitrary)
    direction of the initial/biasing exchange field."""
    g = geometry.bichain()
    h0 = g.get_hamiltonian()
    etots = np.array([_total_energy_isotropic_jax(h0, seed)
        for seed in range(4)])
    diff = etots - np.mean(etots)
    assert np.max(np.abs(diff)) < 1e-6, \
        f"jax VJinteraction total energy is not rotationally invariant: {diff}"


def test_vjinteraction_jax_combined_v_and_anisotropic_j_matches_numpy_engine():
    """V (U) + isotropic J1 + anisotropic J1x/J1y/J1z all active together in
    the same SCF loop -- exercises the vz/vx/vy multi-channel combination
    and the rotate-decouple-rotate-back trick for the x/y channels."""
    g = geometry.bichain()
    h0 = g.get_hamiltonian()
    h1, mf0 = _biased_hamiltonian_and_guess(h0, seed=2)
    kwargs = dict(U=1.5, J1=-0.7, J1x=0.2, J1y=-0.1, J1z=0.3)

    scf_np = VJinteraction(h1.copy(), nk=20, mu=0.0, mf=mf0.copy(),
            maxerror=1e-6, verbose=0, **kwargs)
    scf_jax = VJinteraction(h1.copy(), nk=20, mu=0.0, mf=mf0.copy(),
            maxerror=1e-6, verbose=0, use_jax=True, solver="newton", **kwargs)

    assert scf_np.converged and scf_jax.converged
    assert abs(scf_np.total_energy - scf_jax.total_energy) < 1e-4
    diff = np.max(np.abs(scf_np.mf[(0, 0, 0)] - scf_jax.mf[(0, 0, 0)]))
    assert diff < 1e-3


def test_vjinteraction_jax_handles_filling():
    """A target filling (mu=None) resolves mu *inside* the jax trace each
    step (jnp.sort midpoint), rather than a numpy root-find outside it --
    must converge to the same physics as the numpy engine for the same
    filling target. Compared on total_energy only (not .mf/.hamiltonian):
    from a raw random, unbiased seed this dimer has the same SU(2)-marginal-
    direction degeneracy test_densitydensity_jax_newton_handles_filling
    already documents for the plain-V engine -- the two engines can (and
    empirically do) land on two different, exactly energy-degenerate points
    on that marginal manifold, so only total_energy is a meaningful
    invariant here. See
    test_vjinteraction_jax_filling_target_hamiltonian_matches_numpy_engine
    below for a biased (non-marginal) case where the returned Hamiltonian
    itself is checked."""
    g = geometry.dimer()
    h = g.get_hamiltonian()
    U = 2.0
    rng = np.random.default_rng(3)
    n = h.intra.shape[0]
    m = rng.random((n, n)) - 0.5 + 1j * (rng.random((n, n)) - 0.5)
    m = m + m.T.conjugate()
    mf0 = {(0, 0, 0): m}

    scf_np = VJinteraction(h.copy(), filling=0.5, U=U,
            mf={k: v.copy() for k, v in mf0.items()},
            maxerror=1e-6, verbose=0)
    scf_jax = VJinteraction(h.copy(), filling=0.5, U=U,
            mf={k: v.copy() for k, v in mf0.items()},
            maxerror=1e-7, verbose=0, use_jax=True, solver="newton")

    assert scf_np.converged and scf_jax.converged
    assert abs(scf_np.total_energy - scf_jax.total_energy) < 1e-4


def test_vjinteraction_jax_filling_target_hamiltonian_matches_numpy_engine():
    """Regression test for a real bug found during development: the numpy
    engine shifts every SCF iterate (hence its final scf.hamiltonian) by
    -fermi so the Fermi level sits at 0 under a filling target: see
    spinspin._run_anisotropic_scf's callback_h. generic_vjinteraction_jax's
    own step function computes dm/es/occ directly against mu_eff without
    needing any such shift, but its *returned* h_final must still apply the
    same -fermi shift afterward, or its raw .intra would differ from the
    numpy engine's by a uniform +fermi diagonal offset on every site/spin
    even though total_energy and scf.mf (the mean-field contribution alone,
    unaffected either way) already agreed exactly -- caught by comparing
    scf.hamiltonian.intra directly (not just total_energy) on a
    symmetry-broken (non-marginal) honeycomb Hubbard+exchange system, where
    the fixed point is isolated enough that both engines must land on the
    same one, not just the same energy."""
    g = geometry.honeycomb_lattice()
    h0 = g.get_hamiltonian(has_spin=True)
    h1, mf0 = _biased_hamiltonian_and_guess(h0, seed=0)

    scf_np = VJinteraction(h1.copy(), U=4.0, J1=-0.5, filling=0.5, nk=6,
            mf=mf0.copy(), maxerror=1e-6, verbose=0)
    scf_jax = VJinteraction(h1.copy(), U=4.0, J1=-0.5, filling=0.5, nk=6,
            mf=mf0.copy(), maxerror=1e-6, verbose=0, use_jax=True,
            solver="newton")

    assert scf_np.converged and scf_jax.converged
    assert abs(scf_np.total_energy - scf_jax.total_energy) < 1e-6
    diff_intra = np.max(np.abs(scf_np.hamiltonian.intra - scf_jax.hamiltonian.intra))
    assert diff_intra < 1e-4


def test_vjinteraction_jax_newton_krylov_matches_newton():
    """solver="newton_krylov" solves the same Newton step with matrix-free
    GMRES (jax.jvp Jacobian-vector products) instead of the dense
    jax.jacfwd Jacobian -- must converge to the same physics."""
    g = geometry.bichain()
    h0 = g.get_hamiltonian()
    h1, mf0 = _biased_hamiltonian_and_guess(h0, seed=2)
    kwargs = dict(U=1.5, J1=-0.7, J1x=0.2, J1y=-0.1, J1z=0.3)

    scf_newton = VJinteraction(h1.copy(), nk=20, mu=0.0, mf=mf0.copy(),
            maxerror=1e-6, verbose=0, use_jax=True, solver="newton", **kwargs)
    scf_nk = VJinteraction(h1.copy(), nk=20, mu=0.0, mf=mf0.copy(),
            maxerror=1e-6, verbose=0, use_jax=True, solver="newton_krylov",
            **kwargs)

    assert scf_newton.converged and scf_nk.converged
    assert abs(scf_newton.total_energy - scf_nk.total_energy) < 1e-6


def test_vjinteraction_jax_lbfgs_vu_only_matches_numpy_engine():
    """solver="lbfgs" minimizes ||step(x)-x||^2 with jax.grad + scipy's
    L-BFGS-B, instead of root-finding step(x)=x the way every other solver
    here does (see vjinteraction_jax's module docstring for why -- minimizing
    the actual free energy directly was tried first and abandoned after
    finding the SCF solution is generically a saddle point of that
    functional). With no exchange (pure U), it must still converge to the
    same physics as the plain-mixing numpy engine."""
    g = geometry.bichain()
    h0 = g.get_hamiltonian()
    h1, mf0 = _biased_hamiltonian_and_guess(h0, seed=0)

    scf_np = VJinteraction(h1.copy(), nk=20, mu=0.0, U=2.0, mf=mf0.copy(),
            maxerror=1e-6, verbose=0)
    scf_lbfgs = VJinteraction(h1.copy(), nk=20, mu=0.0, U=2.0, mf=mf0.copy(),
            maxerror=1e-6, verbose=0, use_jax=True, solver="lbfgs")

    assert scf_np.converged and scf_lbfgs.converged
    assert abs(scf_np.total_energy - scf_lbfgs.total_energy) < 1e-4
    diff = np.max(np.abs(scf_np.mf[(0, 0, 0)] - scf_lbfgs.mf[(0, 0, 0)]))
    assert diff < 1e-3


def test_vjinteraction_jax_lbfgs_combined_v_and_anisotropic_j_matches_numpy_engine():
    """Same combined V+isotropic-J+anisotropic-J1x/J1y/J1z system
    test_vjinteraction_jax_combined_v_and_anisotropic_j_matches_numpy_engine
    exercises for solver="newton", now for solver="lbfgs"."""
    g = geometry.bichain()
    h0 = g.get_hamiltonian()
    h1, mf0 = _biased_hamiltonian_and_guess(h0, seed=2)
    kwargs = dict(U=1.5, J1=-0.7, J1x=0.2, J1y=-0.1, J1z=0.3)

    scf_np = VJinteraction(h1.copy(), nk=20, mu=0.0, mf=mf0.copy(),
            maxerror=1e-6, verbose=0, **kwargs)
    scf_lbfgs = VJinteraction(h1.copy(), nk=20, mu=0.0, mf=mf0.copy(),
            maxerror=1e-6, verbose=0, use_jax=True, solver="lbfgs", **kwargs)

    assert scf_np.converged and scf_lbfgs.converged
    assert abs(scf_np.total_energy - scf_lbfgs.total_energy) < 1e-4
    diff = np.max(np.abs(scf_np.mf[(0, 0, 0)] - scf_lbfgs.mf[(0, 0, 0)]))
    assert diff < 1e-3


def test_vjinteraction_jax_lbfgs_matches_newton():
    """solver="lbfgs" and solver="newton" solve the same fixed point two
    different ways (least-squares residual minimization vs. Newton root-
    finding) -- on this biased, non-marginal system they must land on the
    same isolated solution, not just the same energy."""
    g = geometry.bichain()
    h0 = g.get_hamiltonian()
    h1, mf0 = _biased_hamiltonian_and_guess(h0, seed=2)
    kwargs = dict(U=1.5, J1=-0.7, J1x=0.2, J1y=-0.1, J1z=0.3)

    scf_newton = VJinteraction(h1.copy(), nk=20, mu=0.0, mf=mf0.copy(),
            maxerror=1e-6, verbose=0, use_jax=True, solver="newton", **kwargs)
    scf_lbfgs = VJinteraction(h1.copy(), nk=20, mu=0.0, mf=mf0.copy(),
            maxerror=1e-6, verbose=0, use_jax=True, solver="lbfgs", **kwargs)

    assert scf_newton.converged and scf_lbfgs.converged
    assert abs(scf_newton.total_energy - scf_lbfgs.total_energy) < 1e-6
    diff = np.max(np.abs(scf_newton.mf[(0, 0, 0)] - scf_lbfgs.mf[(0, 0, 0)]))
    assert diff < 1e-5


def test_vjinteraction_jax_lbfgs_handles_filling():
    """A target filling resolves mu *inside* the jax trace (same jnp.sort
    trick every other solver here uses) -- solver="lbfgs" must handle this
    the same way solver="newton" already does (regression coverage for the
    filling-dependent term in the residual: unlike a fixed mu, step(x) here
    depends on x both directly and through mu_eff(x))."""
    g = geometry.bichain()
    h0 = g.get_hamiltonian()
    h1, mf0 = _biased_hamiltonian_and_guess(h0, seed=0)

    scf_np = VJinteraction(h1.copy(), nk=20, filling=0.5, U=2.0,
            mf=mf0.copy(), maxerror=1e-6, verbose=0)
    scf_lbfgs = VJinteraction(h1.copy(), nk=20, filling=0.5, U=2.0,
            mf=mf0.copy(), maxerror=1e-6, verbose=0, use_jax=True,
            solver="lbfgs")

    assert scf_np.converged and scf_lbfgs.converged
    assert abs(scf_np.total_energy - scf_lbfgs.total_energy) < 1e-4
    diff_intra = np.max(np.abs(scf_np.hamiltonian.intra - scf_lbfgs.hamiltonian.intra))
    assert diff_intra < 1e-3


def test_rotation_formulas_agree_between_numpy_and_jax_engines():
    """spinspin._block_rotate/_rot_dict/_rot_dm (numpy, module-level -- used
    by the plain SCF loop) and vjinteraction_jax._block_rotate_jax/
    _rot_dict_jax/_rot_dm_jax (JAX ports of the identical formula, needed
    since this module must stay importable without jax, so the numpy
    versions can't just call into jax.numpy) are two independent
    implementations of the same math, kept in sync only by this test rather
    than by sharing code -- this repo has already hit a real bug in exactly
    this rotation-formula code path once (the RPA Goldstone-theorem vertex
    sign error), so a future fix applied to only one of the two copies
    should fail this test rather than silently drift."""
    from pyqula.selfconsistency import spinspin
    from pyqula.selfconsistency import vjinteraction_jax as vjj
    from pyqula.rotate_spin import build_rotation_matrix

    rng = np.random.default_rng(0)
    n_orb = 3  # spinful sites -> matrices are (2*n_orb, 2*n_orb)
    n = 2 * n_orb

    def random_hermitian():
        m = rng.random((n, n)) - 0.5 + 1j * (rng.random((n, n)) - 0.5)
        return m + m.conj().T

    dd = {(0, 0, 0): random_hermitian(), (1, 0, 0): random_hermitian()}
    R = build_rotation_matrix(1, **spinspin._AXIS_ROTATION["x"])
    dd_j = {k: jnp.asarray(v) for k, v in dd.items()}
    R_j = jnp.asarray(R, dtype=jnp.complex128)

    for k in dd:
        diff = np.max(np.abs(spinspin._block_rotate(dd[k], R)
                - np.asarray(vjj._block_rotate_jax(dd_j[k], R_j))))
        assert diff < 1e-12

    rot_dict = spinspin._rot_dict(dd, R)
    rot_dict_j = vjj._rot_dict_jax(dd_j, R_j)
    for k in dd:
        diff = np.max(np.abs(rot_dict[k] - np.asarray(rot_dict_j[k])))
        assert diff < 1e-12

    rot_dm = spinspin._rot_dm(dd, R)
    rot_dm_j = vjj._rot_dm_jax(dd_j, R_j)
    for k in dd:
        diff = np.max(np.abs(rot_dm[k] - np.asarray(rot_dm_j[k])))
        assert diff < 1e-12


def test_vjinteraction_jax_documents_unsupported_configurations():
    """Configurations not carried over to the jax engine must fail loudly."""
    g = geometry.dimer()
    h = g.get_hamiltonian()
    h_nambu = h.copy()
    h_nambu.turn_nambu()
    with pytest.raises(NotImplementedError):
        VJinteraction(h_nambu, mu=0.0, U=2.0, use_jax=True)  # no BdG support yet
    with pytest.raises(ValueError):
        VJinteraction(h.copy(), mu=0.0, U=2.0, use_jax=True, solver="bogus")
    with pytest.raises(NotImplementedError):
        VJinteraction(h.copy(), mu=0.0, U=2.0, use_jax=True,
                integration="kpm")  # no KPM jax counterpart
    with pytest.raises(NotImplementedError):
        VJinteraction(h.copy(), mu=0.0, U=2.0, use_jax=True,
                constrains=["no_charge"])  # needs concrete numpy arrays
