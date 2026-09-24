import numpy as np

from pyqula import geometry
from pyqula import meanfield

# Coverage for the per-site (array) `filling` path added to
# scftk.spinspin._run_anisotropic_scf (VJinteraction/Jinteraction's
# shared SCF core) -- see densitymatrix.full_dm_accumulate_sparse_local_fermi
# for the mechanism (a per-site Lagrange multiplier/local chemical potential,
# warm-started and co-converged with the mean field across the same outer SCF
# loop, since -- unlike a scalar Fermi shift -- there is no single
# diagonalization that gives both the density matrix and the per-site
# potentials at once).


def test_vjinteraction_per_site_filling_converges_to_nonuniform_local_targets():
    """A NON-uniform per-site filling target (0.3/0.7, not a uniform value)
    on a translationally-symmetric 2-site chain supercell must be reached at
    every site individually, not just on lattice average (which a single
    scalar Fermi level could already satisfy without doing anything site-
    resolved). No V/J interaction is active here (V1=0, everything else at
    its default of 0) so this isolates the per-site local-fermi machinery
    itself from the mean-field physics -- lam has to do all the work of
    breaking the lattice's own symmetry to hit different targets on
    otherwise-identical sites."""
    g = geometry.chain().get_supercell(2)
    h = g.get_hamiltonian(has_spin=True)
    filling = np.array([0.3, 0.7])
    scf = meanfield.VJinteraction(h, V1=0.0, mf="ferroZ", nk=24,
            maxerror=1e-5, mix=0.3, maxite=2000, filling=filling)
    assert scf.converged, "SCF did not converge"
    occ = scf.local_occupation
    assert np.allclose(occ, filling, atol=5e-3), (occ, filling)
    # scf.lam / scf.hamiltonian.fermi expose the converged per-site
    # potentials as a diagnostic (e.g. for a future SpinonHamiltonian to
    # read back the Lagrange multipliers enforcing the local constraint)
    assert len(scf.lam) == 2
    assert np.allclose(scf.hamiltonian.fermi, scf.lam)


def test_vjinteraction_uniform_array_filling_matches_scalar_filling():
    """The array path's normalization convention: a per-site array of 0.5
    everywhere must reproduce the SAME physics as today's scalar
    filling=0.5 (both mean 'half of each site's 2-orbital up+down capacity',
    i.e. 1 electron/site on average) -- the correctness sanity check
    against silently drifting to a different (e.g. raw-electron-count, 0-2
    range) convention."""
    g = geometry.chain().get_supercell(2)

    h_scalar = g.get_hamiltonian(has_spin=True)
    scf_scalar = meanfield.VJinteraction(h_scalar, V1=0.0, mf="ferroZ",
            nk=24, maxerror=1e-6, mix=0.3, maxite=500, filling=0.5)
    assert scf_scalar.converged

    h_array = g.get_hamiltonian(has_spin=True)
    filling = np.array([0.5, 0.5])
    scf_array = meanfield.VJinteraction(h_array, V1=0.0, mf="ferroZ",
            nk=24, maxerror=1e-6, mix=0.3, maxite=500, filling=filling)
    assert scf_array.converged
    assert np.allclose(scf_array.local_occupation, 0.5, atol=5e-3)
    assert np.isclose(scf_scalar.total_energy, scf_array.total_energy,
            atol=1e-3), (scf_scalar.total_energy, scf_array.total_energy)


def test_vjinteraction_converged_implies_occupation_within_tolerance():
    """Regression test for a real bug found by smoke-testing this feature:
    _run_anisotropic_scf's outer loop used to validate the occupation
    residual (occ_err, folded into `diff`) from ONE call to f(), but then
    return a DIFFERENT scf built by a second, unvalidated call
    (`f(mfnew)`, the "last iteration, with the unmixed mean field" step
    shared with the scalar-filling path) -- and that second call's own
    array-filling branch mutates the warm-started lam a further (small)
    step as a side effect. On most systems that residual mutation is
    negligible, but per-site occupation vs lam is not smooth on a finite
    k-mesh with the near-zero default smearing: a single k-point eigenvalue
    crossing zero as lam varies flips that state's occupation contribution
    discontinuously. On a trivial 1-site chain (tij=[0.0], so no
    dispersion beyond what the mean field itself induces) with a
    ferromagnetic-favoring J1 and a random initial mf guess, some seeds'
    converged lam sits essentially exactly at such a crossing --
    reproducibly making the OLD code report scf.converged=True with
    scf.local_occupation off by up to ~0.04 (a whole order of magnitude
    above maxerror) in a handful of the 100 seeds tried. The fix re-checks
    the actually-returned scf's own occupation before trusting
    convergence, falling through to another outer iteration (bounded by
    maxite, same as always) instead of reporting a false positive. This
    test asserts the CONTRACT VJinteraction's docstring makes explicit:
    scf.converged=True must imply scf.local_occupation is within maxerror
    of `filling` -- not merely usually true."""
    g = geometry.chain()
    filling = np.array([0.5])
    maxerror = 1e-5
    n_violations = 0
    for seed in range(30):
        np.random.seed(seed)
        h = g.get_hamiltonian(has_spin=True, tij=[0.0])
        scf = meanfield.VJinteraction(h, J1=1.0, nk=24, mix=0.3,
                maxerror=maxerror, maxite=1000, filling=filling, verbose=0)
        if scf.converged:
            err = np.max(np.abs(scf.local_occupation - filling))
            if err >= 2*maxerror:
                n_violations += 1
    assert n_violations == 0, \
        f"{n_violations}/30 seeds falsely reported converged=True with " \
        "local_occupation outside tolerance"


def test_array_filling_partially_filled_kshell_matches_scalar_at_default_T():
    """A one-site spinful chain at filling 0.3 with nk=20 holds 12 of 40
    states, so the level at the Fermi energy (4 degenerate states) is half
    filled. The scalar path puts mu on that level; the array path used to
    step the total count with a fixed gain and, at the default T=1e-7,
    overshot that level forever (with maxite=None it never returned). The
    array [0.3] must reproduce the scalar result, which is the free chain."""
    g = geometry.chain()
    res = []
    for filling in (0.3, np.array([0.3])):
        h = g.get_hamiltonian(has_spin=True)
        np.random.seed(0)
        res.append(meanfield.VJinteraction(h, U=0.0, nk=20, maxerror=1e-5,
                mix=0.1, maxite=200, filling=filling))
    scalar, array = res
    assert scalar.converged and array.converged
    assert np.allclose(array.local_occupation, 0.3, atol=1e-8)
    assert np.isclose(array.total_energy, scalar.total_energy, atol=1e-7)
    assert np.allclose(array.lam, scalar.hamiltonian.fermi, atol=1e-7)


def test_uniform_array_filling_matches_scalar_off_commensurate_mesh():
    """Off a commensurate mesh the scalar path rounds the filling to a
    whole number of k-states; the array path fixes its total count with the
    same Fermi search, so a uniform array must give the same state and
    energy rather than chase an unreachable total count."""
    g = geometry.triangular_lattice()
    res = []
    for filling in (0.3, np.array([0.3])):
        h = g.get_hamiltonian(has_spin=True)
        np.random.seed(1)
        res.append(meanfield.VJinteraction(h, U=1.0, mf="ferroZ", nk=8,
                maxerror=1e-6, mix=0.3, maxite=300, filling=filling))
    scalar, array = res
    assert scalar.converged and array.converged
    assert np.isclose(array.total_energy, scalar.total_energy, atol=1e-6)


def test_array_filling_converges_in_a_gapped_state():
    """A charge-ordered, gapped 2-site chain (U=3, J1=0.5, T=0.05) with
    targets [0.3,0.7]: the total-count part of the old fixed-gain step
    crawled inside the gap (unconverged after 3000 iterations, where the
    scalar path needs under 200). With the count fixed by a Fermi search it
    converges in a few hundred iterations and hits both targets."""
    g = geometry.chain().get_supercell(2)
    h = g.get_hamiltonian(has_spin=True)
    np.random.seed(3)
    filling = np.array([0.3, 0.7])
    scf = meanfield.VJinteraction(h, U=3.0, J1=0.5, mf="ferroZ",
            filling=filling, nk=20, maxerror=1e-6, mix=0.3, maxite=600,
            T=0.05)
    assert scf.converged
    assert np.allclose(scf.local_occupation, filling, atol=1e-5)


def test_array_filling_is_validated_before_any_scf_work():
    """A per-site filling needs one value per site, each a fraction in
    [0,1]; a wrong length used to surface as a numpy broadcast error after
    a diagonalization, and an unreachable target ran to maxite (forever by
    default)."""
    import pytest
    g = geometry.chain().get_supercell(2)
    h = g.get_hamiltonian(has_spin=True)
    with pytest.raises(ValueError, match="one value per site"):
        meanfield.VJinteraction(h, U=1.0, nk=4, maxite=5,
                filling=np.array([0.3, 0.7, 0.5]))
    with pytest.raises(ValueError, match=r"in \[0,1\]"):
        meanfield.VJinteraction(h, U=1.0, nk=4, maxite=5,
                filling=np.array([1.3, -0.2]))


def test_array_filling_is_refused_by_routes_without_it():
    """Only VJinteraction's numpy engine implements a per-site filling.
    Every other route used to pass the array on to a scalar Fermi search
    and fail with 'numpy.ndarray doesn't define __round__'; each now
    refuses it by name before any SCF work."""
    import pytest
    g = geometry.chain().get_supercell(2)
    filling = np.array([0.3, 0.7])
    hs = g.get_hamiltonian(has_spin=False)
    with pytest.raises(ValueError, match="per-site .* needs a spinful"):
        hs.get_mean_field_hamiltonian(V1=1.0, filling=filling, nk=4,
                maxite=5)
    with pytest.raises(ValueError, match="per-site .* needs a spinful"):
        hs.get_mean_field_hamiltonian(V1=1.0, filling=filling, nk=4,
                maxite=5, integration="kpm")
    hf = g.get_hamiltonian(has_spin=True)
    with pytest.raises(NotImplementedError, match="per-site"):
        meanfield.Vinteraction(hf, U=1.0, filling=filling, nk=4, maxite=5)
    with pytest.raises(NotImplementedError, match="per-site"):
        meanfield.Vinteraction(hs, V1=1.0, filling=filling, nk=4, maxite=5)
    from pyqula.scftk.densitydensity_kpm import Vinteraction_kpm
    with pytest.raises(NotImplementedError, match="per-site"):
        Vinteraction_kpm(hf, U=1.0, filling=filling, nk=4, maxite=5)
    with pytest.raises(NotImplementedError, match="use_jax=True .* per-site"):
        meanfield.VJinteraction(hf, U=1.0, filling=filling, nk=4, maxite=5,
                use_jax=True)
