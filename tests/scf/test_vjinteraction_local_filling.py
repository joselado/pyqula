import numpy as np

from pyqula import geometry
from pyqula import meanfield

# Coverage for the per-site (array) `filling` path added to
# selfconsistency.spinspin._run_anisotropic_scf (VJinteraction/Jinteraction's
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
