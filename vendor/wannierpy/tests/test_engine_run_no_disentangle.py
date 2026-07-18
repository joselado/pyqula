"""Wiring-sanity test for the *no-disentanglement* path through
``wannier90.run(backend="python")`` (``overlap_project`` -> ``wann_main``,
taken when ``num_bands == num_wann``) -- the disentangling path has a real
physical reference to check against (GaAs, see ``test_engine_run.py`` and
friends); this path doesn't have an equivalent fixture available in this
repo (it needs a genuinely isolated-bands DFT case, not just a slice of
GaAs's entangled manifold), so it's checked here with synthetic-but-valid
data instead. This exists to catch *wiring* bugs (shape mismatches, wrong
argument order between ``overlap_project`` and ``wann_main``) -- not to
validate the physics, which ``overlap_project``'s own unit test
(``test_engine_overlap.py``) already cross-checks independently.
"""
import numpy as np

from wannier90._engine.kmesh import kmesh_get


def _synthetic_case(num_wann=4, mp_grid=(2, 2, 2), seed=0):
    """A mathematically self-consistent (but not physically-derived)
    M_matrix/A_matrix: M(k, b) = W(k)^dagger @ W(k2) for random unitary
    W(k) exactly satisfies the algebraic constraints a real overlap matrix
    would (each block unitary, consistent under the k -> k+b relation),
    without needing real DFT data."""
    rng = np.random.default_rng(seed)
    real_lattice = 3.0 * np.eye(3)
    recip_lattice = 2 * np.pi * np.linalg.inv(real_lattice).T
    grid = np.array(mp_grid)
    num_kpts = int(np.prod(grid))
    kpt_latt = np.array([
        [i / grid[0], j / grid[1], k / grid[2]]
        for i in range(grid[0]) for j in range(grid[1]) for k in range(grid[2])
    ]).T

    kmesh = kmesh_get(kpt_latt, recip_lattice)

    W = np.empty((num_wann, num_wann, num_kpts), dtype=complex)
    for k in range(num_kpts):
        z = rng.normal(size=(num_wann, num_wann)) + 1j * rng.normal(size=(num_wann, num_wann))
        q, _ = np.linalg.qr(z)
        W[:, :, k] = q

    M_matrix = np.empty((num_wann, num_wann, kmesh.nntot, num_kpts), dtype=complex)
    for k in range(num_kpts):
        for nn in range(kmesh.nntot):
            k2 = int(kmesh.nnlist[k, nn]) - 1
            M_matrix[:, :, nn, k] = W[:, :, k].conj().T @ W[:, :, k2]

    A_matrix = rng.normal(size=(num_wann, num_wann, num_kpts)) + \
        1j * rng.normal(size=(num_wann, num_wann, num_kpts))

    return kmesh, M_matrix, A_matrix, real_lattice, recip_lattice, kpt_latt, grid


def test_no_disentanglement_path_runs_and_converges():
    import wannier90

    num_wann = 4
    kmesh, M_matrix, A_matrix, real_lattice, recip_lattice, kpt_latt, mp_grid = _synthetic_case(num_wann=num_wann)
    num_kpts = kpt_latt.shape[1]

    atoms_cart = np.zeros((3, 1))

    setup_result = wannier90.setup(
        "synthetic", mp_grid, kpt_latt, real_lattice, num_wann, ["X"], atoms_cart,
        win_keywords={"num_wann": num_wann, "num_iter": 200, "conv_tol": 1e-10, "conv_window": 3},
        backend="python",
    )
    assert setup_result.num_bands == setup_result.num_wann == num_wann

    eigenvalues = np.zeros((num_wann, num_kpts))  # unused on the no-disentanglement path

    run_result = wannier90.run(
        "synthetic", setup_result, mp_grid, kpt_latt, real_lattice, ["X"], atoms_cart,
        M_matrix, A_matrix, eigenvalues,
    )

    assert run_result.U_matrix.shape == (num_wann, num_wann, num_kpts)
    assert np.all(run_result.lwindow)
    for k in range(num_kpts):
        gram = run_result.U_matrix[:, :, k].conj().T @ run_result.U_matrix[:, :, k]
        np.testing.assert_allclose(gram, np.eye(num_wann), atol=1e-8)

    # Physically spread is non-negative, but this synthetic construction (an
    # independent random unitary gauge per k, no smoothness across k) can
    # land arbitrarily close to the degenerate zero-spread case, where the
    # true value is indistinguishable from 0 at float64 precision -- seen in
    # practice as tiny (~1e-14) negative noise depending on numpy version/
    # platform (caught by testing an actual clean-venv install, not just
    # this dev environment). Tolerance-bounded rather than `>= 0`.
    tol = 1e-8
    assert np.all(np.isfinite(run_result.wann_spreads))
    assert np.all(run_result.wann_spreads >= -tol)
    assert run_result.spread_total >= -tol
    # spread_invariant is gauge-independent and fixed by M_matrix alone (SMV Sec. III) --
    # the CG minimization only ever reduces the gauge-dependent remainder on top of it.
    assert run_result.spread_total >= run_result.spread_invariant - tol
