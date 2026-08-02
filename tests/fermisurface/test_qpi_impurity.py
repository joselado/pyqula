import numpy as np

from pyqula import geometry


def _away_from_gamma(q, qpi_q, tol=1e-8):
    mask = np.sum(q * q, axis=1) > tol
    return qpi_q[mask]


def test_clean_supercell_has_no_qpi_weight_away_from_gamma():
    """Physical invariant: with no impurities, the supercell is exactly
    as periodic as the primitive cell, so its real-space LDOS is uniform
    site-to-site and its Fourier transform must vanish at every q except
    Gamma (q=0, the DOS itself). num_waves=71 out of 72 sites already
    gets there without growing; qpitk.ldosmap's degenerate-boundary
    guard (see test_ldos_is_independent_of_arpack_starting_vector below)
    is what makes this hold at much smaller num_waves too."""
    g = geometry.honeycomb_lattice()
    h = g.get_hamiltonian(has_spin=False)
    r, ldos_r, q, qpi_q = h.get_qpi_impurity(nsuper=6, impurities=[],
            energies=0.3, num_waves=71, nk=4, delta=0.2, write=False)
    assert np.allclose(ldos_r, ldos_r[0], atol=1e-10)  # uniform LDOS
    assert np.max(_away_from_gamma(q, qpi_q)) < 1e-8


def test_ldos_is_independent_of_arpack_starting_vector():
    """Regression check for qpitk.ldosmap._eigenstates_for_energy_window:
    a 6x6 honeycomb supercell has an exactly 15-fold degenerate level at
    its own Gamma point (verified by direct dense diagonalization), so
    num_waves=20 (nearest sigma=0.3) cuts that manifold in half. Summing
    |psi|^2 over an arbitrary partial slice of a degenerate eigenspace
    is not basis-independent, so without the growing guard this is
    sensitive to ARPACK's (by default random) starting vector -- run
    twice and require bit-for-bit identical output, not just "close"."""
    g = geometry.honeycomb_lattice()
    h = g.get_hamiltonian(has_spin=False)
    kwargs = dict(nsuper=6, impurities=[], energies=0.3,
            num_waves=20, nk=2, delta=0.2, write=False)
    r1, ldos1, q1, qpi1 = h.get_qpi_impurity(**kwargs)
    r2, ldos2, q2, qpi2 = h.get_qpi_impurity(**kwargs)
    assert np.array_equal(ldos1, ldos2)


def test_wide_energy_sweep_is_covered_from_a_small_num_waves_hint():
    """Regression check for the margin/coverage guard in
    qpitk.ldosmap._eigenstates_for_energy_window: energies spanning most
    of the bandwidth, requested with a deliberately tiny num_waves=5
    hint (centered near the band edges' midpoint, i.e. far from either
    requested energy), must still grow enough to actually resolve both
    -- and by electron-hole symmetry of nearest-neighbor honeycomb,
    the two symmetric energies must agree."""
    g = geometry.honeycomb_lattice()
    h = g.get_hamiltonian(has_spin=False)
    r, ldos_r, q, qpi_q = h.get_qpi_impurity(nsuper=6, impurities=[],
            energies=[-2.9, 2.9], num_waves=5, nk=1, delta=0.1, write=False)
    assert np.all(np.isfinite(ldos_r))
    assert np.allclose(ldos_r[0], ldos_r[1], rtol=1e-6)
    assert np.mean(ldos_r[0]) > 1e-3  # actually found real weight, not zero


def test_single_impurity_produces_qpi_weight_away_from_gamma():
    """A single onsite impurity breaks the site-to-site symmetry the
    clean-case test above relies on, so its Fourier transform must show
    real, non-vanishing weight away from Gamma -- this is the actual QPI
    signal."""
    g = geometry.honeycomb_lattice()
    h = g.get_hamiltonian(has_spin=False)
    r, ldos_r, q, qpi_q = h.get_qpi_impurity(nsuper=6,
            impurities=[{"position": [0., 0., 0.], "onsite": 3.0}],
            energies=0.3, num_waves=71, nk=4, delta=0.2, write=False)
    assert np.max(_away_from_gamma(q, qpi_q)) > 1e-3


def test_vacancy_impurity_runs_and_differs_from_onsite_impurity():
    """A vacancy spec (modeled as a strong onsite potential, see
    qpitk.impurity.build_impurity_hamiltonian) should produce a
    different LDOS map than a moderate onsite impurity at the same
    site, not silently do nothing or crash."""
    g = geometry.honeycomb_lattice()
    h = g.get_hamiltonian(has_spin=False)
    r, ldos_a, q, qpi_a = h.get_qpi_impurity(nsuper=6,
            impurities=[{"position": [0., 0., 0.], "onsite": 0.5}],
            energies=0.3, num_waves=20, nk=2, delta=0.2, write=False)
    r, ldos_v, q, qpi_v = h.get_qpi_impurity(nsuper=6,
            impurities=[{"position": [0., 0., 0.], "vacancy": True}],
            energies=0.3, num_waves=20, nk=2, delta=0.2, write=False)
    assert np.all(np.isfinite(ldos_v)) and np.all(np.isfinite(qpi_v))
    assert not np.allclose(ldos_a, ldos_v)


def test_qpi_impurity_returns_arrays_with_consistent_shapes_for_multiple_energies(tmp_path, monkeypatch):
    """Regression check for the multi-energy / file-writing path: shapes
    of the returned arrays must match the number of requested energies,
    and the on-disk MULTIQPI-style output (one LDOS + one QPI file per
    energy, plus a combined DOS.OUT) must actually get written."""
    monkeypatch.chdir(tmp_path)
    g = geometry.square_lattice()
    h = g.get_hamiltonian(has_spin=False)
    energies = np.array([-1.0, 0.5])
    r, ldos_r, q, qpi_q = h.get_qpi_impurity(nsuper=4,
            impurities=[{"position": [0., 0., 0.], "onsite": 2.0}],
            energies=energies, num_waves=10, nk=2, delta=0.3,
            output_folder="QPI_IMPURITY_TEST")
    nsites = len(r)
    nq = 4 * 4  # nsuper x nsuper commensurate q-points
    assert ldos_r.shape == (2, nsites)
    assert q.shape == (nq, 3)
    assert qpi_q.shape == (2, nq)
    assert np.all(np.isfinite(ldos_r)) and np.all(np.isfinite(qpi_q))
    dos = np.genfromtxt("QPI_IMPURITY_TEST/DOS.OUT")
    assert dos.shape == (2, 2)
    assert np.allclose(dos[:, 0], energies)
