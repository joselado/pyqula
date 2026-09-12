"""Souza-Marzari-Vanderbilt disentanglement (``num_wann`` smaller than the
selected band range, plus an optional frozen inner window) as wired into
``get_wannier_hamiltonian``.

These tests assert invariants rather than recorded numbers, because the
invariants are exactly what disentanglement promises and the numbers are
not reproducible: the default trial projection is a fresh random draw
each call and the CG/Z-matrix minimizations land on whichever local
optimum they find.

The invariants:

* **frozen states are reproduced exactly.** Every eigenvalue of the
  original Hamiltonian that falls inside the frozen inner window at a
  wannierization-mesh k-point must also be an eigenvalue of the
  reconstructed Wannier Hamiltonian there. That is the defining property
  of the frozen window (SMV Sec. III.C: the frozen states are *kept*,
  not optimized), and it holds for any converged or unconverged
  disentanglement, so it is a real, seed-independent check.
* **nothing outside the outer window leaks in.** The optimal subspace is
  a compression of the outer-window states only, so by Cauchy
  interlacing every eigenvalue of the reconstructed Hamiltonian must lie
  inside ``[dis_win_min, dis_win_max]``. This is the check that actually
  exercises the window-row bookkeeping (``lwindow`` -> which rows of the
  eigenvalue/eigenvector arrays the optimal-subspace matrix refers to);
  with an outer window covering every band it is vacuous, so the test
  below deliberately uses one that cuts.
* **outside the frozen window the original bands are NOT reproduced**,
  and that is the correct behaviour, not a failure -- disentanglement
  trades exact reproduction of a fixed band set for a smoother subspace.
"""
import numpy as np
import pytest

from pyqula import geometry
from pyqula.klist import kmesh


def _mesh_kpoints(nk, dim):
    return [np.asarray(k)[:dim] for k in kmesh(dim, nk=nk)]


def _spectra(h, hw, nk, dim):
    """(original spectrum, Wannier spectrum) at every wannierization-mesh
    k-point, both sorted ascending."""
    f0, fw = h.get_hk_gen(), hw.get_hk_gen()
    out = []
    for k in _mesh_kpoints(nk, dim):
        k3 = np.zeros(3)
        k3[:dim] = k
        out.append((np.sort(np.linalg.eigvalsh(f0(k3))),
                    np.sort(np.linalg.eigvalsh(fw(k3)))))
    return out


def _frozen_reproduction_error(pairs, froz_min, froz_max):
    """Largest distance from a frozen-window eigenvalue of the original
    Hamiltonian to the nearest eigenvalue of the Wannier one, plus how
    many frozen states were checked (a zero count would make the check
    vacuous)."""
    worst, count = 0.0, 0
    for e0, ew in pairs:
        for e in e0[(e0 >= froz_min) & (e0 <= froz_max)]:
            count += 1
            worst = max(worst, float(np.min(np.abs(ew - e))))
    return worst, count


def test_frozen_window_is_reproduced_exactly_on_the_mesh():
    """Graphene's pz bands are the canonical entangled case: the valence
    band is degenerate with the conduction band at K, so no fixed band
    subset is smoothly separable there. Disentangling one Wannier
    function out of both bands, with the deep part of the valence band
    frozen, must reproduce every frozen state exactly."""
    g = geometry.honeycomb_lattice()
    h = g.get_hamiltonian(has_spin=False)
    froz_max = -1.0

    hw = h.get_wannier_hamiltonian(bands=[0, 1], num_wann=1, nk=12,
                                   dis_froz_max=froz_max, cutoff=0.0)

    assert hw.intra.shape == (1, 1)
    pairs = _spectra(h, hw, nk=12, dim=2)
    worst, count = _frozen_reproduction_error(pairs, -np.inf, froz_max)
    assert count > 50, "frozen window caught too few states to be a real check"
    assert worst < 1e-8


def test_disentangled_band_is_not_the_original_band_outside_the_frozen_window():
    """The expected and correct behaviour: inside the frozen window the
    original band is reproduced to machine precision, outside it the
    optimally-connected subspace is a genuinely different (smoother)
    one. Both halves are asserted together so this cannot pass by the
    reconstruction being trivially equal to the valence band."""
    g = geometry.honeycomb_lattice()
    h = g.get_hamiltonian(has_spin=False)
    froz_max = -1.0

    hw = h.get_wannier_hamiltonian(bands=[0, 1], num_wann=1, nk=12,
                                   dis_froz_max=froz_max, cutoff=0.0)

    inside, outside = 0.0, 0.0
    n_free = 0
    for e0, ew in _spectra(h, hw, nk=12, dim=2):
        # the single Wannier band must always stay inside the two-band
        # manifold it was extracted from (Cauchy interlacing)
        assert e0[0] - 1e-8 <= ew[0] <= e0[-1] + 1e-8
        if e0[0] <= froz_max:
            inside = max(inside, abs(ew[0] - e0[0]))
        else:
            n_free += 1
            outside = max(outside, abs(ew[0] - e0[0]))
    assert inside < 1e-8
    assert n_free > 0, "frozen window left no entangled k-points"
    assert outside > 1e-3, ("outside the frozen window the disentangled band "
                            "should differ from the original valence band")


def test_outer_window_bounds_the_reconstructed_spectrum():
    """The row-bookkeeping check. With an outer window that excludes the
    lowest band at some k-points but not others, the optimal subspace is
    built from a *different* set of bands at different k. If the mapping
    from ``lwindow`` to eigenvalue rows were off, an excluded band's
    energy would leak into the reconstruction and show up below
    ``dis_win_min``."""
    g = geometry.chain().supercell(3)
    h = g.get_hamiltonian(has_spin=False)
    h.add_onsite([0.0, 0.3, -0.2])
    win_min, win_max, froz_max = -1.5, 2.5, -0.3

    hw = h.get_wannier_hamiltonian(bands=[0, 2], num_wann=2, nk=12,
                                   dis_win_min=win_min, dis_win_max=win_max,
                                   dis_froz_min=win_min, dis_froz_max=froz_max,
                                   cutoff=0.0)

    lwindow = hw.wannier_run_result.lwindow
    assert not lwindow.all(), "outer window did not actually cut any band"

    pairs = _spectra(h, hw, nk=12, dim=1)
    worst, count = _frozen_reproduction_error(pairs, win_min, froz_max)
    assert count > 0
    assert worst < 1e-8
    for _, ew in pairs:
        assert ew[0] >= win_min - 1e-8
        assert ew[-1] <= win_max + 1e-8


def test_disentangled_wannier_functions_stay_consistent_with_the_hamiltonian():
    """``wannier_functions`` and the returned hoppings are built from the
    same gauge and must stay tied together: W(k) := sum_R
    wannier_functions[R] exp(-i 2 pi R.k) must be an isometry with
    W(k)^dagger h(k) W(k) == h_wannier(k) at every mesh k-point -- the
    same invariant the fixed-window path is held to (see
    test_wannier_functions_reproduce_hamiltonian_at_every_mesh_kpoint),
    and the only thing that checks the disentangled eigenvector
    bookkeeping, which the spectrum never touches."""
    g = geometry.chain().supercell(3)
    h = g.get_hamiltonian(has_spin=False)
    h.add_onsite([0.0, 0.3, -0.2])

    hw = h.get_wannier_hamiltonian(bands=[0, 2], num_wann=2, nk=12,
                                   dis_win_min=-1.5, dis_win_max=2.5,
                                   dis_froz_min=-1.5, dis_froz_max=-0.3,
                                   cutoff=0.0)

    f0, fw = h.get_hk_gen(), hw.get_hk_gen()
    maxerr, weight = 0.0, 0.0
    for kfrac in kmesh(1, nk=12):
        k3 = np.zeros(3)
        k3[0] = kfrac[0]
        Wk = sum(m * np.exp(-1j * 2 * np.pi * np.dot(R, kfrac))
                 for R, m in hw.wannier_functions.items())
        maxerr = max(maxerr, float(np.max(np.abs(Wk.conj().T @ f0(k3) @ Wk - fw(k3)))))
        maxerr = max(maxerr, float(np.max(np.abs(Wk.conj().T @ Wk - np.eye(2)))))
    weight = sum(np.sum(np.abs(m) ** 2) for m in hw.wannier_functions.values())
    assert maxerr < 1e-8
    assert abs(weight - 2.0) < 1e-8  # one unit of weight per Wannier function


def test_num_wann_defaults_to_the_band_count():
    """Not passing num_wann leaves the fixed-window path exactly as it
    was: num_wann == len(bands), no disentanglement, exact reproduction
    of the selected bands."""
    g = geometry.ladder()
    h = g.get_hamiltonian(has_spin=False)
    hw = h.get_wannier_hamiltonian(bands=[0, 1], nk=16, cutoff=0.0)
    assert hw.wannier_num_wann == 2
    assert hw.wannier_disentanglement_window is None


def test_window_argument_without_num_wann_raises():
    """A frozen window with no num_wann would silently skip
    disentanglement altogether (the engine gates on num_bands>num_wann)
    and hand back the plain fixed-window Hamiltonian -- refuse instead."""
    g = geometry.honeycomb_lattice()
    h = g.get_hamiltonian(has_spin=False)
    with pytest.raises(ValueError):
        h.get_wannier_hamiltonian(bands=[0, 1], nk=6, dis_froz_max=-1.0)


def test_frozen_min_without_frozen_max_raises():
    g = geometry.honeycomb_lattice()
    h = g.get_hamiltonian(has_spin=False)
    with pytest.raises(ValueError):
        h.get_wannier_hamiltonian(bands=[0, 1], num_wann=1, nk=6, dis_froz_min=-3.0)


def test_num_wann_larger_than_band_range_raises():
    g = geometry.honeycomb_lattice()
    h = g.get_hamiltonian(has_spin=False)
    with pytest.raises(ValueError):
        h.get_wannier_hamiltonian(bands=[0, 0], num_wann=2, nk=6)


def test_disentanglement_with_bdg_not_implemented():
    """The electron-hole post-processing assumes num_wann == len(bands)
    (the selection has to be pair-closed); disentanglement breaks that."""
    g = geometry.chain()
    h = g.get_hamiltonian(has_spin=True)
    h.setup_nambu_spinor()
    h.add_swave(0.3)
    with pytest.raises(NotImplementedError):
        h.get_wannier_hamiltonian(bands=[0, 3], num_wann=2, nk=6)


def test_disentanglement_with_symmetries_not_implemented():
    """wannierpy's symmetry-adapted path explicitly does not cover the
    frozen-window case (see _engine/sitesym.py), and the post-hoc
    point-group validation assumes a full multiplet band selection."""
    g = geometry.honeycomb_lattice()
    h = g.get_hamiltonian(has_spin=False)
    with pytest.raises(NotImplementedError):
        h.get_wannier_hamiltonian(bands=[0, 1], num_wann=1, nk=6,
                                  dis_froz_max=-1.0, symmetries="auto")


def test_disentanglement_with_auto_split_clusters_not_implemented():
    g = geometry.honeycomb_lattice()
    h = g.get_hamiltonian(has_spin=False)
    with pytest.raises(NotImplementedError):
        h.get_wannier_hamiltonian(bands=[0, 1], num_wann=1, nk=6,
                                  dis_froz_max=-1.0, auto_split_clusters=True)


def test_internal_find_u_contracts_window_relative_band_rows():
    """``u_matrix_opt``'s rows are window-relative -- row i at k-point k is
    the i-th band *inside the outer window* there, i.e. absolute band
    ``nfirstwin[k]+i``. That is the convention ``dis_project``,
    ``slim_m``, ``dis_extract`` and ``_rotate_m`` all use. ``A_matrix``
    is the one array the engine never slims, so the rows it is contracted
    against have to carry the same ``nfirstwin`` offset; reading it from
    row 0 instead pairs the optimal-subspace states with the overlaps of
    the wrong bands whenever the outer window cuts a band from below.

    Checked here against the definition rather than through a
    Hamiltonian, because it is invisible from outside: the result seeds
    the Wannierisation gauge, and any seed is unitary, so the
    reconstructed spectrum is right either way and only the localization
    suffers.
    """
    from pyqula.wanniertk.wannierpy._engine.disentangle import dis_windows, internal_find_u

    num_bands, num_kpts, num_wann = 4, 2, 2
    # at k=0 the lowest band sits below dis_win_min (nfirstwin=1); at k=1
    # every band is inside the window (nfirstwin=0)
    eigval = np.array([[-3.0, -0.5],
                       [-1.0, 0.0],
                       [0.5, 1.0],
                       [2.0, 2.5]])
    windows = dis_windows(eigval, num_wann, -2.0, 3.0, False, 0.0, 0.0)
    assert list(windows.nfirstwin) == [1, 0]

    rng = np.random.default_rng(1234)
    def _rand(shape):
        return rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    A_matrix = _rand((num_bands, num_wann, num_kpts))
    u_matrix_opt = np.zeros((num_bands, num_wann, num_kpts), dtype=complex)
    for k in range(num_kpts):
        nd = int(windows.ndimwin[k])
        q, _ = np.linalg.qr(_rand((nd, num_wann)))
        u_matrix_opt[:nd, :, k] = q

    got = internal_find_u(u_matrix_opt, A_matrix, windows, num_wann)

    for k in range(num_kpts):
        n0, nd = int(windows.nfirstwin[k]), int(windows.ndimwin[k])
        caa = u_matrix_opt[:nd, :, k].conj().T @ A_matrix[n0:n0 + nd, :, k]
        Z, _, Vh = np.linalg.svd(caa)
        assert np.allclose(got[:, :, k], Z @ Vh, atol=1e-12)
