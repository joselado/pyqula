import numpy as np
import pytest

from pyqula import geometry
from pyqula.klist import kmesh


def _max_band_diff(h_ref, h_wann, band_indices, ks, dim):
    """Compare h_wann's full spectrum against h_ref's band_indices-selected
    spectrum, at every k in ks (fractional, dim components used)."""
    f_ref = h_ref.get_hk_gen()
    f_wann = h_wann.get_hk_gen()
    maxdiff = 0.0
    for k in ks:
        k3 = np.zeros(3)
        k3[:dim] = np.asarray(k)[:dim]
        e_ref = np.sort(np.linalg.eigvalsh(f_ref(k3)))[list(band_indices)]
        e_wann = np.sort(np.linalg.eigvalsh(f_wann(k3)))
        maxdiff = max(maxdiff, float(np.max(np.abs(e_wann - e_ref))))
    return maxdiff


def test_requires_num_bands_or_band_indices():
    g = geometry.honeycomb_lattice()
    h = g.get_hamiltonian(has_spin=False)
    with pytest.raises(ValueError):
        h.get_wannier_hamiltonian(nk=6)


def test_requires_periodic_hamiltonian():
    g = geometry.honeycomb_lattice()
    g.dimensionality = 0
    h = g.get_hamiltonian(has_spin=False)
    h.dimensionality = 0
    with pytest.raises(NotImplementedError):
        h.get_wannier_hamiltonian(num_bands=1)


def test_ladder_lowest_band_reproduces_spectrum_exactly():
    """The symmetric ladder's lowest band is an exact, k-independent
    bonding combination (see wannierpy's examples/pyqula_ladder.py) --
    a genuine physical zero-spread case, useful as a tight correctness
    check: with cutoff=0 (keep every real-space hopping so the Fourier
    reconstruction is exact, not just approximately truncated) the
    Wannierized single-band Hamiltonian must reproduce the original
    lowest band to numerical precision at every mesh k-point."""
    g = geometry.ladder()
    h = g.get_hamiltonian(has_spin=False)
    h1 = h.get_wannier_hamiltonian(num_bands=1, nk=16, cutoff=0.0)

    assert h1.intra.shape == (1, 1)
    assert h1.wannier_spread_total < 1e-6

    ks = kmesh(1, nk=16)
    maxdiff = _max_band_diff(h, h1, [0], ks, dim=1)
    assert maxdiff < 1e-8


def test_ladder_full_manifold_reproduces_spectrum_exactly():
    """num_wann == num_bands == num_orbitals (no truncation): the
    maximally localized Wannier functions are provably exact (zero
    spread) for an algebraic reason independent of the physics -- see
    wannierpy's examples/pyqula_ladder_both_bands.py -- and the
    reconstructed 2-band Hamiltonian must reproduce the full original
    spectrum exactly."""
    g = geometry.ladder()
    h = g.get_hamiltonian(has_spin=False)
    h2 = h.get_wannier_hamiltonian(num_bands=2, nk=16, cutoff=0.0)

    assert h2.intra.shape == (2, 2)
    assert h2.wannier_spread_total < 1e-6

    ks = kmesh(1, nk=16)
    maxdiff = _max_band_diff(h, h2, [0, 1], ks, dim=1)
    assert maxdiff < 1e-8


def test_gapped_honeycomb_valence_band_reproduces_spectrum():
    """A genuinely non-trivial case (dispersive valence band, no
    algebraic degeneracy to trivialize the result): a staggered
    honeycomb lattice (sublattice potential opens a gap) wannierized on
    its single valence band must reproduce that band on the
    wannierization mesh to high accuracy."""
    g = geometry.honeycomb_lattice()
    h = g.get_hamiltonian(has_spin=False)
    h.add_onsite([0.8, -0.8])

    h1 = h.get_wannier_hamiltonian(num_bands=1, nk=12, cutoff=0.0)
    assert h1.intra.shape == (1, 1)

    ks = kmesh(2, nk=12)
    maxdiff = _max_band_diff(h, h1, [0], ks, dim=2)
    assert maxdiff < 1e-4


def test_default_cutoff_still_reproduces_spectrum_reasonably():
    """The default cutoff (1e-6) truncates negligible long-range real-space
    hoppings for compactness, trading exact mesh-point reproduction for a
    small, bounded interpolation error -- check that error stays small."""
    g = geometry.honeycomb_lattice()
    h = g.get_hamiltonian(has_spin=False)
    h.add_onsite([0.8, -0.8])

    h1 = h.get_wannier_hamiltonian(num_bands=1, nk=12)  # default cutoff
    ks = kmesh(2, nk=12)
    maxdiff = _max_band_diff(h, h1, [0], ks, dim=2)
    assert maxdiff < 1e-3


def test_explicit_band_indices_selects_requested_band():
    """band_indices=[1] (the upper/antibonding band of the ladder, not the
    default lowest-band choice) must wannierize *that* band, not band 0."""
    g = geometry.ladder()
    h = g.get_hamiltonian(has_spin=False)
    h1 = h.get_wannier_hamiltonian(band_indices=[1], nk=16, cutoff=0.0)

    assert h1.wannier_band_indices == [1]
    ks = kmesh(1, nk=16)
    maxdiff = _max_band_diff(h, h1, [1], ks, dim=1)
    assert maxdiff < 1e-8


def test_num_bands_larger_than_available_raises():
    g = geometry.ladder()
    h = g.get_hamiltonian(has_spin=False)
    with pytest.raises(ValueError):
        h.get_wannier_hamiltonian(num_bands=3, nk=8)


def test_result_is_multicell_hamiltonian_with_diagnostics():
    g = geometry.ladder()
    h = g.get_hamiltonian(has_spin=False)
    h1 = h.get_wannier_hamiltonian(num_bands=1, nk=12)

    assert h1.is_multicell
    assert h1.has_spin is False
    assert h1.wannier_centres.shape == (1, 3)
    assert h1.wannier_spreads.shape == (1,)
    assert h1.wannier_setup_result is not None
    assert h1.wannier_run_result is not None
