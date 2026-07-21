import numpy as np
import pytest

from pyqula import geometry, islands
from pyqula.hamiltonians import Hamiltonian
from pyqula.klist import kmesh
from pyqula.multihopping import MultiHopping
from pyqula.wanniertk.wannierhamiltonian import WannierHamiltonian


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


def test_requires_bands():
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
        h.get_wannier_hamiltonian(bands=[0, 0])


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
    h1 = h.get_wannier_hamiltonian(bands=[0, 0], nk=16, cutoff=0.0)

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
    h2 = h.get_wannier_hamiltonian(bands=[0, 1], nk=16, cutoff=0.0)

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

    h1 = h.get_wannier_hamiltonian(bands=[0, 0], nk=12, cutoff=0.0)
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

    h1 = h.get_wannier_hamiltonian(bands=[0, 0], nk=12)  # default cutoff
    ks = kmesh(2, nk=12)
    maxdiff = _max_band_diff(h, h1, [0], ks, dim=2)
    assert maxdiff < 1e-3


def test_explicit_band_range_selects_requested_band():
    """bands=[1,1] (the upper/antibonding band of the ladder, not the
    default lowest-band choice) must wannierize *that* band, not band 0."""
    g = geometry.ladder()
    h = g.get_hamiltonian(has_spin=False)
    h1 = h.get_wannier_hamiltonian(bands=[1, 1], nk=16, cutoff=0.0)

    assert h1.wannier_band_indices == [1]
    ks = kmesh(1, nk=16)
    maxdiff = _max_band_diff(h, h1, [1], ks, dim=1)
    assert maxdiff < 1e-8


def test_bands_larger_than_available_raises():
    g = geometry.ladder()
    h = g.get_hamiltonian(has_spin=False)
    with pytest.raises(ValueError):
        h.get_wannier_hamiltonian(bands=[0, 2], nk=8)


def test_result_is_multicell_hamiltonian_with_diagnostics():
    g = geometry.ladder()
    h = g.get_hamiltonian(has_spin=False)
    h1 = h.get_wannier_hamiltonian(bands=[0, 0], nk=12)

    assert h1.is_multicell
    assert h1.has_spin is False
    assert h1.wannier_centres.shape == (1, 3)
    assert h1.wannier_spreads.shape == (1,)
    assert h1.wannier_setup_result is not None
    assert h1.wannier_run_result is not None


def test_result_is_wannier_hamiltonian_instance():
    """get_wannier_hamiltonian must return a WannierHamiltonian -- a
    Hamiltonian subclass, so every ordinary Hamiltonian method (get_bands,
    get_dos, ...) keeps working unchanged -- not a plain Hamiltonian."""
    g = geometry.ladder()
    h = g.get_hamiltonian(has_spin=False)
    h1 = h.get_wannier_hamiltonian(bands=[0, 0], nk=12)

    assert isinstance(h1, WannierHamiltonian)
    assert isinstance(h1, Hamiltonian)
    (ks, es) = h1.get_bands()  # inherited method must work unmodified
    assert len(es) > 0


def test_wannier_functions_reproduce_hamiltonian_at_every_mesh_kpoint():
    """wannier_functions[R][o,n] is the amplitude of Wannier function n
    (translated to cell R) on orbital o of the *original* Hamiltonian h,
    in h's own orbital basis. Physically this means W(k) := sum_R
    wannier_functions[R] * exp(-i*2*pi*R.k) must be an isometry from h's
    orbital space into the Wannierized Hamiltonian's, related to it by
    W(k)^dagger @ h(k) @ W(k) == h1(k) exactly at every mesh k-point --
    the k-space statement of <w_n,0|h|w_n',R> == h1's own hopping[R].
    (The exp(-i*2*pi*R.k) sign -- rather than the H(k)=sum_R h_R
    exp(+i*2*pi*R.k) convention used for ordinary hopping matrices --
    is the fix for issue #30: wannier_functions[R] is a direct
    substitution of pyqula's own Bloch convention |k,o> = (1/sqrt(N))
    sum_R exp(+i*2*pi*k.R)|R,o> into the Wannier-function definition, not
    a Fourier-series coefficient, so it inverts with the opposite sign.)"""
    g = geometry.honeycomb_lattice()
    h = g.get_hamiltonian(has_spin=False)
    h.add_onsite([0.8, -0.8])
    h1 = h.get_wannier_hamiltonian(bands=[0, 0], nk=12, cutoff=0.0)

    hk_gen = h.get_hk_gen()
    hk_gen1 = h1.get_hk_gen()
    ks = kmesh(2, nk=12)
    maxerr = 0.0
    for kfrac in ks:
        k3 = np.zeros(3); k3[:2] = kfrac[:2]
        Wk = sum(m * np.exp(-1j * 2 * np.pi * np.dot(R, kfrac))
                 for R, m in h1.wannier_functions.items())
        lhs = Wk.conj().T @ hk_gen(k3) @ Wk
        rhs = hk_gen1(k3)
        maxerr = max(maxerr, float(np.max(np.abs(lhs - rhs))))
    assert maxerr < 1e-8


def test_wannier_functions_are_normalized():
    """Each Wannier function's total weight (summed over every cell R and
    orbital o) must be 1: W(k) has orthonormal columns by construction
    (a unitary rotation of orthonormal selected-band eigenvectors), so
    Parseval's theorem over the mesh forces this exactly."""
    g = geometry.honeycomb_lattice()
    h = g.get_hamiltonian(has_spin=False)
    h.add_onsite([0.8, -0.8])
    h1 = h.get_wannier_hamiltonian(bands=[0, 0], nk=12, cutoff=0.0)

    total_weight = sum(np.sum(np.abs(m) ** 2) for m in h1.wannier_functions.values())
    assert abs(total_weight - 1.0) < 1e-8


def _bdg_chain():
    g = geometry.chain(2)
    h = g.get_hamiltonian(has_spin=True)
    h.add_rashba(0.2)
    h.add_swave(0.3)
    return h


def test_bdg_full_manifold_is_particle_hole_symmetric_and_exact():
    """Regression test for a real bug: the electron-hole operator induced
    on the CG-converged Wannier gauge is generally *not* the naive
    band-index flip n -> num_orbitals-1-n that _enforce_particle_hole_symmetry
    used to assume (confirmed by direct construction: the two differ by
    O(1), not a small correction) -- the code now recovers the actual
    induced operator from the converged gauge instead. Full manifold
    (num_wann==num_orbitals) is the cleanest check: the raw, unsymmetrized
    reconstruction is always exact regardless of gauge, so any deviation
    here after particle-hole enforcement traces directly to the operator
    used, not to CG non-convergence or a legitimate band-selection issue.
    Also exercises the default full-manifold trial-vector seed (identity):
    confirmed to reliably converge to an (almost) exactly electron-hole
    *covariant* gauge -- constant across the whole mesh, not just unitary
    pointwise -- unlike the paired-orbital heuristic used for partial
    selections (see the two tests below)."""
    h = _bdg_chain()
    n = h.intra.shape[0]
    h1 = h.get_wannier_hamiltonian(bands=[0, n - 1], nk=12, num_iter=3000, conv_tol=1e-12)

    C_wan = h1.wannier_particle_hole_operator
    assert np.max(np.abs(C_wan.conj().T @ C_wan - np.eye(n))) < 1e-8

    for m in h1.get_multihopping().get_dict().values():
        assert np.max(np.abs(C_wan @ np.conj(m) @ np.linalg.inv(C_wan) + m)) < 1e-8

    # With a genuinely covariant gauge, _enforce_particle_hole_symmetry's
    # averaging is a no-op (see its counterpart's docstring), so the
    # reconstruction stays essentially exact -- not just "small enough to
    # pass a loose bound" as it was before the identity-seed default fix.
    ks = kmesh(1, nk=12)
    assert _max_band_diff(h, h1, list(range(n)), ks, dim=1) < 1e-4


def test_bdg_band_selection_across_a_degenerate_multiplet_raises():
    """bands=[3,4] straddles two spin-degenerate multiplets of this model
    (bands {2,3} and {4,5} are each degenerate, so slicing out just {3,4}
    picks an arbitrary, non-covariant 1D cut through each) -- there is no
    well-defined electron-hole operator on that selection, and the code
    must say so rather than silently returning a wrong Hamiltonian (which
    is what it used to do before this check was added). This is the
    *pointwise* failure mode: the induced operator isn't even unitary at a
    single k, caught before the mesh-wide constancy check below runs."""
    h = _bdg_chain()
    with pytest.raises(ValueError, match="not unitary"):
        h.get_wannier_hamiltonian(bands=[3, 4], nk=12, num_iter=2000, conv_tol=1e-12)


def test_bdg_partial_selection_with_non_covariant_gauge_raises():
    """Regression test for a second, subtler real bug found on top of the
    first: even for a partial selection that *is* a well-defined, gapped,
    non-degenerate 2D subspace (so the induced operator is unitary at
    every individual k -- unlike the test above), nothing guarantees the
    CG's converged gauge is the *same* covariant operator at every k. On
    this model (chain(1), Rashba+Zeeman+s-wave, the centred pair [1,2])
    it demonstrably is not (confirmed by direct construction: varies by
    the maximum possible amount, 2.0, for a 2x2 unitary difference) even
    fully converged -- previously this silently reached
    _enforce_particle_hole_symmetry's single-fixed-operator averaging and
    returned a Hamiltonian with an O(0.1-1) wrong spectrum. The mesh-wide
    constancy check now catches this instead of the unitarity check
    above, and the two are distinguished so callers get an accurate
    diagnosis of which failure mode they hit."""
    g = geometry.chain(1)
    h = g.get_hamiltonian(has_spin=True)
    h.add_rashba(0.25)
    h.add_zeeman([0.15, 0.05, 0.2])
    h.add_swave(0.3)
    with pytest.raises(ValueError, match="not the same matrix"):
        h.get_wannier_hamiltonian(bands=[1, 2], nk=12, num_iter=3000, conv_tol=1e-12)


@pytest.mark.parametrize("geom_name", ["triangular_ribbon", "lieb_ribbon"])
def test_ribbon_geometries_with_non_padded_lattice_vectors_do_not_crash(geom_name):
    """Regression test for a real bug: triangular_ribbon/lieb_ribbon (unlike
    e.g. honeycomb_zigzag_ribbon) leave a genuine, lattice-scale, non-padded
    vector in geometry.a2 for a 1D ribbon, which used to crash wannierpy's
    b-vector shell search (kmesh_get) with "unable to satisfy the B1
    completeness relation" -- get_wannier_hamiltonian now builds its own
    clean padding instead of trusting geometry.a2/a3 for non-periodic axes."""
    g = getattr(geometry, geom_name)(3)
    h = g.get_hamiltonian(has_spin=False)
    n = h.intra.shape[0]
    h1 = h.get_wannier_hamiltonian(bands=[0, n - 1], nk=8, num_iter=1500, cutoff=0.0)

    ks = kmesh(1, nk=8)
    assert _max_band_diff(h, h1, list(range(n)), ks, dim=1) < 1e-6


def _wrap_finite_as_fake_1d(h0, boxsize=1000.0):
    """The standard "Gamma-point-only Wannier functions of a finite
    system" trick: wrap a dimensionality=0 (molecular/island) Hamiltonian
    as dimensionality=1 with a huge lattice vector (periodic images never
    couple) and a single intracell hopping equal to the original intra
    matrix, then Wannierize with nk=1."""
    g2 = h0.geometry.copy()
    g2.dimensionality = 1
    g2.a1 = np.array([boxsize, 0.0, 0.0])
    h = Hamiltonian(g2)
    h.has_spin = h0.has_spin
    h.has_eh = getattr(h0, "has_eh", False)
    h.is_sparse = False
    h.is_multicell = True
    h.set_multihopping(MultiHopping({(0, 0, 0): np.array(h0.intra)}))
    return h


def test_requires_periodic_hamiltonian_rejects_genuine_0d():
    """Direct Wannierization of a dimensionality=0 Hamiltonian is not
    supported (no Bloch phases to build a k-mesh from) and must say so
    rather than doing something undefined."""
    g = islands.get_geometry(name="triangular", n=1.4, nedges=3)
    h = g.get_hamiltonian(has_spin=False)
    with pytest.raises(NotImplementedError):
        h.get_wannier_hamiltonian(bands=[0, 0], nk=1)


def test_bdg_island_via_gamma_point_trick_reproduces_spectrum_exactly():
    """Regression test for a real bug: islands.get_geometry's construction
    pipeline (supercell -> polygon cut -> remove_unibonded) leaves a stale
    geometry.frac_r array cached from an intermediate, larger supercell
    even after sculpt.remove/remove_sites correctly reset has_fractional
    to False (that flag reset doesn't erase the old array, and
    _particle_hole_operator read frac_r directly without checking the
    flag first, unlike every other frac_r access in this module) --
    causing a spurious "got num_orbitals=112 for 81 sites" NotImplementedError
    on a 28-site island's Nambu Hamiltonian (112 == 4*28, but frac_r's
    stale length was 81, an intermediate pre-cut supercell size). Fixed by
    using geometry.r (always current) for the site count instead, and by
    having sculpt.remove/remove_sites delete the stale frac_r array
    outright rather than just flagging it stale.

    This also exercises the electron-hole-covariance requirement discussed
    in test_bdg_partial_selection_with_non_covariant_gauge_raises for a
    *partial* band selection: with nk=1 there is only ever one mesh
    k-point (self-paired, k=-k=0), so the "same operator at every k" check
    is trivially satisfied and partial BdG selections work here, unlike on
    a genuinely dispersive (nk>1) lattice."""
    g = islands.get_geometry(name="triangular", n=1.4, nedges=3)
    h0 = g.get_hamiltonian(has_spin=True)
    h0.add_rashba(0.2)
    h0.add_zeeman([0.15, 0.05, 0.2])
    h0.add_swave(0.3)
    n = h0.intra.shape[0]
    h = _wrap_finite_as_fake_1d(h0)

    mid = n // 2
    hwan = h.get_wannier_hamiltonian(bands=[mid - 1, mid], nk=1, num_iter=1000)

    e_ref = np.sort(np.linalg.eigvalsh(h0.intra))[[mid - 1, mid]]
    e_wan = np.sort(np.linalg.eigvalsh(hwan.intra))
    assert np.max(np.abs(e_ref - e_wan)) < 1e-8


def test_backfolded_supercell_bands_respect_marzari_vanderbilt_lower_bound():
    """Regression test for a real bug (GitHub issue #29): the position
    regauging (``phase``/``wrap_phase`` in ``_build_overlaps``) that
    converts pyqula's periodic-gauge eigenvector *coefficients* into
    Wannier90's "convention I" had the wrong sign. Converting the *basis
    vectors* from periodic gauge to convention I multiplies them by
    ``exp(+i2*pi*k.tau)`` -- but a coefficient picks up the *opposite*
    phase from the basis vector it multiplies (writing the same state as
    ``sum_b c_b|b>_perio = sum_b c'_b|b>_conv1`` and substituting
    ``|b>_perio = exp(-i2*pi*k.tau_b)|b>_conv1`` gives
    ``c'_b = c_b * exp(-i2*pi*k.tau_b)``), and ``C_full`` stores
    coefficients, not basis vectors. The wrong sign left the CG
    minimizing against corrupted overlap data -- innocuous-looking for
    many simple/symmetric geometries (where it happens to have no
    effect), but for a general geometry it silently inflated the
    converged spread, sometimes enormously (4x+ the true value in the
    case that exposed it: a band group with an internal exact
    degeneracy/touching point, e.g. a backfolded band in a supercell).

    Test: Wannierize a single isolated (gapped) band in a small "primitive"
    cell (spread S0, no ambiguity -- single band, nothing to get wrong
    about internal degeneracies), then a 3x supercell of the same
    geometry's bottom 3 (backfolded copies of that same band, touching at
    two points in the BZ). Marzari-Vanderbilt theory guarantees the
    gauge-invariant spread is a lower bound for *any* valid gauge of that
    3-band manifold -- and 3 copies of the single-band Wannier function,
    each merely translated (translation doesn't change a function's own
    spread), is one such valid gauge, with total spread exactly 3*S0. So
    the supercell's converged total spread must not exceed 3*S0 (up to
    normal CG/mesh convergence slack) -- and, since translation symmetry
    relates the 3 backfolded copies, their *individual* spreads should
    come out close to equal too, not wildly lopsided."""
    g = geometry.square_ribbon(3)
    g0 = g.get_supercell(3)
    g1 = g0.remove(g0.closest_index([0., 0., 0.]))  # isolated "defect" band
    g2 = g1.get_supercell(3)  # defect repeated 3x -> bands [0,1,2] touch

    h1 = g1.get_hamiltonian(has_spin=False)
    h2 = g2.get_hamiltonian(has_spin=False)

    hwan1 = h1.get_wannier_hamiltonian(bands=[0, 0], nk=16, num_iter=200)
    S0 = hwan1.wannier_spreads[0]

    hwan2 = h2.get_wannier_hamiltonian(bands=[0, 2], nk=12, num_iter=300)
    bound = 3 * S0
    assert hwan2.wannier_spread_total < 1.5 * bound  # generous CG/mesh slack
    spreads = hwan2.wannier_spreads
    assert np.max(spreads) / np.min(spreads) < 1.5  # roughly equal, not lopsided
