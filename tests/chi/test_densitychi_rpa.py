"""Oracle tests for h.get_densitychi_RPA (chitk/densitychi.py).

This entry point had no coverage at all. Its sibling plasmon_bands is tested,
but does NOT call densitychi_RPA (verified: zero calls) -- they are siblings in
the same module, not layers, so that gave no transitive coverage.

The oracle here is the RPA structure itself, which is analytic:

    chi_RPA = chi_0 (1 - V_q chi_0)^{-1}

so for a small coupling the series gives, to first order,

    chi_RPA = chi_0 + chi_0 V_q chi_0 + O(V^2)

Testing against that expansion checks the kernel AND the interaction
coefficient, independently of the implementation -- rather than pinning
whatever the code currently returns.

It also pins a specific documented claim. densitychi_RPA's own docstring
derives the charge-channel coefficient of the onsite U as

    U n_up n_down = (U/4) n^2 - (U/4) m_z^2  ->  a_charge = U/2

so a pure-U interaction must enter the kernel as U/2 on the diagonal, not U.
That factor is exactly the kind of thing that is easy to get wrong and
invisible without an external check -- the spin channel uses -2U for the same
physical U.
"""
import numpy as np
import pytest

from pyqula import geometry, parallel
from testutils import temporary_attr

ENERGIES = np.linspace(0.5, 1.2, 4)
NK, DELTA = 4, 0.2


def _h():
    g = geometry.honeycomb_lattice()
    return g.get_hamiltonian(has_spin=False)


def _chi(h, **kw):
    with temporary_attr(parallel, "cores", 1):
        es, chis = h.get_densitychi_RPA(energies=ENERGIES, nk=NK, delta=DELTA,
                                        **kw)
    return es, np.asarray(chis)


def test_zero_interaction_returns_the_bare_susceptibility_exactly():
    """With every coupling zero the RPA kernel is the identity, so the result
    must be chi_0 EXACTLY (not approximately) -- any drift here means the
    kernel is being applied when it should not be."""
    h = _h()
    _, chi_default = _chi(h)
    _, chi_zero = _chi(h, U=0.0, V1=0.0, V2=0.0, V3=0.0)
    assert np.array_equal(chi_default, chi_zero)


def test_small_U_matches_the_first_order_RPA_series_with_the_documented_U_over_2():
    """chi_RPA = chi_0 + chi_0 V chi_0 + O(V^2), with V = (U/2) * identity for
    a pure onsite U (the coefficient densitychi_RPA's docstring derives).

    Checked by confirming the first-order prediction is far closer to the
    truth than the zeroth-order one, and that the residual scales like U^2.
    A wrong coefficient (e.g. U instead of U/2) fails the first check.
    """
    h = _h()
    _, chi0 = _chi(h)
    n = chi0.shape[-1]
    I = np.identity(n, dtype=np.complex128)

    def residual(U):
        _, chiU = _chi(h, U=U)
        pred = np.array([c0 + c0 @ ((U / 2.0) * I) @ c0 for c0 in chi0])
        first_order = np.max(np.abs(chiU - pred))
        zeroth_order = np.max(np.abs(chiU - chi0))
        return first_order, zeroth_order

    r1, r0 = residual(0.05)
    assert r1 < r0 / 10.0, (
        f"first-order RPA series with a_charge=U/2 should be much closer than "
        f"chi_0 alone (got {r1:.3e} vs {r0:.3e}) -- if this fails, the charge "
        f"kernel coefficient is not U/2")

    # residual must be second order: halving U should cut it by roughly 4
    r_small, _ = residual(0.025)
    assert r_small < r1 / 2.5, (r_small, r1)


def test_wrong_coefficient_U_would_be_rejected():
    """Guard on the guard: the same comparison using a_charge = U (instead of
    U/2) must NOT satisfy the first-order test, otherwise the test above
    would pass for the wrong kernel too."""
    h = _h()
    _, chi0 = _chi(h)
    n = chi0.shape[-1]
    I = np.identity(n, dtype=np.complex128)
    U = 0.05
    _, chiU = _chi(h, U=U)
    good = np.max(np.abs(chiU - np.array([c0 + c0 @ ((U / 2.0) * I) @ c0
                                          for c0 in chi0])))
    bad = np.max(np.abs(chiU - np.array([c0 + c0 @ (U * I) @ c0
                                         for c0 in chi0])))
    assert good < bad, (good, bad)


def test_neighbor_interaction_changes_the_response_and_stays_finite():
    """V1 must actually enter (it is a different code path from U: a
    neighbor-shell hopping matrix rather than a diagonal shift) and must not
    produce NaN/inf at these couplings."""
    h = _h()
    _, chi0 = _chi(h)
    _, chiV = _chi(h, V1=0.3)
    assert np.all(np.isfinite(chiV))
    assert not np.allclose(chiV, chi0)


def test_response_matrix_shape_matches_the_site_count():
    """The charge channel is N-dimensional (one entry per site), NOT the
    spin-doubled 2N that scftk.spinspin's Hartree-Fock helper returns -- a
    distinction densitychi.py's module docstring calls out explicitly."""
    h = _h()
    es, chis = _chi(h)
    nsites = len(h.geometry.r)
    assert chis.shape == (len(ENERGIES), nsites, nsites), chis.shape
    assert len(es) == len(ENERGIES)
