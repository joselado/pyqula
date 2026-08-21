"""The magnon dispersion itself, once the Goldstone mode is in place.

test_goldstone.py checks that the acoustic branch starts at zero; these
check that it goes somewhere sensible afterwards, and that the public
entry points return what they say they do.
"""
import numpy as np
import pytest

from pyqula import geometry

NK = 6


def _neel_honeycomb(nk=NK):
    g = geometry.honeycomb_lattice()
    h = g.get_hamiltonian()
    return h.get_mean_field_hamiltonian(U=3.0, filling=0.5, mf="antiferro",
                                        nk=nk, maxerror=1e-10)


def test_the_acoustic_branch_of_an_antiferromagnet_is_linear_in_q():
    """An antiferromagnet has linearly dispersing spin waves, E = c|Q|,
    unlike a ferromagnet's quadratic ones. Measured on this state:
    0.0500, 0.0999, 0.1985 at Q = 0.01, 0.02, 0.04, i.e. a straight line
    through the origin to better than a percent."""
    h = _neel_honeycomb(nk=8)
    qs = np.array([0.01, 0.02, 0.04])
    es = np.array([h.get_magnon_energies(nk=8, Q=[q, 0., 0.], n=1)[0].real
                   for q in qs])
    assert np.all(es > 0)
    slopes = es/qs
    assert np.max(np.abs(slopes - slopes[0]))/slopes[0] < 0.05


def test_magnon_bands_scan_a_path_and_reach_zero_at_gamma():
    h = _neel_honeycomb(nk=4)
    qs, es = h.get_magnon_bands(method="tdhf", nk=4, nq=4, n=2)
    assert len(qs) == len(es)
    assert len(set(qs.tolist())) == 4  # one group of energies per q-point
    # the path ends at Gamma, where the acoustic branch must vanish. The
    # test is on the REAL part: the zero eigenvalue of a Casida matrix is
    # defective, so what is left of it at a finite SCF tolerance is a
    # residual imaginary part of order sqrt(that tolerance) -- 5e-5 here --
    # while the energy itself is zero to 1e-13
    last = es[qs == qs.max()]
    assert np.min(np.abs(last.real)) < 1e-6
    assert np.min(np.abs(last)) < 1e-3  # ... and the mode is there at all


def test_the_rpa_and_tdhf_methods_are_both_reachable():
    """method= dispatches without changing the old default: the site-basis
    RPA still refuses a non-onsite interaction, and still runs for an
    onsite one."""
    h = _neel_honeycomb(nk=4)
    qs, es = h.get_magnon_bands(method="tdhf", nk=4, nq=2, n=1)
    assert len(es) == 2
    with pytest.raises(ValueError):
        h.get_magnon_bands(method="nonsense")


def test_a_non_magnetic_mean_field_is_reported_rather_than_dispersed():
    """An RPA or TDHF calculation on top of an unpolarized reference runs
    perfectly happily and means nothing. Here it cannot even be set up --
    the spin generator has no weight in the pair basis -- and says so."""
    from testutils import gapped_honeycomb
    h = gapped_honeycomb(mass=1.0)  # gapped, but not magnetic
    with pytest.raises(ValueError):
        h.get_goldstone_residual(nk=4, V=np.zeros((4, 4), dtype=np.complex128))
