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


def test_a_non_magnetic_mean_field_is_reported_rather_than_dispersed():
    """An RPA or TDHF calculation on top of an unpolarized reference runs
    perfectly happily and means nothing. Here it cannot even be set up --
    the spin generator has no weight in the pair basis -- and says so."""
    from testutils import gapped_honeycomb
    h = gapped_honeycomb(mass=1.0)  # gapped, but not magnetic
    with pytest.raises(ValueError):
        h.get_goldstone_residual(nk=4, V=np.zeros((4, 4), dtype=np.complex128))
