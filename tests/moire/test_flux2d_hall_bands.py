import numpy as np

from pyqula import geometry
from pyqula import specialhamiltonian


def _flux_spectrum(n=4, m=1, nk=41):
    """Sorted band energies of specialhamiltonian.flux2d, which puts m flux
    quanta through an n-cell supercell of a honeycomb lattice by a Peierls
    substitution."""
    g = geometry.honeycomb_lattice()
    h = specialhamiltonian.flux2d(g, n=n, m=m)
    (k, e) = h.get_bands(nk=nk)
    return np.sort(np.array(e))


def test_flux2d_spectrum_is_chiral_and_time_reversal_symmetric(tmp_path,
                                                                monkeypatch):
    """Two exact symmetries of a Peierls-substituted bipartite lattice.

    The Peierls phases sit on the bonds, which all connect the two
    sublattices, so the chiral symmetry survives the field and the spectrum
    stays symmetric under E -> -E. That symmetry is the whole content of the
    old sum(e) reference (sum(e) = sum_k Tr H(k) = 0), and it holds for
    every flux, every supercell and every hopping -- so the old assertion
    saw nothing about the field. Asserted here directly, next to the second
    symmetry it does *not* imply: reversing the field leaves the spectrum
    unchanged."""
    monkeypatch.chdir(tmp_path)  # get_bands writes BANDS.OUT to cwd
    e = _flux_spectrum(n=4, m=1)
    assert np.allclose(e, -e[::-1], atol=1e-9)
    assert np.allclose(e, _flux_spectrum(n=4, m=-1), atol=1e-9)


def test_flux2d_fragments_the_honeycomb_bands_into_hofstadter_subbands(tmp_path,
                                                                       monkeypatch):
    """With no field the supercell just refolds the pristine honeycomb
    bands, whose exact half-bandwidth is 3t: the spectrum runs from -3 to 3
    and is continuous apart from the single band-centre feature. Switching
    on one flux quantum per four cells narrows the bandwidth and breaks the
    two bands into Hofstadter sub-bands, opening four gaps wider than 0.1
    along the k-path.

    The gap count is a recorded structural fingerprint of the flux 1/4
    spectrum rather than a closed-form invariant -- a Hofstadter spectrum at
    a given rational flux has no cheap analytic one -- but it is a
    fingerprint of the *flux*: m=2 gives eight such gaps, m=3 gives six, and
    flux 1/8 (n=8) gives two. The old sum(e) reference distinguished none of
    them, being zero for all."""
    monkeypatch.chdir(tmp_path)
    e0 = _flux_spectrum(n=4, m=0)
    assert np.isclose(np.max(np.abs(e0)), 3.0, atol=1e-6)  # pristine bandwidth
    assert np.sum(np.diff(e0) > 0.1) <= 1

    e = _flux_spectrum(n=4, m=1)
    assert np.max(np.abs(e)) < 2.9  # the field narrows the bands
    assert np.sum(np.diff(e) > 0.1) == 4
