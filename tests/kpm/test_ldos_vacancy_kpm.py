import numpy as np

from pyqula import geometry
from pyqula import ldos


def test_ldos_vacancy_kpm_matches_reference(tmp_path, monkeypatch):
    """Regression check for KPM local DOS at a vacancy site in a honeycomb
    supercell, at a small size (supercell(8) instead of 30) and coarse
    resolution (60 energies instead of the 500 default, delta=0.3 instead
    of the KPM default 0.01 -- the polynomial order scales as 1/delta, so
    this is the dominant cost, not the supercell size or energy count):
    the total LDOS must match the value recorded from a known-good run.

    The value moved from 12.48573379499043 to 12.489853029925431 (0.03%)
    when delta became the half width at half maximum of the KPM peaks, the
    same as the Lorentzian of the ED modes: dos_site(mode="KPM") used to
    take 5*scale/delta polynomials, 165 here, peaks of half width 0.11
    rather than delta=0.3, and now takes 62 (kpmtk.kernels.jackson_npol).
    The sum over 60 energies of a broadened LDOS barely depends on the
    width."""
    monkeypatch.chdir(tmp_path)
    g = geometry.honeycomb_lattice()
    g = g.supercell(8)
    r0 = [0., 0., 0.]
    g = g.remove(g.closest_index(r0))
    h = g.get_hamiltonian(has_spin=False)
    (es, ds) = ldos.dos_site(h, nk=5, mode="KPM", i=g.closest_index(r0),
                              energies=np.linspace(-1., 1., 60), delta=0.3)
    assert np.isclose(np.sum(ds), 12.489853029925431, atol=1e-4)
