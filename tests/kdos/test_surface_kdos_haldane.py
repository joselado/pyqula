import numpy as np

from pyqula import geometry
from pyqula import kdos


def test_surface_kdos_haldane_matches_reference(tmp_path, monkeypatch):
    """Regression check for kdos.surface (surface vs. bulk spectral function)
    on a Haldane-gapped honeycomb lattice, at a coarse energy/k resolution
    (surface_kdos's nk defaults to len(energies)=100; here energies has only
    20 points): the surface and bulk DOS sums must match the values recorded
    from a known-good run."""
    monkeypatch.chdir(tmp_path)  # writes KDOS.OUT to cwd
    g = geometry.honeycomb_lattice()
    h = g.get_hamiltonian()
    h.add_haldane(0.05)
    (ks, es, ds, db) = kdos.surface(h, energies=np.linspace(-1., 1., 20))
    assert np.isclose(np.sum(ds), 696.7058122199123, atol=1e-4)
    assert np.isclose(np.sum(db), 665.752849518547, atol=1e-4)
