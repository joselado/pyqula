import numpy as np

from pyqula import geometry
from pyqula import specialhamiltonian


def test_flux2d_bands_match_reference(tmp_path, monkeypatch):
    """Regression check for specialhamiltonian.flux2d (commensurate flux
    per unit cell via Peierls substitution) at a small supercell multiplier
    (n=4): the band energy sum must match the value recorded from a
    known-good run."""
    monkeypatch.chdir(tmp_path)  # get_bands writes BANDS.OUT to cwd
    g = geometry.honeycomb_lattice()
    h = specialhamiltonian.flux2d(g, n=4)
    (k, e) = h.get_bands()
    assert np.isclose(np.sum(e), -3.907985046680551e-14, atol=1e-6)
