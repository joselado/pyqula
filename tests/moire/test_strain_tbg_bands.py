import numpy as np

from pyqula import specialgeometry
from pyqula import specialhamiltonian


def test_strained_tbg_bands_match_reference(tmp_path, monkeypatch):
    """Regression check for strained twisted bilayer graphene at the
    smallest commensurate moire index (n=1): the band energy sum must match
    the value recorded from a known-good run."""
    monkeypatch.chdir(tmp_path)  # get_bands writes BANDS.OUT to cwd
    g = specialgeometry.tbg(1)
    g.add_strain(0.03)
    h = specialhamiltonian.twisted_bilayer_graphene(g=g, ti=0.4)
    (k, e) = h.get_bands(num_bands=20)
    assert np.isclose(np.sum(e), -298.24522113253124, atol=1e-6)
