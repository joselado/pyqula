import os
import numpy as np

from pyqula import geometry
from pyqula import sculpt
from pyqula import pseudocontact

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def test_pseudocurrent_image_island_matches_reference(tmp_path, monkeypatch):
    """Regression check for a pseudo-contact correlator on an island built
    from a bitmap image, at a smaller island size (size=10 instead of 20):
    the summed correlator written to CORRELATOR.OUT must match the value
    recorded from a known-good run."""
    monkeypatch.chdir(tmp_path)  # write_correlator writes CORRELATOR.OUT to cwd
    g = geometry.honeycomb_lattice()
    imfile = os.path.join(DATA_DIR, "contact.png")
    g0 = sculpt.image2island(imfile, g, size=10, color="black")
    gc = sculpt.image2island(imfile, g, size=10, color="red")
    g = sculpt.add(g0, gc)
    g = g.clean()
    h = g.get_hamiltonian(has_spin=False, is_sparse=True)
    h.add_peierls(0.02)
    indexes = sculpt.common(g, gc)
    pseudocontact.write_correlator(h, index=indexes, e=0.4)
    d = np.genfromtxt("CORRELATOR.OUT")
    assert np.isclose(np.sum(d), 335.61398621256933, atol=1e-2)
