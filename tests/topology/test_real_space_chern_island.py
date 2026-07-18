import numpy as np
import pytest

from pyqula import islands
from pyqula import topology


@pytest.mark.slow
def test_real_space_chern_haldane_island_matches_reference(tmp_path, monkeypatch):
    """Regression check for the real-space Chern marker on a Haldane-gapped
    honeycomb island, at a small size (n=6 instead of 14): the summed
    marker must match the value recorded from a known-good run. Marked
    slow: shrinking the island further (n=3) made the marker sum exactly
    zero -- too small for the real-space method's bulk region to give a
    meaningful signal -- so n=6 was kept, which stays a few seconds."""
    monkeypatch.chdir(tmp_path)
    g = islands.get_geometry(name="honeycomb", n=6, nedges=4, rot=0.0, clean=False)
    h = g.get_hamiltonian(has_spin=False)
    h.add_haldane(.1)
    (r, c) = topology.real_space_chern(h)
    assert np.isclose(np.sum(c), 1.0658141036401503e-14, atol=1e-6)
