import numpy as np
import pytest

from pyqula import islands
from pyqula import interactions
from pyqula import scftypes


@pytest.mark.slow
def test_scf_graphene_island_antiferro_magnetization_matches_reference(tmp_path, monkeypatch):
    """Regression check for an antiferromagnetic Hubbard SCF calculation on
    a small honeycomb island (n=2 instead of 4) with a Zeeman field: the
    summed magnetization must match the value recorded from a known-good
    run. Marked slow: SCF convergence itself (not the island size) drives
    the runtime."""
    monkeypatch.chdir(tmp_path)
    g = islands.get_geometry(name="honeycomb", n=2, nedges=3, rot=0.0)
    h = g.get_hamiltonian(fun=None, has_spin=True)
    h.add_zeeman([0., .4, 0.])
    mf = scftypes.guess(h, mode="antiferro")
    scf = scftypes.selfconsistency(h, filling=0.5, g=1.0, mix=0.9, mf=mf, mode="U")
    m = scf.hamiltonian.get_magnetization()
    assert np.isclose(np.sum(m), 13.200000000000001, atol=1e-4)
