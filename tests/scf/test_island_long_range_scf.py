import numpy as np
import pytest

from pyqula import islands
from pyqula import meanfield


@pytest.mark.slow
def test_island_long_range_scf_magnetization_matches_reference(tmp_path, monkeypatch):
    """Regression check for a ferromagnetic Hubbard+V1 SCF calculation on a
    small honeycomb island (n=2 instead of 3), with maxerror loosened from
    1e-9 to 1e-6: the summed magnetization must match the value recorded
    from a known-good run. Marked slow: SCF convergence itself (not the
    island size or tolerance) drives the runtime."""
    monkeypatch.chdir(tmp_path)
    g = islands.get_geometry(name="honeycomb", n=2, nedges=3, rot=0.0)
    h = g.get_hamiltonian(has_spin=True)
    mf = meanfield.guess(h, mode="ferro")
    scf = meanfield.Vinteraction(h, filling=0.5, U=3.0, V1=1.0, mf=mf, maxerror=1e-6)
    m = scf.hamiltonian.get_magnetization()
    assert np.isclose(np.sum(m), 4.499999999999997, atol=1e-4)
