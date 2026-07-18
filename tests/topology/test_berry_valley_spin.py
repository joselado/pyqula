import numpy as np

from pyqula import geometry
from pyqula import topology
from testutils import temporary_attr


def test_berry_valley_spin_matches_reference(tmp_path, monkeypatch):
    """Regression check for topology.write_berry on a spinful
    antiferromagnetic honeycomb lattice, plain and projected onto
    valley*sz, at a coarse k-path (nk=20 instead of the 600-point default):
    the Berry curvature sums must match the values recorded from a
    known-good run."""
    monkeypatch.chdir(tmp_path)  # writes BERRY_CURVATURE.OUT to cwd
    g = geometry.honeycomb_lattice()
    h = g.get_hamiltonian(has_spin=True)
    h.add_antiferromagnetism(0.6)

    op = h.get_operator("valley") * h.get_operator("sz")
    with temporary_attr(topology.parallel, "cores", 1):
        (x1, y1) = topology.write_berry(h, nk=20)
        (x, y) = topology.write_berry(h, operator=op, nk=20)
    assert np.isclose(np.sum(y1), -2.4100271627976746e-13, atol=1e-6)
    assert np.isclose(np.sum(y), 319.81988453041356, atol=1e-4)
