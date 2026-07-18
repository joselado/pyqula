import numpy as np

from pyqula import geometry
from pyqula import topology
from testutils import temporary_attr


def test_berry_valley_matches_reference(tmp_path, monkeypatch):
    """Regression check for topology.write_berry on a spinless honeycomb
    lattice with a sublattice imbalance, plain and projected onto the
    valley operator, at a coarse k-path (nk=20 instead of the 600-point
    default): the Berry curvature sums must match the values recorded from
    a known-good run."""
    monkeypatch.chdir(tmp_path)  # writes BERRY_CURVATURE.OUT to cwd
    g = geometry.honeycomb_lattice()
    h = g.get_hamiltonian(has_spin=False)
    h.add_sublattice_imbalance(0.6)

    op = h.get_operator("valley", projector=True)
    with temporary_attr(topology.parallel, "cores", 1):
        (x1, y1) = topology.write_berry(h, nk=20)
        (x, y) = topology.write_berry(h, operator=op, nk=20)
    assert np.isclose(np.sum(y1), 5.551115123125783e-15, atol=1e-6)
    assert np.isclose(np.sum(y), 159.90994226520772, atol=1e-4)
