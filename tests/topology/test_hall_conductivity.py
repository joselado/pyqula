"""topology.hall_conductivity on a quantum-anomalous-Hall honeycomb model.

topology.hall_conductivity is an alias of topology.chern, so the quantity
it returns is sigma_xy in units of e^2/h, which inside a gap is exactly the
Chern number. That is the thing worth asserting: it is quantized, it is
mesh-independent, and it agrees with h.get_chern -- rather than a recorded
sum over a chemical-potential sweep, whose metallic points are
non-universal, sensitive to nk, and unreadable as physics (the previous
version of this file pinned np.sum(sigmas) = 1.632 over five mu values, of
which only the one in the gap carried any invariant content).

The Fukui-Hatsugai-Suzuki construction underneath counts vortices in link
variables, so in a gap it is quantized to machine precision, not merely
close to an integer -- hence the 1e-9 tolerances below. It does need a mesh
fine enough to resolve the curvature: nk=6 returns 0 for this model, which
is why the meshes checked start at 8.
"""
import numpy as np

from pyqula import geometry
from pyqula import topology

GAP_CHERN = 2.0 # one unit from each of the two valleys


def _qah_hamiltonian(zeeman=0.2):
    """Zeeman + Rashba honeycomb lattice: a Chern insulator at half
    filling, with the Fermi level inside the gap."""
    g = geometry.honeycomb_lattice()
    h = g.get_hamiltonian(has_spin=True)
    h.add_zeeman([0., 0., zeeman])
    h.add_rashba(0.2)
    return h


def test_hall_conductivity_in_the_gap_is_the_chern_number(tmp_path, monkeypatch):
    """Two independent code paths for the same invariant: the alias, and
    h.get_chern on the same Hamiltonian."""
    monkeypatch.chdir(tmp_path)
    h = _qah_hamiltonian()
    sigma = topology.hall_conductivity(h, nk=8)
    assert np.isclose(sigma, GAP_CHERN, atol=1e-9), sigma
    assert np.isclose(sigma, h.get_chern(nk=14), atol=1e-9)


def test_hall_conductivity_in_the_gap_is_mesh_independent(tmp_path, monkeypatch):
    """Quantization, stated as the property that distinguishes a
    topological invariant from a quadrature: refining the mesh must not
    move the answer at all. This is also what separates a real gap from an
    accidental one -- the same model with the Zeeman field switched off is
    gapless (Rashba alone does not gap the Dirac points) and then returns 2
    at nk=8 but 1 at nk=20."""
    monkeypatch.chdir(tmp_path)
    h = _qah_hamiltonian()
    for nk in (8, 10, 12, 20):
        assert np.isclose(topology.hall_conductivity(h, nk=nk), GAP_CHERN,
                          atol=1e-9), nk


def test_reversing_the_magnetization_reverses_the_hall_conductivity(
        tmp_path, monkeypatch):
    """sigma_xy is odd under time reversal, and the Zeeman field is the
    only term breaking it here. A magnitude-only check would not see a
    sign error; this does."""
    monkeypatch.chdir(tmp_path)
    up = topology.hall_conductivity(_qah_hamiltonian(zeeman=+0.2), nk=8)
    down = topology.hall_conductivity(_qah_hamiltonian(zeeman=-0.2), nk=8)
    assert np.isclose(down, -up, atol=1e-9), (up, down)
