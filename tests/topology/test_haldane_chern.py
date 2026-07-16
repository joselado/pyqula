from pyqula import geometry
from pyqula import topology


def test_haldane_chern_number_is_quantized(tmp_path, monkeypatch):
    """The Chern number of the Haldane model must be an integer: zero with
    no second-neighbor flux (time-reversal symmetric), and nonzero once the
    Haldane flux opens a topological gap."""
    monkeypatch.chdir(tmp_path)  # topology.chern writes *.OUT files to cwd
    g = geometry.honeycomb_lattice()
    g = g.get_supercell(2)

    h_trivial = g.get_hamiltonian()
    c_trivial = topology.chern(h_trivial, nk=8)
    assert abs(round(c_trivial) - c_trivial) < 1e-6
    assert round(c_trivial) == 0

    h_topological = g.get_hamiltonian()
    h_topological.add_haldane(0.2)
    c_topological = topology.chern(h_topological, nk=8)
    assert abs(round(c_topological) - c_topological) < 1e-6
    assert round(c_topological) != 0
