import numpy as np

from pyqula import geometry


def test_dwave_pairing_multi_fermi_surface_matches_reference(tmp_path, monkeypatch):
    """Regression check for get_multi_fermi_surface with a dx2y2 pairing on
    a square lattice, at a coarse mesh (nk=20, 20 energies instead of
    200/100): the total weight must match the value recorded from a
    known-good run."""
    monkeypatch.chdir(tmp_path)
    g = geometry.square_lattice()
    h = g.get_hamiltonian()
    h.add_onsite(-2.)
    h.add_pairing(mode="dx2y2", delta=0.2)
    out = h.get_multi_fermi_surface(energies=np.linspace(-2.0, 2.0, 20), delta=4e-2,
                                     nk=20, nsuper=1)
    assert np.isclose(np.sum(out), 11644.867730249154, atol=1e-2)


def test_fwave_pairing_multi_fermi_surface_matches_reference(tmp_path, monkeypatch):
    """Regression check for get_multi_fermi_surface with a nodal f-wave
    pairing on a triangular lattice, at a coarse mesh (nk=15, 30 energies
    instead of 100/200): the total weight must match the value recorded
    from a known-good run."""
    monkeypatch.chdir(tmp_path)
    g = geometry.triangular_lattice()
    h = g.get_hamiltonian()
    h.add_pairing(mode="nodal_fwave", delta=0.2)
    out = h.get_multi_fermi_surface(energies=np.linspace(-2.0, 2.0, 30), delta=4e-2,
                                     nk=15, nsuper=1)
    assert np.isclose(np.sum(out), 10485.513437055704, atol=1e-2)
