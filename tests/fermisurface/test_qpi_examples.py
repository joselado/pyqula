import numpy as np

from pyqula import geometry


def test_multifermisurface_qpi_square_lattice_matches_reference(tmp_path, monkeypatch):
    """Regression check for get_qpi (response mode) on a square lattice, at
    a coarse mesh (nk=10, 11 energies instead of 80/101): the total DOS
    written to DOS.OUT must match the value recorded from a known-good
    run."""
    monkeypatch.chdir(tmp_path)
    g = geometry.square_lattice()
    h = g.get_hamiltonian()
    h.get_qpi(delta=0.1, nk=10, energies=np.linspace(-4, 4, 11))
    dos = np.genfromtxt("DOS.OUT")
    assert np.isclose(np.sum(dos), 3544.8171918487706, atol=1e-2)


def test_multiqpi_poor_man_triangular_lattice_matches_reference(tmp_path, monkeypatch):
    """Regression check for get_qpi (poor-man convolution mode, mode="pm")
    on a triangular lattice, at a coarse mesh (nk=10, 10 energies instead
    of 50/100): the total DOS written to DOS.OUT must match the value
    recorded from a known-good run."""
    monkeypatch.chdir(tmp_path)
    g = geometry.triangular_lattice()
    h = g.get_hamiltonian(has_spin=False)
    h.get_qpi(delta=1e-1, mode="pm", info=True, nk=10,
              energies=np.linspace(-6., 6., 10))
    dos = np.genfromtxt("DOS.OUT")
    assert np.isclose(np.sum(dos), 166.2915260634556, atol=1e-2)
