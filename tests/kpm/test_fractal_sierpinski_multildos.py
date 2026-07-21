import numpy as np

from pyqula import geometry


def test_fractal_sierpinski_multildos_matches_reference(tmp_path, monkeypatch):
    """Regression check for get_multildos on a Sierpinski triangle
    fractal, at a shallower recursion (n=3 instead of 7) and coarser
    energy mesh (30 points instead of 100): the total DOS written to
    MULTILDOS/DOS.OUT must match the value recorded from a known-good
    run."""
    monkeypatch.chdir(tmp_path)
    g = geometry.sierpinski(n=3, mode="triangular")
    h = g.get_hamiltonian(has_spin=False)
    h.get_multildos(energies=np.linspace(-3.0, 3.0, 30), delta=1e-2)
    dos = np.genfromtxt("MULTILDOS/DOS.OUT")
    assert np.isclose(np.sum(dos), 4188.630243168944, atol=1e-2)
