import numpy as np

from pyqula import geometry


def test_fractal_sierpinski_multildos_is_the_dos(tmp_path, monkeypatch):
    """get_multildos on a Sierpinski triangle fractal, at a shallower
    recursion (n=3 instead of 7) and a coarser energy mesh (30 points
    instead of 100).

    This used to pin sum(DOS.OUT) == 4188.630243168944, which was the
    un-normalized value: multi_ldos_tb handed the eigenvalues to
    calculate_dos raw, without the 1/pi of the Lorentzian that
    dos.dos_kmesh applies, so the recorded constant locked in the error.
    The invariant replacing it is that MULTILDOS/DOS.OUT is a density of
    states: the one h.get_dos computes from the same eigenvalues on the
    refined grid multi_ldos_tb builds internally. Its sum is 1333.28,
    which is the old constant divided by pi."""
    monkeypatch.chdir(tmp_path)
    g = geometry.sierpinski(n=3, mode="triangular")
    h = g.get_hamiltonian(has_spin=False)
    energies = np.linspace(-3.0, 3.0, 30)
    delta = 1e-2
    es2 = np.linspace(min(energies), max(energies), len(energies)*10)
    ref = h.get_dos(energies=es2, delta=delta, write=False)[1]
    h.get_multildos(energies=energies, delta=delta)
    dos = np.genfromtxt("MULTILDOS/DOS.OUT").T
    assert np.max(np.abs(dos[0]-es2)) < 1e-12  # same energy grid
    assert np.max(np.abs(dos[1]-ref)) < 1e-10*np.max(np.abs(ref))
