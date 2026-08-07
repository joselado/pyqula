import numpy as np

from pyqula import geometry


def test_dos_green_mode_matches_ed(tmp_path, monkeypatch):
    """Regression check for dos.py's mode="Green"/"RG" DOS normalization:
    it used to return the raw -Im[Tr G] from green.green_operator without
    the standard 1/pi DOS prefactor that mode="ED" (dos_kmesh) applies,
    making Green-mode DOS values ~pi times too large. Green and ED are
    different algorithms so exact agreement isn't expected, but the
    factor-of-pi discrepancy (relative L2 norm ~2.1-2.4 before the fix,
    collapsing to ~0.1-0.2 after) is what this test guards against."""
    monkeypatch.chdir(tmp_path)
    h = geometry.honeycomb_lattice().get_hamiltonian()
    energies = np.linspace(-3.0, 3.0, 60)

    (_, ys_ed) = h.get_dos(mode="ED", energies=energies, nk=20)
    (_, ys_green) = h.get_dos(mode="Green", energies=energies, nk=20, delta=0.1)
    ys_ed = np.array(ys_ed)
    ys_green = np.array(ys_green)

    reldiff = np.linalg.norm(ys_green - ys_ed) / np.linalg.norm(ys_ed)
    assert reldiff < 0.5
