import numpy as np

from pyqula import geometry


def test_use_kpm_returns_the_kpm_dos_on_the_energies_asked_for(
        tmp_path, monkeypatch):
    """h.get_dos(use_kpm=True) is h.get_dos(mode="KPM"). It used to call the
    KPM with a window built from min(energies) twice and without the
    energies, so it returned the DOS on a grid of its own, spanning
    [-|min E|,|min E|] whatever the energies asked for"""
    monkeypatch.chdir(tmp_path)
    g = geometry.chain().get_supercell(50)
    g.dimensionality = 0
    h = g.get_hamiltonian(is_sparse=True, has_spin=False)
    energies = np.linspace(-1., 3., 41) # not symmetric around zero
    np.random.seed(3)
    (x, y) = h.get_dos(use_kpm=True, energies=energies, delta=0.1, ntries=4,
                       write=False)
    np.random.seed(3)
    (x2, y2) = h.get_dos(mode="KPM", energies=energies, delta=0.1, ntries=4,
                         write=False)
    assert np.allclose(x, energies)
    assert np.allclose(y, y2)
