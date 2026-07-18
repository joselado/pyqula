import numpy as np

from pyqula import geometry
from pyqula.specialhamiltonian import NbSe2


def test_fermi_surface_valley_operator_matches_reference(tmp_path, monkeypatch):
    """Regression check for get_fermi_surface on a gapped honeycomb
    lattice, valley-resolved, at a coarse mesh (nk=15 instead of 100): the
    total weight must match the value recorded from a known-good run."""
    monkeypatch.chdir(tmp_path)  # writes FERMI_MAP.OUT to cwd
    g = geometry.honeycomb_lattice()
    h = g.get_hamiltonian(has_spin=True)
    h.add_onsite(0.6)
    h.get_bands()
    (kx, ky, fs) = h.get_fermi_surface(nk=15, operator="valley", num_waves=10, mode="lowest")
    assert np.isclose(np.sum(fs), -8.881784197001252e-15, atol=1e-6)


def test_operator_fermi_surface_nbse2_sz_matches_reference(tmp_path, monkeypatch):
    """Regression check for get_fermi_surface on the NbSe2 SOC model,
    sz-resolved, at a coarse mesh (nk=15 instead of 100): the total weight
    must match the value recorded from a known-good run."""
    monkeypatch.chdir(tmp_path)
    h = NbSe2(soc=0.9)
    (kx, ky, fs) = h.get_fermi_surface(e=0., nk=15, delta=3e-1, operator="sz")
    assert np.isclose(np.sum(fs), -5.950795411990839e-14, atol=1e-6)
