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


def test_fermi_surface_qtci_backend_matches_grid(tmp_path, monkeypatch):
    """backend="qtci" reconstructs the Fermi surface mesh from a quantics
    tensor cross interpolation (qutecipy) instead of brute-force evaluating
    every mesh point; it must reproduce the grid-based result on a smooth,
    broadened (delta=0.3) spectral weight, which compresses well."""
    monkeypatch.chdir(tmp_path)  # writes FERMI_MAP.OUT to cwd
    g = geometry.honeycomb_lattice()
    h = g.get_hamiltonian()
    h.add_onsite(0.6)

    nk = 32  # a power of two so the qtci mesh lines up exactly with the grid mesh
    kx1, ky1, fs1 = h.get_fermi_surface(nk=nk, delta=0.3, write=False)
    kx2, ky2, fs2 = h.get_fermi_surface(nk=nk, delta=0.3, write=False,
            backend="qtci", tolerance=1e-3)
    assert np.allclose(kx1, kx2) and np.allclose(ky1, ky2)
    assert np.max(np.abs(fs1 - fs2)) < 1e-2


def test_fermi_surface_qtci_backend_non_power_of_two_nk(tmp_path, monkeypatch):
    """When nk isn't a power of two, qtci internally rounds up to the
    nearest 2**R for the quantics mesh, then must interpolate the result
    back onto the exact nk x nk grid the caller asked for (same shape and
    k-points as the grid backend). Interpolating a coarse, non-aligned
    mesh trades some pointwise accuracy for a peaked function, so this
    checks the total spectral weight (integrated over the BZ) rather than
    a strict per-point tolerance."""
    monkeypatch.chdir(tmp_path)
    g = geometry.honeycomb_lattice()
    h = g.get_hamiltonian()
    h.add_onsite(0.6)

    nk = 20  # not a power of two
    kx1, ky1, fs1 = h.get_fermi_surface(nk=nk, delta=0.3, write=False)
    kx2, ky2, fs2 = h.get_fermi_surface(nk=nk, delta=0.3, write=False,
            backend="qtci", tolerance=1e-3)
    assert fs1.shape == (nk * nk,) and fs2.shape == (nk * nk,)
    assert np.allclose(kx1, kx2) and np.allclose(ky1, ky2)
    assert abs(fs1.sum() - fs2.sum()) / fs1.sum() < 1e-2
