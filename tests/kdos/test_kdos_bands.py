import numpy as np

from pyqula import geometry
from pyqula.kdos import kdos_bands


def _serial_reference(h, kpath, energies, delta, operator=None):
    """Verbatim pre-refactor per-kpoint loop for mode='ED': one
    h.get_dos(ks=[k], ...) call per kpoint."""
    out = []
    for k in kpath:
        (es, ds) = h.get_dos(ks=[k], operator=operator, energies=energies, delta=delta)
        out.append((es, ds))
    return out


def test_kdos_bands_ed_matches_serial_reference(tmp_path, monkeypatch):
    """kdos_bands' default (mode='ED') case now diagonalizes the whole
    kpath at once via h.get_bands instead of dispatching one h.get_dos
    call per kpoint through pcall; check it still agrees."""
    monkeypatch.chdir(tmp_path)
    g = geometry.honeycomb_lattice()
    h = g.get_hamiltonian()
    h.add_zeeman([0., 0., 0.3])
    kpath = h.geometry.get_kpath(None, nk=8)
    energies = np.linspace(-3, 3, 50)
    delta = 0.1

    new = kdos_bands(h, kpath=kpath, energies=energies, delta=delta, mode="ED")
    old = _serial_reference(h, kpath, energies, delta)
    old_y = np.concatenate([o[1] for o in old])
    assert np.allclose(new[2], old_y)


def test_kdos_bands_ed_with_operator_matches_serial_reference(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    g = geometry.honeycomb_lattice()
    h = g.get_hamiltonian()
    h.add_zeeman([0., 0., 0.3])
    kpath = h.geometry.get_kpath(None, nk=6)
    energies = np.linspace(-3, 3, 30)
    delta = 0.1

    new = kdos_bands(h, kpath=kpath, energies=energies, delta=delta,
                      mode="ED", operator="sz")
    old = _serial_reference(h, kpath, energies, delta, operator="sz")
    old_y = np.concatenate([o[1] for o in old])
    assert np.allclose(new[2], old_y)


def test_kdos_bands_green_and_kpm_modes_still_work(tmp_path, monkeypatch):
    """Smoke test: green and KPM modes keep their original pcall path and
    shouldn't be affected by the ED-mode change."""
    monkeypatch.chdir(tmp_path)
    g = geometry.honeycomb_lattice()
    h = g.get_hamiltonian()
    kpath = h.geometry.get_kpath(None, nk=4)
    energies = np.linspace(-3, 3, 20)

    out_green = kdos_bands(h, kpath=kpath, energies=energies, delta=0.1, mode="green")
    assert np.all(np.isfinite(out_green[2]))

    out_kpm = kdos_bands(h, kpath=kpath, energies=energies, delta=0.1,
                          mode="KPM", scale=5.0, ntries=2)
    assert np.all(np.isfinite(out_kpm[2]))
