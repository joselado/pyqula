import numpy as np

from pyqula import geometry
from pyqula import kdos


def test_kdos_long_range_square_lattice_matches_reference(tmp_path, monkeypatch):
    """Regression check for get_kdos on a square lattice with long-range
    hoppings and Rashba SOC, at a coarse resolution (15 energies and
    nit=15 instead of 100, and a 8x8 instead of 100x100 k-path scan for
    the bare band structure): the surface/bulk KDOS sums and the summed
    band energies must match the values recorded from a known-good run."""
    monkeypatch.chdir(tmp_path)
    g = geometry.square_lattice()
    h = g.get_hamiltonian(tij=[0.5, 0., 0., 0., 0.6, 0.3])
    h.add_rashba(0.5)
    energies = np.linspace(-6.0, 6.0, 15)
    (k, e, ds, db) = h.get_kdos(energies=energies, delta=1e-2, nit=15)

    ebsum = 0.0
    for ky in np.linspace(0., 1.0, 8):
        kpath = [[kx, ky, 0.] for kx in np.linspace(0., 1.0, 8)]
        (kb, eb) = h.get_bands(kpath=kpath)
        ebsum += np.sum(eb)

    assert np.isclose(np.sum(ds), 110.00793179113124, atol=1e-4)
    assert np.isclose(np.sum(db), 109.57374632696465, atol=1e-4)
    assert np.isclose(ebsum, 56.00000000000002, atol=1e-4)


def test_kdos_interface_haldane_domain_wall_matches_reference(tmp_path, monkeypatch):
    """Regression check for kdos.interface between two honeycomb
    Hamiltonians with opposite Haldane flux, at a coarse resolution (15
    energies, nk=10 instead of 100/50): the total interface KDOS must
    match the value recorded from a known-good run."""
    monkeypatch.chdir(tmp_path)  # writes KDOS_INTERFACE.OUT to cwd
    g = geometry.honeycomb_lattice()
    h = g.get_hamiltonian(has_spin=False)
    h1 = h.copy()
    h2 = h.copy()
    h1.add_haldane(0.1)
    h2.add_haldane(-0.1)
    out = kdos.interface(h1, h2, energies=np.linspace(-1.0, 1.0, 15), nk=10, delta=3e-2)
    assert np.isclose(np.sum(out), 1409.4394293855391, atol=1e-4)
