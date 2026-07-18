import numpy as np

from pyqula import geometry
from pyqula import ribbon
from pyqula import multicell


def test_kagome_frustration_ribbon_bands_match_reference(tmp_path, monkeypatch):
    """Regression check for a non-collinear-magnetism kagome ribbon (three
    sublattice moments plus an s-wave term on one edge), at a small width
    (n=6 instead of 20): the interface-resolved band energies must match
    the values recorded from a known-good run."""
    monkeypatch.chdir(tmp_path)
    g = geometry.kagome_lattice()
    g.has_sublattice = True
    g.sublattice = [-1, 1, 0]
    g = ribbon.bulk2ribbon(g, n=6)
    h = g.get_hamiltonian()
    m1 = np.array([1., 0., 0.])
    m2 = np.array([-.5, np.sqrt(3.) / 2., 0.])
    m3 = np.array([-.5, -np.sqrt(3.) / 2., 0.])
    mm = 3.0
    ms = []
    for (r, s) in zip(g.r, g.sublattice):
        if r[1] < 0.0:
            if s == -1: ms.append(m1 * mm)
            if s == 1: ms.append(m2 * mm)
            if s == 0: ms.append(m3 * mm)
        else:
            ms.append([0., 0., 0.])

    def fs(r):
        if r[1] > 0.0: return 0.3
        else: return 0.0

    h.add_magnetism(ms)
    h.add_swave(fs)
    h.shift_fermi(fs)
    (k, e, c) = h.get_bands(operator="interface")
    assert np.isclose(np.sum(e), 9.393374966748524e-12, atol=1e-6)
    assert np.isclose(np.sum(c), 9600.0, atol=1e-4)


def test_edge_states_chern_ribbon_matches_reference(tmp_path, monkeypatch):
    """Regression check for a topological-superconductor triangular-lattice
    ribbon (Rashba + Zeeman + s-wave pairing), at a small width (n=15
    instead of 60) and coarse LDOS mesh (nk=20, nrep=5 instead of 100/30):
    the y-position-resolved bands and zero-energy LDOS must match the
    values recorded from a known-good run."""
    monkeypatch.chdir(tmp_path)
    g = geometry.triangular_lattice()
    h = g.get_hamiltonian()
    h.add_rashba(1.0)
    h.add_zeeman([0., 0., 1.0])
    h.add_onsite(-6.0)
    h.add_swave(0.4)
    hr = multicell.bulk2ribbon(h, n=15)
    (kb, eb, cb) = hr.get_bands(operator="yposition")
    (x, y, ld) = hr.get_ldos(e=0.0, delta=1e-3, nk=20, nrep=5)
    assert np.isclose(np.sum(eb), 3.920774815924233e-11, atol=1e-6)
    assert np.isclose(np.sum(cb), -4.976019596369952e-13, atol=1e-6)
    assert np.isclose(np.sum(ld), 1.3926654082948746, atol=1e-4)
