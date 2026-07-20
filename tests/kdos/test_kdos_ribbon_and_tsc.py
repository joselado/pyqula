import numpy as np

from pyqula import geometry


def test_kdos_edge_tsc_electron_operator_matches_reference(tmp_path, monkeypatch):
    """Regression check for get_kdos with the 'electron' Nambu-block
    operator on a Rashba+Zeeman+s-wave triangular lattice, at a coarse
    resolution (ng=15 instead of 100): the surface/bulk KDOS sums must
    match the values recorded from a known-good run."""
    monkeypatch.chdir(tmp_path)
    g = geometry.triangular_lattice()
    h = g.get_hamiltonian()
    h.add_onsite(2.)
    h.add_rashba(0.5)
    h.add_exchange([0., 0., 0.5])
    h.add_swave(0.1)
    ng = 15

    es0 = np.linspace(-0.2, 0.2, ng)
    klist = [(k, 0, 0) for k in np.linspace(-0.5, 0.5, ng)]
    op = h.get_operator("electron")
    (ks, es, db, ds) = h.get_kdos(energies=es0, kpath=klist, operator=op, delta=1. / ng)
    assert np.isclose(np.sum(db), 214.38625582215218, atol=1e-4)
    assert np.isclose(np.sum(ds), 205.79278261743835, atol=1e-4)
