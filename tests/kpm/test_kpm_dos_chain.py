import numpy as np

from pyqula import geometry


def test_kpm_dos_chain_matches_reference(tmp_path, monkeypatch):
    """Regression check for the stochastic-trace KPM DOS on a large 1D
    chain, at a smaller size (supercell(300) instead of 3000), coarse
    energy mesh (50 points instead of 200), and a looser broadening
    (delta=1e-2 instead of 1e-4 -- the KPM polynomial order scales as
    1/delta and dominates the runtime): the total DOS must match the value
    recorded from a known-good run, within a generous tolerance since
    ntries=10 stochastic trace vectors give ~1-2% run-to-run variance."""
    monkeypatch.chdir(tmp_path)
    g = geometry.chain()
    g = g.get_supercell(300)
    g.dimensionality = 0
    h = g.get_hamiltonian(is_sparse=True, has_spin=False)
    (x, y) = h.get_dos(mode="KPM",
                energies=np.linspace(-3.0, 3.0, 50),
                delta=1e-2,
                ntries=10)
    assert np.isclose(np.sum(y), 21.6, rtol=0.1)
