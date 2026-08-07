import numpy as np

from pyqula import geometry


def test_kpm_dos_chain_matches_reference(tmp_path, monkeypatch):
    """Regression check for the stochastic-trace KPM DOS on a large 1D
    chain, at a smaller size (supercell(300) instead of 3000), coarse
    energy mesh (50 points instead of 200), and a looser broadening
    (delta=1e-2 instead of 1e-4 -- the KPM polynomial order scales as
    1/delta and dominates the runtime): the total DOS must match the value
    recorded from a known-good run, within a generous tolerance given
    ntries=10 stochastic trace vectors' run-to-run variance.

    The reference value here is ~100x the value this test used to pin
    (21.6): dos.py's dos_kpm normalized its k-mesh average by `nk` (the
    k-mesh density kwarg, default 100) instead of `numk = len(ks)`. For a
    dimensionality=0 system (this test's case) there is exactly one
    k-point regardless of `nk`, so the old code divided a single-k-point
    result by an unrelated default parameter -- an unphysical ~100x
    undercount, fixed alongside this test."""
    monkeypatch.chdir(tmp_path)
    g = geometry.chain()
    g = g.get_supercell(300)
    g.dimensionality = 0
    h = g.get_hamiltonian(is_sparse=True, has_spin=False)
    (x, y) = h.get_dos(mode="KPM",
                energies=np.linspace(-3.0, 3.0, 50),
                delta=1e-2,
                ntries=10)
    assert np.isclose(np.sum(y), 2220.0, rtol=0.15)
