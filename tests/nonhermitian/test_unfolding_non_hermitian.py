import numpy as np

from pyqula import geometry


def test_unfolding_non_hermitian_chain_matches_reference(tmp_path, monkeypatch):
    """Regression check for band unfolding on a non-Hermitian chain
    supercell with a complex onsite modulation, at a small supercell (n=10
    instead of 40) and coarse kdos energy mesh (30 points instead of 200):
    the unfolded band energies/weights and the KDOS-bands array must match
    the values recorded from a known-good run."""
    monkeypatch.chdir(tmp_path)
    g0 = geometry.chain()
    n = 10
    g = g0.get_supercell(n, store_primal=True)
    h = g.get_hamiltonian(has_spin=False, non_hermitian=True)
    omega = 1. / n
    ons = lambda r: 0.1j * np.cos(np.pi * 2 * omega * r[0])
    h.add_onsite(ons)

    kpath = g.get_kpath() * n
    (ks, es, ds) = h.get_bands(operator="unfold", kpath=kpath)
    out = h.get_kdos_bands(operator="unfold", kpath=kpath,
                            energies=np.linspace(0., 1., 30), eigmode="real")

    assert np.isclose(np.sum(es), -8.08242361927114e-14 + 3.422222857109533e-14j, atol=1e-6)
    assert np.isclose(np.sum(ds), 4000.0324260411435 + 0j, atol=1e-3)
    assert np.isclose(np.sum(out), 31589.64648870226, atol=1e-2)
