import numpy as np

from pyqula import geometry


def test_unfolding_defect_reduces_to_primal_cell():
    """If every replica but one is removed from a supercell, the
    remaining Hamiltonian is exactly the primal cell, so its
    unfolding weight must be exactly 1 at every k-point."""
    g0 = geometry.chain()
    n = 3
    g = g0.get_supercell(n, store_primal=True)
    g_reduced = g.remove([1, 2])  # keep only the first replica
    h = g_reduced.get_hamiltonian(has_spin=False)
    kpath = [[k * n, 0., 0.] for k in np.linspace(0., 1., 15)]
    (ks, es, ds) = h.get_bands(operator="unfold", kpath=kpath)
    assert np.allclose(ds, 1.0, atol=1e-10)


def test_unfolding_defective_chain_weight_scales_with_remaining_orbitals():
    """Removing atoms from a supercell must reduce the total
    unfolding weight (summed over bands/kpoints) in proportion to the
    fraction of orbitals that remain, matching the value obtained
    from the unmodified, complete-supercell code path."""
    g0 = geometry.chain()
    n = 6
    g = g0.get_supercell(n, store_primal=True)
    h = g.get_hamiltonian(has_spin=False)
    kpath = g.get_kpath() * n
    (ks, es, ds) = h.get_bands(operator="unfold", kpath=kpath)
    total_full = np.sum(ds)

    g2 = g0.get_supercell(n, store_primal=True)
    g2 = g2.remove([2])
    h2 = g2.get_hamiltonian(has_spin=False)
    (ks2, es2, ds2) = h2.get_bands(operator="unfold", kpath=kpath)
    total_defect = np.sum(ds2)

    ratio = len(g2.r) / len(g.r)
    assert np.isclose(total_defect, total_full * ratio, atol=1e-8)


def test_unfolding_defective_honeycomb_supercell_scales():
    """Same invariant as above, but for a 2-atom primal cell, to also
    exercise the multi-atom-per-primal-cell branch of the mapping."""
    g0 = geometry.honeycomb_lattice()
    g = g0.get_supercell([3, 3], store_primal=True)
    h = g.get_hamiltonian(has_spin=False)
    kpath = g.get_kpath() * 3
    (ks, es, ds) = h.get_bands(operator="unfold", kpath=kpath)
    total_full = np.sum(ds)

    g2 = g0.get_supercell([3, 3], store_primal=True)
    g2 = g2.remove([2, 5])
    h2 = g2.get_hamiltonian(has_spin=False)
    (ks2, es2, ds2) = h2.get_bands(operator="unfold", kpath=kpath)
    total_defect = np.sum(ds2)

    ratio = len(g2.r) / len(g.r)
    assert np.isclose(total_defect, total_full * ratio, atol=1e-6)
