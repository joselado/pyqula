import numpy as np
from scipy.linalg import eigh

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


def test_unfolding_non_diagonal_supercell_reduces_to_primal_cell():
    """Non-diagonal-M analog of test_unfolding_defect_reduces_to_primal_cell:
    build a supercell with a general (non-orthogonal, off-diagonal) integer
    matrix M, keep only the replica at n=(0,0,0), and check the unfolding
    weight is exactly 1 everywhere, exactly like a diagonal supercell."""
    g0 = geometry.chain()
    M = [[3, 0, 0], [1, 1, 0], [0, 0, 1]]  # non-diagonal, det=3
    g = g0.get_supercell(M, store_primal=True)
    keep0 = np.where(np.all(g.supercell_replica == [0, 0, 0], axis=1))[0]
    remove = [i for i in range(len(g.r)) if i not in keep0]
    g_reduced = g.remove(remove)
    h = g_reduced.get_hamiltonian(has_spin=False)
    kpath = [[k, 0.3 * k, 0.] for k in np.linspace(0., 1., 15)]
    (ks, es, ds) = h.get_bands(operator="unfold", kpath=kpath)
    assert np.allclose(ds, 1.0, atol=1e-10)


def test_unfolding_non_diagonal_supercell_weight_scales_with_remaining_orbitals():
    """Non-diagonal-M analog of
    test_unfolding_defective_chain_weight_scales_with_remaining_orbitals:
    removing atoms from a non-diagonal supercell must reduce the total
    unfolding weight in proportion to the fraction of remaining orbitals,
    exactly as for a diagonal supercell."""
    g0 = geometry.honeycomb_lattice()
    M = [[2, 1, 0], [0, 1, 0], [0, 0, 1]]  # non-diagonal, det=2
    g = g0.get_supercell(M, store_primal=True)
    h = g.get_hamiltonian(has_spin=False)
    kpath = g.get_kpath()
    (ks, es, ds) = h.get_bands(operator="unfold", kpath=kpath)
    total_full = np.sum(ds)

    g2 = g0.get_supercell(M, store_primal=True)
    g2 = g2.remove([2])
    h2 = g2.get_hamiltonian(has_spin=False)
    (ks2, es2, ds2) = h2.get_bands(operator="unfold", kpath=kpath)
    total_defect = np.sum(ds2)

    ratio = len(g2.r) / len(g.r)
    assert np.isclose(total_defect, total_full * ratio, atol=1e-6)


def test_unfolding_non_diagonal_supercell_matches_direct_diagonalization():
    """Direct correctness check for the k-remapping used by the
    non-diagonal-M unfolding path (bloch_phase_matrix_matrix's
    k_primal = Minv@k_super). The two tests above are blind to a
    wrong Minv (e.g. using M instead of its inverse): one only keeps
    the n=(0,0,0) replica, where every phase factor is exp(i*0)=1
    regardless of Minv/M, and the other only checks a weight-sum
    invariant that doesn't depend on how k is reparametrized.

    Here we independently brute-force enumerate, for a fixed
    supercell k-point, every primal k-point k0=Minv@(k_super+j) (j
    integer) that folds onto it -- there must be exactly N=|det(M)|
    distinct ones (mod the primal reciprocal lattice) -- and check
    that a) the union of h0's eigenvalues at those points exactly
    reproduces the supercell spectrum at k_super (pure Bloch-folding
    identity, independent of the unfolding operator itself), and b)
    the "unfold" operator assigns full weight (n0, the primal orbital
    count) to exactly the supercell eigenstates whose energy matches
    h0's spectrum at the j=0 representative k0=Minv@k_super, and ~0
    weight to the rest (which come from the other N-1 folded
    primal k-points)."""
    g0 = geometry.honeycomb_lattice()
    M = np.array([[2, 1, 0], [0, 1, 0], [0, 0, 1]])  # non-diagonal, det=2
    Minv = np.linalg.inv(M.astype(float))
    g = g0.get_supercell(M, store_primal=True)
    h = g.get_hamiltonian(has_spin=False)
    h0 = g0.get_hamiltonian(has_spin=False)
    n0 = h0.intra.shape[0]

    from pyqula.unfolding import bloch_projector
    op = bloch_projector(h)

    k_super = np.array([0.37, 0.81, 0.])
    Hk = h.get_hk_gen()(k_super)
    Es, Vs = eigh(Hk)

    # brute-force enumerate the N distinct primal k-points folding onto k_super
    N = int(round(abs(np.linalg.det(M))))
    cands = []
    for j0 in range(-4, 5):
        for j1 in range(-4, 5):
            j = np.array([j0, j1, 0.])
            cands.append((Minv @ (k_super + j)) % 1.0)
    uniq = []
    for c in cands:
        if not any(np.allclose(c, u, atol=1e-8) for u in uniq):
            uniq.append(c)
    assert len(uniq) == N

    hk0 = h0.get_hk_gen()
    primal_energies = np.sort(np.concatenate([eigh(hk0(k0))[0] for k0 in uniq]))
    assert np.allclose(np.sort(Es), primal_energies, atol=1e-8)

    k0_ref = Minv @ k_super  # the j=0 representative
    ref_energies = eigh(hk0(k0_ref))[0]
    for i in range(Vs.shape[1]):
        v = Vs[:, i]
        w = np.abs(op.m(v, k_super).dot(np.conjugate(v)))
        if np.any(np.isclose(Es[i], ref_energies, atol=1e-8)):
            assert np.isclose(w, n0, atol=1e-6)
        else:
            assert np.isclose(w, 0.0, atol=1e-6)
