import numpy as np
from scipy.linalg import eigh

from pyqula import geometry


def dense(m):
    """Return a dense array out of a possibly sparse matrix"""
    return np.array(m.todense()) if hasattr(m, "todense") else np.array(m)


def folded_kpoints(Minv, k, dim):
    """Every primitive-cell k-point that folds onto the supercell
    k-point k, enumerated by brute force from k0 = Minv@(k+j) with j an
    integer vector"""
    rng = range(-5, 6)
    out = []
    for j0 in rng:
        for j1 in rng:
            for j2 in (rng if dim == 3 else [0]):
                c = (Minv @ (k + np.array([j0, j1, j2], dtype=float))) % 1.0
                if not any(np.allclose(c, u, atol=1e-8) for u in out):
                    out.append(c)
    return out


def check_unfolding(g0, M, k=(0.37, 0.81, 0.29)):
    """Shared check for a general (possibly non-orthogonal) supercell
    matrix M: build the supercell, and verify both that the supercell
    spectrum at a k-point is the union of the primitive spectra at the
    N=|det M| k-points folding onto it, and that the unfold operator
    gives weight N to exactly those states coming from the j=0
    representative k0=Minv@k and zero to the rest."""
    from pyqula.unfolding import bloch_projector
    g = g0.get_supercell(M, store_primal=True)
    Ms = np.array(g.supercell_matrix)
    Minv = np.linalg.inv(Ms.astype(float))
    N = int(round(abs(np.linalg.det(Ms))))
    assert N == len(g.r) // len(g0.r)
    h = g.get_hamiltonian(has_spin=False)
    h0 = g0.get_hamiltonian(has_spin=False)
    op = bloch_projector(h)
    k = np.array(k, dtype=float)
    for d in range(g.dimensionality, 3):
        k[d] = 0.
    (es, ws) = eigh(dense(h.get_hk_gen()(k)))
    uniq = folded_kpoints(Minv, k, g.dimensionality)
    assert len(uniq) == N
    primal = np.sort(np.concatenate([eigh(dense(h0.get_hk_gen()(k0)))[0]
                                     for k0 in uniq]))
    assert np.allclose(np.sort(es), primal, atol=1e-8)
    ref = eigh(dense(h0.get_hk_gen()(Minv @ k)))[0]
    for i in range(ws.shape[1]):
        v = ws[:, i]
        w = np.abs(op.m(v, k).dot(np.conjugate(v)))
        # the full weight is N=|det M|, the number of primitive cells in
        # the supercell, and not the number of primitive orbitals: the
        # projector's rows have N entries of modulus one, so a state that
        # unfolds completely onto k0 comes out with |P v|^2 = N
        expected = N if np.any(np.isclose(es[i], ref, atol=1e-8)) else 0.
        assert np.isclose(w, expected, atol=1e-6)
    return g


def test_unfolding_sqrt3_triangular():
    """sqrt(3) x sqrt(3) supercell of a triangular lattice, built with
    the rotated matrix M=[[2,1],[-1,1]]: neither diagonal nor
    orthogonal, so the per-axis k-scaling of the legacy path cannot
    describe it."""
    check_unfolding(geometry.triangular_lattice(),
                    [[2, 1, 0], [-1, 1, 0], [0, 0, 1]])


def test_unfolding_sqrt3_honeycomb():
    """Same sqrt(3) x sqrt(3) cell on a two-atom primitive cell, which
    also exercises the orbital-per-atom bookkeeping of the projector."""
    check_unfolding(geometry.honeycomb_lattice(),
                    [[2, 1, 0], [-1, 1, 0], [0, 0, 1]])


def test_unfolding_sqrt5_square():
    """A sqrt(5) x sqrt(5) cell of the square lattice, rotated by
    atan(1/2), with an odd number of primitive cells."""
    check_unfolding(geometry.square_lattice(),
                    [[2, 1, 0], [-1, 2, 0], [0, 0, 1]])


def test_unfolding_three_dimensional_supercell():
    """A general matrix supercell in 3d. The replica bookkeeping is
    recorded in three dimensions and the k-mapping is the full 3x3
    inverse, so nothing in this path is restricted to 2d."""
    check_unfolding(geometry.cubic_lattice(),
                    [[2, 1, 0], [0, 1, 0], [0, 1, 2]])


def test_float_supercell_is_the_compact_sqrt3_cell():
    """g.get_supercell(np.sqrt(3)) asks for a cell three times larger,
    with no constraint on its shape. Ranking the candidates by volume
    alone used to leave the shape to a lexicographic accident and
    returned a nearly degenerate cell (3.7 degrees between lattice
    vectors 7.8 and 5.2 times the primitive one), whose first neighbors
    fell outside the cells the hopping generator searches, so the
    resulting Hamiltonian had no hoppings at all."""
    for g0 in [geometry.triangular_lattice(), geometry.honeycomb_lattice()]:
        g = g0.get_supercell(np.sqrt(3), store_primal=True)
        assert len(g.r) == 3 * len(g0.r)
        n1 = np.linalg.norm(g.a1)
        n2 = np.linalg.norm(g.a2)
        n0 = np.linalg.norm(g0.a1)
        assert np.isclose(n1, np.sqrt(3) * n0, atol=1e-8)
        assert np.isclose(n2, np.sqrt(3) * n0, atol=1e-8)
        angle = np.degrees(np.arccos(g.a1.dot(g.a2) / (n1 * n2)))
        assert np.isclose(min(angle, 180. - angle), 60., atol=1e-6)


def test_float_supercell_hamiltonian_folds():
    """The cell above is not only compact, its Hamiltonian is the right
    one: the supercell spectrum is the union of the primitive spectra
    at the three k-points that fold onto each supercell k-point, which
    is exactly what the degenerate cell failed."""
    check_unfolding(geometry.triangular_lattice(), np.sqrt(3))
    check_unfolding(geometry.honeycomb_lattice(), np.sqrt(3))
    # the compact cell of a given area is not always right-handed, and
    # sqrt(7) comes out with det(M)=-7, so the projector has to be blind
    # to the sign of the determinant as well
    check_unfolding(geometry.triangular_lattice(), np.sqrt(7))
    check_unfolding(geometry.square_lattice(), np.sqrt(2))


def test_unfolded_kpath_maps_back_to_the_primitive_path():
    """The k-path the unfolded bands are computed along has to trace the
    primitive Brillouin zone, meaning that Minv applied to it returns
    the primitive path itself."""
    g0 = geometry.honeycomb_lattice()
    for M in [[[2, 1, 0], [-1, 1, 0], [0, 0, 1]],
              [[3, 0, 0], [0, 3, 0], [0, 0, 1]],
              [[2, 1, 0], [0, 1, 0], [0, 0, 1]]]:
        g = g0.get_supercell(M, store_primal=True)
        Minv = np.linalg.inv(np.array(g.supercell_matrix, dtype=float))
        kp = g.get_unfolded_kpath(nk=20)
        k0 = np.array(g0.get_kpath(nk=20))
        assert np.allclose(np.array([Minv @ k for k in kp]), k0, atol=1e-10)


def test_unfolded_kpath_reduces_to_the_scaled_path_for_a_diagonal_supercell():
    """For a diagonal n x n supercell the answer is the primitive path
    times n, so the helper has to give back what the examples write by
    hand."""
    g0 = geometry.honeycomb_lattice()
    n = 3
    g = g0.get_supercell(n, store_primal=True)
    kp = g.get_unfolded_kpath(nk=30)
    assert np.allclose(kp, np.array(g.get_kpath(nk=30)) * n, atol=1e-12)


def test_unfolded_bands_along_the_unfolded_kpath_reproduce_the_primitive_bands():
    """End to end: along the k-path the helper returns, the supercell
    states carrying the full unfolding weight are at the energies of the
    primitive bands at the corresponding primitive k-point, so the
    unfolded band structure of a clean sqrt(3) x sqrt(3) supercell is
    the band structure of the primitive cell."""
    g0 = geometry.triangular_lattice()
    g = g0.get_supercell([[2, 1, 0], [-1, 1, 0], [0, 0, 1]], store_primal=True)
    h = g.get_hamiltonian(has_spin=False)
    h0 = g0.get_hamiltonian(has_spin=False)
    N = 3
    kpath = g.get_unfolded_kpath(nk=8)
    (ks, es, ds) = h.get_bands(operator="unfold", kpath=kpath)
    hk0 = h0.get_hk_gen()
    k0path = np.array(g0.get_kpath(nk=8))
    for (ik, k0) in enumerate(k0path):
        e0 = eigh(dense(hk0(k0)))[0]  # primitive bands at this k-point
        sel = np.abs(np.array(ks) - ik) < 1e-8  # states at this k-point
        eb = np.array(es)[sel]
        wb = np.array(ds)[sel]
        assert np.isclose(np.sum(wb), N * len(e0), atol=1e-6)
        kept = np.sort(eb[wb > N / 2.])  # those carrying the weight
        assert len(kept) == len(e0)
        assert np.allclose(kept, np.sort(e0), atol=1e-8)
