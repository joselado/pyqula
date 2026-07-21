import numpy as np
import pytest

from pyqula import geometry
from pyqula.symmetrytk import pointgroup
from pyqula.wanniertk import wannierize as wz


def test_mesh_index_of_uses_minimum_image_distance_not_naive_mod():
    """_mesh_index_of must treat a component that lands on the wrong side
    of the [0,1) wrap by pure floating-point roundoff (e.g. -1e-15, which
    np.mod(...,1.0) turns into ~0.999999999999999) as the ~0 distance it
    actually is on the torus -- not as ~1 away. Regression test for a
    real reproduction: a dense unimodular 3x3 M (of the kind
    geometry.pyrochlore_lattice()'s own symmetries produce) whose
    k_image's matrix inverse/matmul accumulates exactly this roundoff at
    a boundary that is mathematically exactly 0, on a mesh that is
    genuinely closed under the corresponding symmetry."""
    M = np.array([[-1, 0, -1], [-1, 1, 0], [-1, 1, 1]])
    assert abs(abs(np.linalg.det(M)) - 1) < 1e-9  # genuinely unimodular
    mp_grid = np.array([3, 3, 3])
    kpt_latt = wz._monkhorst_pack(mp_grid)
    images = np.zeros_like(kpt_latt)
    images[:3] = np.linalg.inv(M).T @ kpt_latt[:3]

    idx = wz._mesh_index_of(kpt_latt, images, atol=1e-6, on_miss=lambda k: f"miss at {k}")
    # a genuine unimodular-M action on a Gamma-centred mesh is a bijection
    assert sorted(idx.tolist()) == list(range(kpt_latt.shape[1]))


def test_compile_symmetry_rejects_multi_orbital_per_site_with_clear_error():
    """A Hamiltonian with more than one orbital per site (times spin) is
    not supported by point-group enforcement (pointgroup.py only models a
    site permutation + spin rotation, not general orbital character) --
    this must raise pyqula's own clear NotImplementedError, not an opaque
    numpy shape-mismatch deep inside the P(k) verification."""
    g = geometry.chain(n=1)
    h = g.get_hamiltonian(has_spin=False)
    h.intra = np.zeros((3, 3), dtype=complex)  # simulate 3 orbitals/site
    h.get_hk_gen = lambda *a, **k: (lambda kk: np.eye(3, dtype=complex))

    identity = pointgroup.SymmetryOperation(np.eye(3), name="E")
    with pytest.raises(NotImplementedError):
        pointgroup.compile_symmetry(h, identity)


def test_auto_symmetries_full_manifold_reproduces_spectrum():
    g = geometry.honeycomb_lattice()
    h = g.get_hamiltonian(has_spin=False)
    hwan = h.get_wannier_hamiltonian(bands=[0, 1], nk=8, num_iter=200, symmetries="auto",
                                      trial_vectors=np.eye(2, dtype=complex))
    assert hasattr(hwan, "wannier_symmetries")
    assert any(c.op.name == "E" for c in hwan.wannier_symmetries)

    hk_ref, hk_wan = h.get_hk_gen(), hwan.get_hk_gen()
    rng = np.random.default_rng(0)
    for _ in range(6):
        k = np.zeros(3); k[:2] = rng.random(2)
        e_ref = np.sort(np.linalg.eigvalsh(hk_ref(k)))
        e_wan = np.sort(np.linalg.eigvalsh(hk_wan(k)))
        assert np.max(np.abs(e_ref - e_wan)) < 1e-6


def test_no_symmetries_by_default():
    g = geometry.honeycomb_lattice()
    h = g.get_hamiltonian(has_spin=False)
    hwan = h.get_wannier_hamiltonian(bands=[0, 1], nk=8, num_iter=200)
    assert not hasattr(hwan, "wannier_symmetries")


def test_explicit_symmetries_list_is_verified_and_used():
    g = geometry.honeycomb_lattice()
    h = g.get_hamiltonian(has_spin=False)
    inversion = pointgroup.SymmetryOperation(-np.eye(3), name="inversion")
    hwan = h.get_wannier_hamiltonian(bands=[0, 1], nk=8, num_iter=200,
                                      symmetries=[inversion])
    names = {c.op.name for c in hwan.wannier_symmetries}
    assert "inversion" in names
    assert "E" in names


def test_explicit_symmetry_not_present_raises():
    g = geometry.honeycomb_lattice()
    h = g.get_hamiltonian(has_spin=False)
    h.intra[0, 0] += 0.7  # breaks inversion (swaps sublattices)
    inversion = pointgroup.SymmetryOperation(-np.eye(3), name="inversion")
    with pytest.raises(ValueError):
        h.get_wannier_hamiltonian(bands=[0, 1], nk=8, num_iter=200,
                                   symmetries=[inversion])


def test_has_eh_symmetries_not_implemented():
    g = geometry.chain()
    h = g.get_hamiltonian(has_spin=True)
    h.add_swave(0.1)
    assert h.has_eh
    with pytest.raises(NotImplementedError):
        h.get_wannier_hamiltonian(bands=[0, h.intra.shape[0] - 1], nk=6,
                                   symmetries="auto")


def test_degenerate_multiplet_selection_raises():
    # kagome's bottom two (Dirac-touching) bands are not, at every mesh
    # k-point, a subspace the in-plane mirrors map isomorphically onto
    # themselves -- this must be caught, not silently mis-symmetrized
    g = geometry.kagome_lattice()
    h = g.get_hamiltonian(has_spin=False)
    with pytest.raises(ValueError):
        h.get_wannier_hamiltonian(bands=[0, 1], nk=6, num_iter=50, symmetries="auto")


def test_group_averaging_is_exactly_covariant_on_synthetic_data():
    """White-box check of the actual math in _enforce_point_group_symmetry:
    build a deliberately non-covariant per-k gauge (true eigenvectors of a
    genuine invariant 2-band subspace, rotated by an independent random
    unitary at every mesh k-point -- i.e. exactly the kind of gauge an
    unconstrained CG run could converge to) and confirm the group-averaged
    result satisfies D(R,k) H(k) D(R,k)^dagger == H(k_dst) to machine
    precision for every non-identity element."""
    g = geometry.chain(n=4)
    h = g.get_hamiltonian(has_spin=False)
    h.intra += np.diag([0.8, -0.8, -0.8, 0.8])  # inversion-symmetric gap opener

    inversion = pointgroup.SymmetryOperation(-np.eye(3), name="inversion")
    group = pointgroup.close_group(h, [inversion])
    assert len(group) == 2  # {E, inversion}

    nk = 12
    mp_grid = np.array([nk, 1, 1])
    kpt_latt = wz._monkhorst_pack(mp_grid)
    num_kpts = kpt_latt.shape[1]
    num_orb, band_indices, num_wann = 4, [0, 1], 2

    hk_gen = h.get_hk_gen()
    rng = np.random.default_rng(11)
    W_k_mesh = np.empty((num_orb, num_wann, num_kpts), dtype=complex)
    H_k_mesh = np.empty((num_wann, num_wann, num_kpts), dtype=complex)
    for k in range(num_kpts):
        Hk = np.asarray(hk_gen(kpt_latt[:, k]), dtype=complex)
        _, v = np.linalg.eigh(Hk)
        csel = v[:, band_indices]
        A = rng.standard_normal((num_wann, num_wann)) + 1j * rng.standard_normal((num_wann, num_wann))
        q, _ = np.linalg.qr(A)
        W = csel @ q
        W_k_mesh[:, :, k] = W
        H_k_mesh[:, :, k] = W.conj().T @ Hk @ W

    H_sym = wz._enforce_point_group_symmetry(H_k_mesh, W_k_mesh, kpt_latt, group)

    for compiled in group:
        if compiled.op.name == "E":
            continue
        tgt = wz._symmetry_target_index(compiled, kpt_latt)
        D_all = wz._point_group_wannier_operators(compiled, W_k_mesh, kpt_latt, tgt)
        for k in range(num_kpts):
            D = D_all[:, :, k]
            lhs = H_sym[:, :, tgt[k]]
            rhs = D @ H_sym[:, :, k] @ D.conj().T
            assert np.max(np.abs(lhs - rhs)) < 1e-10
