import numpy as np
import pytest

from pyqula import geometry


def test_spin_operators_refuse_a_spinless_hamiltonian():
    """A spin operator on a has_spin=False Hamiltonian used to come back as
    None, which every observable reads as "no operator", i.e. the identity:
    <sz> silently returned the charge density and a spin-projected DOS
    equalled the unprojected one. It must refuse instead."""
    h = geometry.chain().get_hamiltonian(has_spin=False)
    for name in ["sz", "sx", "sy"]:
        with pytest.raises(ValueError):
            h.get_vev(name)


@pytest.mark.parametrize("name", ["spair", "singlet"])
def test_pairing_operators_refuse_a_spinless_nambu_hamiltonian(name):
    """The pairing operators are singlet/d-vector components in the
    spin x electron-hole basis. Built on a spinless Nambu Hamiltonian they
    came out twice the size of its Hilbert space instead of raising."""
    h = geometry.chain().get_hamiltonian(has_spin=False)
    h.add_swave(0.2)
    with pytest.raises(NotImplementedError):
        h.get_operator(name)
    hs = geometry.chain().get_hamiltonian()
    hs.add_swave(0.2)
    op = hs.get_operator(name).get_matrix()
    assert op.shape == hs.intra.shape  # spinful Nambu is fine


@pytest.mark.parametrize("name", ["spair", "singlet"])
@pytest.mark.parametrize("has_spin", [True, False])
def test_pairing_operators_refuse_a_hamiltonian_without_nambu(name, has_spin):
    """A pairing operator on a Hamiltonian with no electron-hole degree of
    freedom used to be built anyway: "singlet" calls add_swave on a copy,
    which promotes that copy into Nambu space, so the operator came back at
    twice the Hilbert-space dimension and only failed later, inside a raw
    numpy matmul that named neither pyqula nor the requirement."""
    h = geometry.chain().get_hamiltonian(has_spin=has_spin)
    with pytest.raises(ValueError):
        h.get_operator(name)


def test_velocity_operator_is_c3_symmetric_on_a_honeycomb_lattice():
    """The 2D velocity operator must inherit the C3 symmetry of the
    lattice: three symmetry-equivalent k-points of the same band must have
    the same speed. Built out of the raw k-derivative in the lattice gauge
    it did not -- the intracell bond of the honeycomb lattice was missing
    (and the x and y derivative orders were swapped)."""
    g = geometry.honeycomb_lattice()
    h = g.get_hamiltonian()
    op = h.get_operator("velocity")
    hk = h.get_hk_gen()
    B = np.array([g.b1, g.b2])[:, :2]

    def speeds(kred):
        k = np.array([kred[0], kred[1], 0.])
        es, ws = np.linalg.eigh(np.array(hk(k)))
        return es, np.array([op.braket(ws[:, i], k=k).real
                             for i in range(len(es))])

    k0 = np.array([0.13, 0.27])
    ref_e, ref_v = speeds(k0)
    K0 = k0[0]*g.b1[:2] + k0[1]*g.b2[:2]
    for n in [1, 2]:
        th = 2*np.pi*n/3.
        R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        kr = np.linalg.solve(B.T, R@K0)
        e, v = speeds(kr)
        assert np.allclose(e, ref_e, atol=1e-10), (n, e, ref_e)
        assert np.allclose(v, ref_v, atol=1e-8), (n, v, ref_v)


def test_sparse_hamiltonians_reach_the_density_matrix_routines():
    """is_sparse=True used to kill get_vev/get_density_matrix with an
    opaque numba TypingError, because the batched H(k) array was built out
    of sparse matrices. The sparse answer must equal the dense one."""
    g = geometry.honeycomb_lattice().supercell(2)
    hd = g.get_hamiltonian()
    hs = g.get_hamiltonian(is_sparse=True)
    for h in (hd, hs):
        h.add_exchange([0., 0., 0.3])
    assert np.allclose(hd.get_vev("sz", nk=6), hs.get_vev("sz", nk=6),
                       atol=1e-10)
    assert np.allclose(np.array(hd.get_density_matrix(nk=6)),
                       np.array(hs.get_density_matrix(nk=6)), atol=1e-10)


def test_sublattice_lowering_generator_starts_only_on_one_sublattice():
    """operators.get_sigma_minus builds a first-neighbor hopping that
    starts only on sublattice A, so its intra-cell block is sigma_minus.
    It was unreachable until get_hamiltonian started honouring `fun` (the
    old name of `tij`): the function was dropped, a plain first-neighbor
    Hamiltonian was built instead, and the site at index 0 -- which is
    exactly sublattice A here -- was reported as not found by get_index."""
    from pyqula import operators
    g = geometry.honeycomb_lattice()
    h = g.get_hamiltonian(has_spin=False)
    operators.get_sigma_minus(h)  # must not raise
    a = [i for i in range(len(g.r)) if g.sublattice[i] == 1]
    assert len(a) > 0

    def fun(r1, r2):
        i1 = g.get_index(r1, replicas=True)
        if i1 is None or g.sublattice[i1] != 1:
            return 0.0
        dr = r1 - r2
        return 1.0 if 0.9 < dr.dot(dr) < 1.1 else 0.0

    intra = np.array(g.get_hamiltonian(has_spin=False, tij=fun).intra)
    for i in range(intra.shape[0]):
        if i not in a:
            assert np.allclose(intra[i, :], 0.), (i, intra)
    assert np.max(np.abs(intra)) > 0.5   # and it is not empty
