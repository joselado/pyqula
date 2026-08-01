import numpy as np

from pyqula import geometry
from pyqula import classicalspin
from pyqula.classicalspintk.align import most_perpendicular_vector, most_perp_basis


def _small_heisenberg_model(seed=0):
    """Small 0d Heisenberg spin model, minimized, small enough to run
    in well under a second"""
    np.random.seed(seed)
    g = geometry.triangular_lattice()
    g = g.get_supercell(3) # 9 sites
    sm = classicalspin.SpinModel(g)
    sm.add_heisenberg(Jij=[1.0], Jm=[1., 1., 1.])
    sm.minimize_energy(tries=1)
    return sm


def test_local_energy_sum_matches_total_energy():
    sm = _small_heisenberg_model()
    local = sm.get_local_energy()
    assert np.isclose(np.sum(local), sm.energy(), atol=1e-4)


def test_local_energy_handles_site_without_exchange_pairs():
    """Regression test: get_local_energy used to crash with an
    IndexError for any site with zero exchange interactions, because
    the empty-pairs list became a (0,) array instead of (0,2), which
    broke jax's indsjs[:,0] indexing"""
    g = geometry.chain()
    g = g.get_supercell(2)
    sm = classicalspin.SpinModel(g)
    sm.add_field([0., 0., 1.0]) # only a field, no exchange interactions
    local = sm.get_local_energy()
    assert np.allclose(local, 1.0) # each spin starts along +z, aligned with b


def test_regroup_merges_duplicate_and_reverse_pairs():
    pairs = np.array([[0, 1], [1, 0], [0, 1]])
    j0 = np.diag([1., 2., 3.])
    j1 = np.diag([4., 5., 6.]) # stored for the reversed (1,0) pair
    j2 = np.diag([0.5, 0.5, 0.5])
    js = np.array([j0, j1, j2])
    outp, outj = classicalspin.regroup(pairs, js)
    assert outp.shape == (1, 2)
    assert tuple(outp[0]) == (0, 1)
    expected = j0 + j1.transpose() + j2
    assert np.allclose(outj[0], expected)


def test_add_tensor_matches_add_heisenberg():
    """add_tensor with a Heisenberg-generating function must reproduce
    the same energy as add_heisenberg's Hamiltonian-based pair
    construction -- two independent code paths for the same model"""
    g = geometry.chain()
    g = g.get_supercell(4)
    g.dimensionality = 0

    sm1 = classicalspin.SpinModel(g)
    sm1.add_heisenberg(Jij=[1.0], Jm=[1., 1., 1.])

    sm2 = classicalspin.SpinModel(g)
    fun = classicalspin.generating_functions(name="Heisenberg", J=1.0)
    sm2.add_tensor(fun)

    np.random.seed(2)
    theta = np.random.random(len(g.r)) * np.pi
    phi = np.random.random(len(g.r)) * np.pi * 2
    sm1.theta, sm1.phi = theta.copy(), phi.copy()
    sm2.theta, sm2.phi = theta.copy(), phi.copy()
    assert np.isclose(sm1.energy(), sm2.energy())


def test_most_perpendicular_vector_is_orthogonal_to_aligned_set():
    np.random.seed(3)
    vs = np.array([[0., 0., 1.] for _ in range(5)])
    v = most_perpendicular_vector(vs)
    assert abs(v[2]) < 1e-3 # perpendicular to the common z axis


def test_most_perp_basis_puts_aligned_vectors_in_plane():
    np.random.seed(4)
    vs = np.array([[0., 0., 1.] for _ in range(5)])
    ovs = most_perp_basis(vs)
    # in the rotated frame, the component along the most-perpendicular
    # vector (3rd axis) must vanish since the inputs were all
    # perpendicular to it by construction
    assert np.allclose(ovs[:, 2], 0., atol=1e-3)
