import numpy as np

from pyqula import geometry


def _model(is_sparse):
    g = geometry.square_lattice()
    h = g.get_hamiltonian(is_sparse=is_sparse)
    h.add_onsite(-0.6)
    h.add_swave(0.3)
    return h


def test_superfluid_weight_is_the_same_sparse_and_dense():
    """is_sparse=True is a public get_hamiltonian argument and add_swave
    propagates it, so every superfluid-weight route must work on a sparse
    Hamiltonian and give the dense answer. The twist operators used to
    densify with np.asarray, which on a scipy sparse matrix returns a 0-d
    object array instead of a dense one, so all four entry points died with
    an IndexError on the very next shape lookup."""
    hd = _model(False)
    hs = _model(True)
    Dd = hd.get_superfluid_weight(nk=6)
    Ds = hs.get_superfluid_weight(nk=6)
    assert np.allclose(Dd, Ds, atol=1e-10), (Dd, Ds)
    assert Dd[0, 0] > 0.  # and the model really is superconducting


def test_every_superfluid_entry_point_accepts_a_sparse_hamiltonian():
    """TwistOperators is the constructor shared by the Kubo route, the
    finite-difference route, the conventional/geometric decomposition and
    the BKT temperature, so all four failed identically."""
    hd = _model(False)
    hs = _model(True)
    assert np.allclose(hs.get_superfluid_weight(nk=6, mode="finite_difference"),
                       hd.get_superfluid_weight(nk=6, mode="finite_difference"),
                       atol=1e-8)
    cs = hs.get_superfluid_weight(nk=6, decompose=True)
    cd = hd.get_superfluid_weight(nk=6, decompose=True)
    assert set(cs) == set(cd)
    for key in ["total", "conventional", "geometric"]:
        assert np.allclose(np.array(cs[key]), np.array(cd[key]), atol=1e-10)
    assert abs(hs.get_bkt_temperature(nk=6)
               - hd.get_bkt_temperature(nk=6)) < 1e-8
