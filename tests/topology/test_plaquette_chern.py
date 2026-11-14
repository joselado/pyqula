"""The plaquette Chern number of topologytk/plaquette.py, which takes the
states from a callable, against h.get_chern() on the occupied Bloch states,
and the invariances it promises: the phase and the basis of the states
inside the group, and a positive metric of the inner product."""
import numpy as np
import pytest

from pyqula import geometry
from pyqula.topologytk.plaquette import chern_from_states, berry_fluxes


def _haldane(t2, spinful=False):
    g = geometry.honeycomb_lattice()
    h = g.get_hamiltonian(has_spin=spinful)
    h.add_haldane(t2)
    h.add_onsite(0.05) # a small shift, so zero is not a symmetry point
    return h


def _occupied(h):
    hk = h.get_hk_gen()
    def states(q):
        es, ws = np.linalg.eigh(hk(q))
        return ws.T[es < 0.] # rows are the occupied eigenvectors
    return states


@pytest.mark.parametrize("t2", [0.1, -0.1])
def test_matches_get_chern(t2):
    h = _haldane(t2)
    c0 = h.get_chern(nk=20, write=False)
    c = chern_from_states(_occupied(h), nq=12)
    assert abs(c - round(c)) < 1e-8 # an integer by construction
    assert abs(round(c)) == 1 # a Chern insulator, so not a vacuous check
    assert round(c) == round(c0)


def test_gauge_and_basis_invariance():
    """Random phases per q, and a random rotation inside a group of two
    states (the two spin copies), leave the integer unchanged"""
    h = _haldane(0.1, spinful=True)
    occ = _occupied(h)
    c0 = chern_from_states(occ, nq=10)
    assert abs(round(c0)) == 2
    def scrambled(q):
        v = occ(q)
        seed = int(1e6*(q[0] + 3.7*q[1])) % (2**31)
        rng = np.random.default_rng(seed)
        m = rng.normal(size=(2, 2)) + 1j*rng.normal(size=(2, 2)) # invertible
        return m@v # a new basis of the same subspace, not orthonormal
    assert round(chern_from_states(scrambled, nq=10)) == round(c0)


def test_positive_metric_gives_same_integer():
    h = _haldane(0.1)
    occ = _occupied(h)
    c0 = chern_from_states(occ, nq=12)
    metric = np.array([0.3, 2.5]) # positive, not the identity
    c1 = chern_from_states(occ, nq=12, metric=metric)
    assert round(c1) == round(c0)
    f0, _ = berry_fluxes(occ, nq=12)
    f1, _ = berry_fluxes(occ, nq=12, metric=metric)
    assert np.max(np.abs(f0 - f1)) > 1e-4 # the curvature itself does change


def test_shifted_mesh_gives_same_integer():
    h = _haldane(0.1)
    occ = _occupied(h)
    c0 = chern_from_states(occ, nq=12)
    c1 = chern_from_states(occ, nq=12, shift=(0.37/12, 0.11/12))
    assert round(c1) == round(c0)


def test_raises_when_count_changes():
    h = _haldane(0.1)
    hk = h.get_hk_gen()
    def states(q):
        es, ws = np.linalg.eigh(hk(q))
        return ws.T[es < 0.] if q[0] < 0.5 else ws.T # count jumps
    with pytest.raises(ValueError, match="number of states"):
        chern_from_states(states, nq=6)


def test_raises_on_unresolved_mesh():
    """A state that jumps to an orthogonal one between two neighbours of
    the mesh, which is what a group that is not separated from the rest
    looks like to the plaquette sum"""
    def states(q):
        return np.array([1., 0.]) if q[0] < 0.5 else np.array([0., 1.])
    with pytest.raises(ValueError, match="almost orthogonal"):
        chern_from_states(states, nq=6)


def test_the_mesh_and_its_guards():
    from pyqula.topologytk.plaquette import mesh
    q = mesh(4, shift=(0.03, 0.07))
    assert q.shape == (4, 4, 3)
    assert np.allclose(q[2, 1], [2./4 + 0.03, 1./4 + 0.07, 0.])
    assert np.allclose(q[:, :, 2], 0.)
    for nq in (1, 2.5):
        with pytest.raises(ValueError, match="at least 2"):
            mesh(nq)
    with pytest.raises(ValueError, match="two components"):
        mesh(4, shift=(0.1, 0.2, 0.3))


def test_states_given_on_the_mesh_are_those_of_the_callable():
    """mesh_berry_fluxes takes states already computed at the points of
    mesh(nq, shift), and gives the fluxes of the callable on that mesh;
    its checks of the shape are the ones the callable goes through"""
    from pyqula.topologytk.plaquette import mesh, mesh_berry_fluxes
    h = _haldane(0.1)
    occ = _occupied(h)
    nq, shift = 7, (0.03, 0.07)
    V = [[occ(q) for q in row] for row in mesh(nq, shift)]
    f0, w0 = berry_fluxes(occ, nq=nq, shift=shift)
    f1, w1 = mesh_berry_fluxes(V)
    assert np.max(np.abs(f0 - f1)) < 1e-14 and w0 == w1
    assert abs(round(np.sum(f1)/(2.*np.pi))) == 1
    V[3][2] = np.ones((2, 2)) # two states at one point
    with pytest.raises(ValueError, match="number of states"):
        mesh_berry_fluxes(V)
    with pytest.raises(ValueError, match="nq x nq mesh"):
        mesh_berry_fluxes([row[0:3] for row in V])


def test_copies_of_one_state_have_n_times_its_flux():
    """A group of n modes that share one vector, as the three components of
    a spin triplet, has n times the flux of one, which is why the record
    reports the largest flux: once n times it passes pi the integer is
    wrong with no other sign"""
    from pyqula.topologytk.plaquette import mesh, mesh_berry_fluxes, ModeChern
    h = _haldane(0.1)
    occ = _occupied(h)
    nq = 6
    V = [[occ(q) for q in row] for row in mesh(nq)]
    f1, _ = mesh_berry_fluxes(V)
    V3 = [[np.kron(np.identity(3), v) for v in row] for row in V]
    f3, _ = mesh_berry_fluxes(V3)
    r1 = ModeChern(f1, 1., mesh(nq), np.zeros((nq, nq, 1)), 0.1, None)
    r3 = ModeChern(f3, 1., mesh(nq), np.zeros((nq, nq, 3)), 0.1, None)
    assert abs(r1.max_flux - np.max(np.abs(f1))) < 1e-15
    assert 0.1 < r1.max_flux < np.pi/3. # 3 copies stay below pi here
    assert np.max(np.abs(f3 - 3.*f1)) < 1e-12
    assert abs(r3.chern - 3.*r1.chern) < 1e-10 and abs(r3.max_flux
                                                       - 3.*r1.max_flux) < 1e-12
    assert r1.bound is None and r1.energies.shape == (nq, nq, 1)


def test_the_routes_share_one_default_mesh_and_one_2d_guard():
    import inspect
    from pyqula.topologytk import plaquette
    from pyqula.bsetk import topology as bt
    from pyqula.chitk import kernelchern as kc
    for f in (plaquette.mesh, plaquette.berry_fluxes,
              plaquette.chern_from_states, plaquette.states_on_mesh,
              bt.mode_chern, bt.mode_berry_fluxes, kc.kernel_chern):
        assert inspect.signature(f).parameters["nq"].default == 12, f
    h1 = geometry.chain().get_hamiltonian()
    with pytest.raises(ValueError, match="of a band needs a two-dimensional"):
        plaquette.require_2d(h1, "a band")
    plaquette.require_2d(_haldane(0.1), "a band")
