import numpy as np
import pytest

from pyqula import geometry
from pyqula.chitk import localspinchi, rpa, spinchi

# chitk.localspinchi builds the bubble on one of three routes: one spin
# species for a Hamiltonian that is the same for the two spins, the bands
# of the two spins apart when S_z is conserved, and the full spinful
# states otherwise. The two fast routes leave out matrix elements that
# vanish exactly, so they must give the numbers of the general route and
# of chitk.chiAB to roundoff. The local response, q=None, takes the states
# at k+q from one diagonalization of the mesh, and a degenerate bubble is
# dressed block by block by a vertex with one block per component; both
# must give the result of the direct sum.

ES = np.linspace(-1., 3., 7)


def honeycomb(field=None):
    g = geometry.honeycomb_lattice().get_supercell((2, 1))
    h = g.get_hamiltonian(has_spin=True)
    h.add_onsite(0.3) # away from half filling
    if field is not None: h.add_zeeman(field)
    return h


def general_route(h, q, **kw):
    """The bubble of the spinful route, which every Hamiltonian can take"""
    energies, T, sites, route, frame = localspinchi._arguments(h, ES,
            kw["delta"], kw.get("T", None), None)
    acc = localspinchi._bubble_at_q(h, localspinchi._SPINFUL,
            np.array(q, dtype=np.float64), energies, kw["nk"], kw["delta"],
            T, sites)
    return acc


def isotropic_vertex(n, rng):
    """A real-space vertex with the same block for the three components
    and none between them, as the exchange of a spinon mean field"""
    k0, k1 = rng.normal(size=(n, n)), 0.3*rng.normal(size=(n, n))
    I3 = np.identity(3)
    return {(0, 0, 0): np.kron(I3, 0.3*(k0 + k0.T)), (1, 0, 0): np.kron(I3, k1),
            (-1, 0, 0): np.kron(I3, k1.T)}


@pytest.mark.parametrize("q", [[0.13, 0.31, 0.], [0., 0., 0.], [0.5, 0., 0.]])
@pytest.mark.parametrize("T", [None, 0.3])
def test_collinear_route_agrees_with_chiAB(q, T):
    """A Zeeman field along z conserves S_z: the bands of the two spins
    are diagonalized apart, and the result is the one of chitk.chiAB and of
    the general route, with no element between z and x or y"""
    h = honeycomb([0., 0., 0.3])
    assert localspinchi._route(h) == localspinchi._COLLINEAR
    kw = dict(q=q, energies=ES, delta=0.05, nk=6, T=T)
    ref = rpa.chi_ops_RPA(h, ops=spinchi._full_spin_operators(h), V=None,
                          **kw)[1]
    out = localspinchi.bare_spin_chi(h, **kw)[1]
    assert np.max(np.abs(ref)) > 0.1
    assert np.allclose(out, ref, atol=1e-12)
    assert np.allclose(out, general_route(h, **kw), atol=1e-12)
    m = len(h.geometry.r)
    assert np.max(np.abs(out[:, :2*m, 2*m:])) < 1e-14
    assert np.max(np.abs(out[:, 0:m, m:2*m])) > 1e-3 # chi^xy of the field


def test_collinear_route_on_the_real_axis():
    """With delta=0 the pairs of zero factor are left out on the collinear
    route as on the general one, so the response is finite at w=0, q=0"""
    h = geometry.honeycomb_lattice().get_hamiltonian(has_spin=True)
    h.add_sublattice_imbalance(0.6) # gapped at half filling
    h.add_zeeman([0., 0., 0.2])
    assert localspinchi._route(h) == localspinchi._COLLINEAR
    kw = dict(q=[0., 0., 0.], nk=4, delta=0., T=1e-3)
    with np.errstate(all="ignore"): # a wrong code gives 0/0
        out = localspinchi.bare_spin_chi(h, energies=[0., 1e-7], **kw)[1]
        energies, T, sites, route, frame = localspinchi._arguments(h,
                [0., 1e-7], 0., 1e-3, None)
        ref = localspinchi._bubble_at_q(h, localspinchi._SPINFUL,
                np.zeros(3), energies, 4, 0., 1e-3, sites)
    assert np.all(np.isfinite(out)) and np.max(np.abs(out[0])) > 0.05
    assert np.allclose(out, ref, atol=1e-12)


@pytest.mark.parametrize("q", [[0.13, 0.31, 0.], [0.5, 0., 0.]])
def test_block_dressing_agrees_with_chi_ops_RPA(q):
    """A spin-degenerate bubble dressed by a vertex with one block per
    component: the block dressing gives the full 3N x 3N one, and the
    static kernel gives the 3N eigenvalues of the full kernel"""
    h = honeycomb()
    n = len(h.geometry.r)
    V = isotropic_vertex(n, np.random.default_rng(1))
    kw = dict(q=q, energies=ES, delta=0.05, nk=6)
    ref = rpa.chi_ops_RPA(h, ops=spinchi._full_spin_operators(h), V=V,
                          **kw)[1]
    out = localspinchi.spin_chi(h, V=V, **kw)[1]
    assert np.max(np.abs(out - localspinchi.bare_spin_chi(h, **kw)[1])) > 0.01
    assert np.allclose(out, ref, atol=1e-11)
    ev = localspinchi.static_kernel_eigenvalues(h, V, q, nk=6, delta=0.05)
    chi0 = localspinchi.bare_spin_chi(h, q=q, energies=[0.], nk=6,
                                      delta=0.05)[1][0]
    Vq = np.asarray(rpa.interaction_at_q(V, h, q))
    assert localspinchi._common_block(Vq, n) is not None # the block route
    full = np.linalg.eigvals(np.identity(3*n) - Vq@chi0)
    full = full[np.argsort(full.real)]
    assert len(ev) == 3*n
    assert np.allclose(ev, full, atol=1e-12)


@pytest.mark.parametrize("kind", ["xxz", "dm"])
def test_vertex_that_tells_the_components_apart_takes_the_full_kernel(kind):
    """A spin-degenerate bubble with a vertex that is not the same block
    for the three components, an exchange weaker along z or an
    antisymmetric one between x and y: the block dressing does not apply,
    and the response and the static kernel are those of the 3N x 3N
    matrices"""
    h = honeycomb()
    n = len(h.geometry.r)
    V = isotropic_vertex(n, np.random.default_rng(3))
    if kind == "xxz":
        V = {d: np.kron(np.diag([1., 1., 0.5]), np.identity(n))@m
             for d, m in V.items()}
    else:
        A = np.zeros((3, 3))
        A[0, 1], A[1, 0] = 1., -1.
        V[(0, 0, 0)] = V[(0, 0, 0)] + 0.2*np.kron(A, np.identity(n))
    q = [0.13, 0.31, 0.]
    Vq = np.asarray(rpa.interaction_at_q(V, h, q))
    assert localspinchi._common_block(Vq, n) is None
    kw = dict(q=q, energies=ES, delta=0.05, nk=6)
    ref = rpa.chi_ops_RPA(h, ops=spinchi._full_spin_operators(h), V=V,
                          **kw)[1]
    out = localspinchi.spin_chi(h, V=V, **kw)[1]
    assert np.max(np.abs(ref)) > 0.1
    assert np.allclose(out, ref, atol=1e-11)
    ev = localspinchi.static_kernel_eigenvalues(h, V, q, nk=6, delta=0.05)
    chi0 = localspinchi.bare_spin_chi(h, q=q, energies=[0.], nk=6,
                                      delta=0.05)[1][0]
    full = np.linalg.eigvals(np.identity(3*n) - Vq@chi0)
    assert np.allclose(ev, full[np.argsort(full.real)], atol=1e-12)


def turning(r):
    """A field that turns from site to site, so that the Hamiltonian has
    two spin directions and takes the general route. A uniform field in
    any direction takes the collinear one, in its own spin frame."""
    return [0.2, 0., 0.] if r[0] > 0. else [0., 0., 0.3]


FIELDS = [None, [0., 0., 0.3], [0.2, -0.1, 0.3], turning]
FIELD_IDS = ["none", "z", "tilted", "turning"]


def raising_from_blocks(chi, m):
    """chi^xx + chi^yy - i chi^xy + i chi^yx from the full response"""
    x, y = slice(0, m), slice(m, 2*m)
    return chi[:, x, x] + chi[:, y, y] - 1j*chi[:, x, y] + 1j*chi[:, y, x]


@pytest.mark.parametrize("field", FIELDS, ids=FIELD_IDS)
@pytest.mark.parametrize("q", [[0.13, 0.31, 0.], None])
def test_raising_response_is_the_combination_of_the_blocks(field, q):
    """bare_raising_chi contracts the matrix element of S^+ alone, on the
    three routes and for the sites asked for; it is the combination of the
    blocks of the full response"""
    h = honeycomb(field)
    if field is turning:
        assert localspinchi._route(h) == localspinchi._SPINFUL
    sites = [3, 0]
    kw = dict(q=q, energies=ES, delta=0.05, nk=3, sites=sites)
    out = localspinchi.bare_raising_chi(h, **kw)[1]
    ref = raising_from_blocks(localspinchi.bare_spin_chi(h, **kw)[1],
                              len(sites))
    assert out.shape == (len(ES), 2, 2) and np.max(np.abs(ref)) > 0.05
    assert np.allclose(out, ref, atol=1e-12)


@pytest.mark.parametrize("field", FIELDS, ids=FIELD_IDS)
def test_local_response_is_the_average_over_the_mesh_on_every_route(field):
    """q=None takes the states at k+q from one diagonalization of the
    mesh; it is the average of the response at each q of the mesh, also
    dressed, on the three routes"""
    h = honeycomb(field)
    if field is turning:
        assert localspinchi._route(h) == localspinchi._SPINFUL
    n = len(h.geometry.r)
    V = isotropic_vertex(n, np.random.default_rng(2))
    kw = dict(energies=ES, delta=0.05, nk=3)
    for vertex in (None, V):
        out = localspinchi.spin_chi(h, V=vertex, **kw)[1]
        ref = np.mean([localspinchi.spin_chi(h, V=vertex, q=q, **kw)[1]
                       for q in h.geometry.get_kmesh(nk=3)], axis=0)
        assert np.max(np.abs(ref)) > 0.05
        assert np.allclose(out, ref, atol=1e-12)
