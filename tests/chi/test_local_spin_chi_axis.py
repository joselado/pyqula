import numpy as np
import pytest

from pyqula import geometry
from pyqula.chitk import localspinchi, rpa, spinchi
from pyqula.multihopping import MultiHopping

# A Hamiltonian that depends on the spin along one direction n, a Zeeman
# field along n, conserves the spin along n. chitk.localspinchi then
# builds the bubble on its collinear route in the spin frame where n is
# the z axis and rotates the response back, so the cost does not depend on
# the direction of the field. The result must be the one of the general
# route and of chitk.chiAB for a direction that is not an axis, bare,
# dressed with a vertex that tells the components apart, for the local
# response and for chi^{+-}. A Hamiltonian with two spin directions must
# still take the general route.

ES = np.linspace(-1., 3., 7)
FIELD = [0.2, -0.1, 0.25] # not along an axis, not in a coordinate plane
Q = [0.13, 0.31, 0.]


def honeycomb(field=None):
    g = geometry.honeycomb_lattice().get_supercell((2, 1))
    h = g.get_hamiltonian(has_spin=True)
    h.add_onsite(0.3) # away from half filling
    if field is not None: h.add_zeeman(field)
    return h


def general_route(h, q, nk=6, delta=0.05, T=None, energies=ES):
    """The response of the spinful route in the basis of h, which every
    Hamiltonian can take"""
    energies, T, sites, route, frame = localspinchi._arguments(h, energies,
            delta, T, None)
    return localspinchi._bubble_at_q(h, localspinchi._SPINFUL,
            np.array(q, dtype=np.float64), energies, nk, delta, T, sites)


def anisotropic_vertex(n, rng):
    """A real-space vertex that is weaker along z and has an antisymmetric
    part between x and y, so that a response rotated to the wrong frame is
    dressed wrongly"""
    k0, k1 = rng.normal(size=(n, n)), 0.3*rng.normal(size=(n, n))
    D = np.diag([1., 0.8, 0.5])
    A = np.zeros((3, 3))
    A[0, 1], A[1, 0] = 1., -1.
    return {(0, 0, 0): np.kron(D, 0.3*(k0 + k0.T)) + 0.2*np.kron(A, np.identity(n)),
            (1, 0, 0): np.kron(D, k1), (-1, 0, 0): np.kron(D, k1.T)}


def test_spin_axis_of_a_hamiltonian():
    """One direction: a field, also against z, and a spin-orbit coupling
    along z with a field along z. None: no dependence on the spin, a field
    that turns from site to site, a Rashba coupling."""
    axis = localspinchi._spin_axis
    n = np.array(FIELD)/np.linalg.norm(FIELD)
    found = axis(honeycomb(FIELD))
    assert abs(abs(found@n) - 1.) < 1e-12
    assert abs(abs(axis(honeycomb([0., 0., -0.3]))[2]) - 1.) < 1e-12
    assert axis(honeycomb()) is None
    h = honeycomb([0., 0., 0.2])
    h.add_kane_mele(0.1)
    assert abs(abs(axis(h)[2]) - 1.) < 1e-12
    h = honeycomb()
    h.add_zeeman(lambda r: [0.2, 0., 0.] if r[0] > 0. else [0., 0., 0.2])
    assert axis(h) is None
    h = honeycomb([0., 0., 0.2])
    h.add_rashba(0.1)
    assert axis(h) is None


def test_routes_and_frames():
    h = honeycomb(FIELD)
    assert localspinchi._route(h) == localspinchi._COLLINEAR
    route, (U, R) = localspinchi._route_frame(h)
    assert route == localspinchi._COLLINEAR
    n = np.array(FIELD)/np.linalg.norm(FIELD)
    # U takes n.sigma to +-sigma_z, and R is the rotation that goes with it
    ns = sum(ni*s for ni, s in zip(n, localspinchi._SIGMA))
    assert np.allclose(np.abs(U.conj().T@ns@U), np.diag([1., 1.]), atol=1e-12)
    assert np.allclose(R@R.T, np.identity(3), atol=1e-12)
    assert abs(np.linalg.det(R) - 1.) < 1e-12
    assert np.allclose(np.abs(R[:, 2]), np.abs(n), atol=1e-12)
    # S_z conserved: the frame of h itself, as before
    hz = honeycomb([0., 0., 0.3])
    assert localspinchi._route(hz) == localspinchi._COLLINEAR
    assert localspinchi._route_frame(hz) == (localspinchi._COLLINEAR, None)
    # two directions: the general route
    h2 = honeycomb()
    h2.add_zeeman(lambda r: [0.2, 0., 0.] if r[0] > 0. else [0., 0., 0.2])
    assert localspinchi._route(h2) == localspinchi._SPINFUL


@pytest.mark.parametrize("q", [Q, [0., 0., 0.], [0.5, 0., 0.]])
@pytest.mark.parametrize("T", [None, 0.3])
def test_field_off_z_agrees_with_the_general_route(q, T):
    h = honeycomb(FIELD)
    kw = dict(q=q, energies=ES, delta=0.05, nk=6, T=T)
    out = localspinchi.bare_spin_chi(h, **kw)[1]
    ref = rpa.chi_ops_RPA(h, ops=spinchi._full_spin_operators(h), V=None,
                          **kw)[1]
    assert np.max(np.abs(ref)) > 0.1
    assert np.allclose(out, ref, atol=1e-12)
    assert np.allclose(out, general_route(h, q, T=T), atol=1e-12)
    # the field mixes z with x and y in the lab
    m = len(h.geometry.r)
    assert np.max(np.abs(out[:, :2*m, 2*m:])) > 1e-3


def test_response_follows_the_rotation_of_the_field():
    """The same field along z and along n: the response along n is the
    one along z, chi_nn = chi_zz, a number that knows nothing of the
    frame"""
    b = np.linalg.norm(FIELD)
    n = np.array(FIELD)/b
    kw = dict(q=Q, energies=ES, delta=0.05, nk=6)
    chiz = localspinchi.bare_spin_chi(honeycomb([0., 0., b]), **kw)[1]
    chin = localspinchi.bare_spin_chi(honeycomb(FIELD), **kw)[1]
    m = chiz.shape[-1]//3
    along = np.einsum("a,waibj,b->wij", n, chin.reshape(-1, 3, m, 3, m), n)
    assert np.max(np.abs(chiz[:, 2*m:, 2*m:])) > 0.05
    assert np.allclose(along, chiz[:, 2*m:, 2*m:], atol=1e-12)


def test_dressed_response_off_z_takes_the_vertex_of_the_lab():
    h = honeycomb(FIELD)
    n = len(h.geometry.r)
    V = anisotropic_vertex(n, np.random.default_rng(5))
    kw = dict(q=Q, energies=ES, delta=0.05, nk=6)
    ref = rpa.chi_ops_RPA(h, ops=spinchi._full_spin_operators(h), V=V,
                          **kw)[1]
    out = localspinchi.spin_chi(h, V=V, **kw)[1]
    assert np.max(np.abs(out - localspinchi.bare_spin_chi(h, **kw)[1])) > 0.01
    assert np.allclose(out, ref, atol=1e-11)
    ev = localspinchi.static_kernel_eigenvalues(h, V, Q, nk=6, delta=0.05)
    chi0 = general_route(h, Q, energies=[0.])[0]
    Vq = np.asarray(rpa.interaction_at_q(V, h, Q))
    full = np.linalg.eigvals(np.identity(3*n) - Vq@chi0)
    assert np.allclose(ev, full[np.argsort(full.real)], atol=1e-11)


@pytest.mark.parametrize("q", [Q, None])
def test_raising_and_local_response_off_z(q):
    """chi^{+-} of the lab for the sites asked for, and the local response
    bare and dressed, against the general route at every q of the mesh"""
    h = honeycomb(FIELD)
    n = len(h.geometry.r)
    qs = [q] if q is not None else h.geometry.get_kmesh(nk=3)
    full = np.mean([general_route(h, qi, nk=3) for qi in qs], axis=0)
    kw = dict(q=q, energies=ES, delta=0.05, nk=3)
    assert np.allclose(localspinchi.bare_spin_chi(h, **kw)[1], full,
                       atol=1e-12)
    # chi^{+-} of two sites, in the order asked for
    sites = [3, 0]
    pick = np.concatenate([np.array(sites) + a*n for a in range(3)])
    ref = localspinchi._raising_part(full[:, pick][:, :, pick])
    out = localspinchi.bare_raising_chi(h, sites=sites, **kw)[1]
    assert out.shape == (len(ES), 2, 2) and np.max(np.abs(ref)) > 0.05
    assert np.allclose(out, ref, atol=1e-12)
    # dressed: every q with its own vertex, in the lab
    V = anisotropic_vertex(n, np.random.default_rng(6))
    ref = np.mean([localspinchi._dyson(general_route(h, qi, nk=3),
            np.asarray(rpa.interaction_at_q(V, h, qi))) for qi in qs], axis=0)
    out = localspinchi.spin_chi(h, V=V, **kw)[1]
    assert np.allclose(out, ref, atol=1e-11)


def test_complex_spin_dependent_hopping_along_one_direction():
    """A Kane-Mele coupling and a field along z, with every spin rotated
    by hand to a direction that is not an axis: the spin part of the
    hoppings is complex and along one direction, and the collinear route
    in its frame gives the general one"""
    h = honeycomb([0., 0., 0.2])
    h.add_kane_mele(0.1)
    a, b = 0.7, 1.1 # a rotation by a about y, then by b about z
    sy, sz = localspinchi._SIGMA[1], localspinchi._SIGMA[2]
    W = (np.cos(b/2)*np.identity(2) - 1j*np.sin(b/2)*sz)@(
            np.cos(a/2)*np.identity(2) - 1j*np.sin(a/2)*sy)
    n = len(h.geometry.r)
    Wn = np.kron(np.identity(n), W)
    h.set_multihopping(MultiHopping({d: Wn@np.array(m)@Wn.conj().T
            for d, m in h.get_multihopping().get_dict().items()}))
    assert not localspinchi._conserves_sz(h)
    assert localspinchi._route(h) == localspinchi._COLLINEAR
    kw = dict(q=Q, energies=ES, delta=0.05, nk=6)
    out = localspinchi.bare_spin_chi(h, **kw)[1]
    ref = general_route(h, Q)
    assert np.max(np.abs(ref)) > 0.1
    assert np.allclose(out, ref, atol=1e-12)


def test_two_spin_directions_take_the_general_route():
    """A field that turns from site to site is not collinear: the general
    route, which is the one of chitk.chiAB"""
    h = honeycomb()
    h.add_zeeman(lambda r: [0.2, 0., 0.] if r[0] > 0. else [0., 0., 0.2])
    assert localspinchi._route(h) == localspinchi._SPINFUL
    kw = dict(q=Q, energies=ES, delta=0.05, nk=6)
    ref = rpa.chi_ops_RPA(h, ops=spinchi._full_spin_operators(h), V=None,
                          **kw)[1]
    out = localspinchi.bare_spin_chi(h, **kw)[1]
    assert np.allclose(out, ref, atol=1e-12)
    for q in (Q, None):
        full = localspinchi.bare_spin_chi(h, q=q, energies=ES, delta=0.05,
                                          nk=3)[1]
        out = localspinchi.bare_raising_chi(h, q=q, energies=ES, delta=0.05,
                                            nk=3)[1]
        assert np.allclose(out, localspinchi._raising_part(full), atol=1e-12)


def test_axis_search_holds_one_hopping_matrix_and_runs_once(monkeypatch):
    """The search of the spin direction reads the hopping matrices one at
    a time. So on top of what reading them costs, its memory is a few
    copies of one matrix and not of all of them, and a call of the bubble
    does it once. Rashba is the worst case: it is neither spin degenerate
    nor S_z conserving, so the search runs, and it finds no direction."""
    import tracemalloc
    def peak_of(f):
        tracemalloc.start()
        out = f()
        peak = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
        return out, peak
    g = geometry.honeycomb_lattice().get_supercell((6, 6))
    for (term, found) in (("rashba", False), ("field", True)):
        h = g.get_hamiltonian(has_spin=True)
        if term == "rashba": h.add_rashba(0.2)
        else: h.add_zeeman(FIELD)
        def read(): # what any check of the hoppings has to hold
            return [np.asarray(m)
                    for m in h.get_multihopping().get_dict().values()]
        hoppings, base = peak_of(read)
        one = hoppings[0].nbytes
        assert len(hoppings) >= 9 and base >= 9*one
        del hoppings
        n, peak = peak_of(lambda: localspinchi._spin_axis(h))
        assert (n is not None) == found
        assert peak < base + 5*one, (peak, base, one)
    # one search per call of the bubble
    calls = []
    axis = localspinchi._spin_axis
    def counting(*args, **kwargs):
        calls.append(1)
        return axis(*args, **kwargs)
    monkeypatch.setattr(localspinchi, "_spin_axis", counting)
    hs = honeycomb(FIELD)
    localspinchi.bare_spin_chi(hs, q=Q, energies=ES, delta=0.05, nk=2)
    localspinchi.spin_chi(hs, q=None, energies=ES, delta=0.05, nk=2)
    localspinchi.bare_raising_chi(hs, q=Q, energies=ES, delta=0.05, nk=2)
    assert len(calls) == 3, calls
