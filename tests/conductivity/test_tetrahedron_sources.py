"""The source kernel of the simplex optical conductivity on tetrahedra.

The source kernel builds the sum over simplices once for all frequencies
(vertex sources, Taylor series of close clusters, and the edges and faces
whose vertex form would cancel too much evaluated per frequency, each
point source summed with the others of its energy bin far from it),
and the direct kernel evaluates every simplex at every frequency with the
divided differences of simplexdos._dd. They are the same integral of the
same interpolant, so they agree to rounding. The invariant checked here is
that agreement, on single tetrahedra with generic, nearly confluent and
exactly confluent nodes (two, three and four equal nodes, and two pairs),
at broadenings from 1e-2 to 1e-6, and on the sum over a 3D mesh.
"""
import numpy as np
import pytest

from pyqula import geometry
from pyqula import multihopping
from pyqula.conductivitytk import kubo
from pyqula.conductivitytk import kubosimplex


def _tetrahedra(pattern, eps, n, rng):
    """n tetrahedra with energies in [0.2, 3] and spread up to 1. pattern
    groups the four nodes, e.g. (3,1) is three nodes within eps of each
    other and one more; pattern None is four generic nodes"""
    out = []
    for i in range(n):
        c = rng.uniform(0.2, 3.)
        s = 10**rng.uniform(-2, 0)
        e = c+s*rng.uniform(-0.5, 0.5, 4)
        if pattern is not None:
            ee = []
            for g, size in enumerate(pattern):
                base = e[g]
                for q in range(size):
                    ee.append(base+(eps*rng.uniform(-1, 1) if q > 0 else 0.))
            e = np.array(ee)
        rng.shuffle(e)
        out.append(e)
    return np.array(out)


def _one_by_one(E, ws, delta, kernel):
    """K! psi_K[e] of every tetrahedron of E (rows) at every frequency, as
    one Cartesian component per tetrahedron"""
    nt = E.shape[0]
    sim = np.arange(4*nt).reshape(nt, 4).astype(np.int64)
    W = np.zeros((4*nt, 1, nt), dtype=np.complex128)
    for t in range(nt): W[4*t:4*t+4, 0, t] = 1.
    return kubosimplex._kernels[kernel](sim, E.reshape(-1, 1), W,
            np.arange(nt, dtype=np.int64), np.array([True]), ws, delta)


def _exact(e, w, delta):
    """K! psi_K[e] at z = w + i delta with mpmath at 150 digits, exactly
    equal nodes split by 1e-30"""
    import mpmath
    mpmath.mp.dps = 150
    z = mpmath.mpc(w, delta)
    es = [mpmath.mpf(float(x)) for x in e]
    for i in range(len(es)):
        for j in range(i):
            if es[i] == es[j]: es[i] += mpmath.mpf(10)**(-30)*(i+1)
    tot = 0
    for i in range(4):
        d = 1
        for j in range(4):
            if j != i: d *= es[i]-es[j]
        x = es[i]-z
        tot += -x**2/2*(mpmath.log(x)-mpmath.mpf(3)/2)/d
    return complex(6*tot)


PATTERNS = [None, (4,), (3, 1), (2, 2), (2, 1, 1), (1, 1, 2)]


def _compare(E, ws, delta):
    """Both kernels on the tetrahedra E at the frequencies ws, to 1e-11 of
    the largest value of each tetrahedron on ws. Where they differ by more,
    the source kernel is checked against mpmath to 1e-11 instead. Returns
    the number of such entries."""
    src = _one_by_one(E, ws, delta, "sources")
    ref = _one_by_one(E, ws, delta, "simplices")
    peak = np.max(np.abs(ref), axis=0)
    bad = np.abs(src-ref) > 1e-11*peak[None, :]
    for iw, t in zip(*np.nonzero(bad)):
        try:
            exact = _exact(E[t], ws[iw], delta)
        except ImportError: # without mpmath, the floor of both
            assert abs(src[iw, t]-ref[iw, t]) < 1e-10*peak[t]
            continue
        assert abs(src[iw, t]-exact) < 1e-11*peak[t], (E[t], ws[iw])
    return int(np.sum(bad))


@pytest.mark.parametrize("delta", [1e-2, 1e-3, 1e-6])
def test_sources_agree_with_the_direct_kernel_on_random_tetrahedra(delta):
    """Every tetrahedron, at every frequency, to 1e-11 of its own peak
    over the frequencies, which include its node energies so that the peak
    is sampled. The direct kernel takes the exact divided difference when
    the spread is above 0.05 |c - z|, and near resonance for a spread of
    about delta that loses up to 3e-11 of the peak by itself; where the
    two differ by more than 1e-11, the source kernel is checked against
    mpmath instead (and the difference is then the direct kernel's)."""
    rng = np.random.default_rng(7)
    nflag = 0
    for pattern in PATTERNS:
        epss = [0.] if pattern is None else [0., 1e-14, 1e-9, 1e-5,
                0.3*delta, delta, 3*delta]
        for eps in epss:
            E = _tetrahedra(pattern, eps, 12, rng)
            ws = np.sort(np.concatenate([np.linspace(0., 4., 21), E.ravel(),
                    E.mean(axis=1)]))
            nflag += _compare(E, ws, delta)
    assert nflag < 100 # the floor of the direct kernel is rare


@pytest.mark.parametrize("delta", [1e-2, 1e-6])
def test_sources_agree_with_the_direct_kernel_far_from_the_window(delta):
    """Tetrahedra at negative energies, as the reverse terms at -dE, with
    the frequencies on [0, 4]: most of them go to the moments of their
    energy bin, the others to the vertex form or kept whole. To 1e-11 of
    the largest value of each tetrahedron on the window, with the same
    mpmath check where the direct kernel is the one that is off"""
    rng = np.random.default_rng(3)
    for pattern in PATTERNS:
        for eps in [0., 1e-9, 0.3*delta, 3*delta]:
            E = -_tetrahedra(pattern, eps, 12, rng)
            _compare(E, np.linspace(0., 4., 41), delta)


def test_exactly_confluent_tetrahedra_are_exact():
    """Four equal nodes give the resonance 1/(z - e) itself, and three or
    two equal nodes the closed forms of the confluent divided difference,
    which the source kernel takes from the first term of its series"""
    delta = 1e-3
    ws = np.linspace(0., 3., 61)
    z = ws+1j*delta
    a, b = 0.7, 1.9
    E = np.array([[a, a, a, a], [a, a, a, b]])
    s = _one_by_one(E, ws, delta, "sources")
    assert np.allclose(s[:, 0], 1./(z-a), rtol=1e-14, atol=0.)
    # 3! psi_3[a,a,a,b] = 6 (psi_3[a,a,b] - psi_1(a)/2)/(b - a), with
    # psi_3[a,a,b] = (psi_3[a,b] - psi_2(a))/(b - a)
    xa, xb = a-z, b-z
    p3 = lambda x: -x**2/2*(np.log(x)-1.5)
    p2 = lambda x: -(x*np.log(x)-x)
    p1 = lambda x: -np.log(x)
    d1 = (p3(xb)-p3(xa))/(b-a)
    d2 = (d1-p2(xa))/(b-a)
    d3 = (d2-p1(xa)/2.)/(b-a)
    assert np.allclose(s[:, 1], 6*d3, rtol=1e-11, atol=0.)


def test_a_tetrahedron_with_close_nodes_is_split():
    """A tetrahedron with a close pair or two, of spread a few delta, would
    lose too much in its vertex form far from resonance. It is not kept
    whole: its Newton table is split, so that only its close edges and
    faces are evaluated per frequency (types _direct+1 and _direct+2) and
    the rest goes to the vertex form, and the sum is still the one of the
    direct kernel to 1e-12 of its peak"""
    delta = 1e-3
    ws = np.linspace(0., 4., 81)
    for e in [[0.5, 1.2, 1.202, 2.1], [0.5, 1.2, 1.203, 1.205],
              [1., 1.004, 1.5, 1.503]]:
        E = np.array([e])
        sim = np.arange(4).reshape(1, 4).astype(np.int64)
        W = np.ones((4, 1, 1), dtype=np.complex128)
        src = kubosimplex.simplex_sources(sim, E.reshape(-1, 1), W,
                np.zeros(1, dtype=np.int64), np.array([True]), delta,
                (ws.min(), ws.max()))
        tp = src[3]
        d = kubosimplex._direct
        assert np.any(tp >= d) # some nodes are evaluated per frequency
        assert np.all((tp < d) | (tp == d+1) | (tp == d+2)) # but not all
        s = _one_by_one(E, ws, delta, "sources")[:, 0]
        ref = _one_by_one(E, ws, delta, "simplices")[:, 0]
        assert np.max(np.abs(s-ref)) < 1e-12*np.max(np.abs(ref))


def _dense(m):
    return np.array(m.todense()) if hasattr(m, "todense") else np.array(m)


def _layered_haldane(tm=0.05):
    """Haldane layers stacked along z with a kz dependent mass, as in
    test_optical_3d.py"""
    h2 = geometry.honeycomb_lattice().get_hamiltonian(has_spin=False)
    h2.add_haldane(0.1)
    h2.add_sublattice_imbalance(0.2)
    h = h2.copy()
    h.geometry.a3 = np.array([0., 0., 2.])
    h.geometry.dimensionality = 3
    hd = {(R[0], R[1], 0): _dense(m) for R, m in h2.get_hopping_dict().items()}
    M = np.diag([tm, -tm]).astype(np.complex128)
    hd[(0, 0, 1)] = M
    hd[(0, 0, -1)] = M
    h.dimensionality = 3
    h.set_multihopping(multihopping.MultiHopping(hd))
    return h


def test_sources_agree_with_the_direct_kernel_on_a_3d_mesh():
    """The whole sum over the tetrahedra of a 3D mesh, both kernels on the
    same pair fields, every Cartesian component, to 1e-11 of the largest
    entry, at a broadening above and below the mesh spacing; the default
    kernel in 3D is the source kernel"""
    h = _layered_haldane()
    nk = 8
    ks = kubo._kmesh(h, nk)
    es, vs, cellvol, scale = kubo._bands_and_velocities(h, ks)
    pairs, dE, W, comps, trans = kubosimplex.pair_fields(es, vs, 0.01,
            1e-6*scale, True, True)
    pairs, dE, W = kubosimplex._prune(pairs, dE, W)
    ws = np.linspace(0., 4., 41)
    for delta in [3e-2, 1e-3]:
        args = (3, nk, pairs, dE, W, comps, trans, ws, delta, cellvol)
        s0 = kubosimplex._sigma_from_fields(*args)
        s1 = kubosimplex._sigma_from_fields(*args, kernel="sources")
        s2 = kubosimplex._sigma_from_fields(*args, kernel="simplices")
        assert np.max(np.abs(s0-s1)) == 0.
        scale = np.max(np.abs(s2))
        assert np.max(np.abs(s2[:, 2, 2])) > 1e-3*scale # z is compared
        assert np.max(np.abs(s1-s2)) < 1e-11*scale


def test_the_source_sum_stays_at_the_rounding_of_the_direct_kernel():
    """At nk = 24 the vertex sources cancel by a factor of about 5e3 at
    every frequency, so a plain running sum over them lost 2.2e-13 of the
    largest entry against the direct kernel (7e-13 at nk = 48), a loss
    that grew with nk. The sums are now compensated, and the two kernels
    agree to 6e-14, which includes the 3e-14 of the direct kernel itself"""
    h = _layered_haldane()
    nk = 24
    ks = kubo._kmesh(h, nk)
    es, vs, cellvol, scale = kubo._bands_and_velocities(h, ks)
    pairs, dE, W, comps, trans = kubosimplex.pair_fields(es, vs, 1e-3,
            1e-6*scale, False, True)
    pairs, dE, W = kubosimplex._prune(pairs, dE, W)
    args = (3, nk, pairs, dE, W, comps, trans, np.linspace(0.1, 4., 40),
            1e-2, cellvol)
    s1 = kubosimplex._sigma_from_fields(*args, kernel="sources")
    s2 = kubosimplex._sigma_from_fields(*args, kernel="simplices")
    assert np.max(np.abs(s1-s2)) < 1.2e-13*np.max(np.abs(s2))


def test_the_source_sum_does_not_depend_on_the_number_of_threads():
    """The sum over the sources is split in tasks whose partial sums are
    added in a fixed order. The number of tasks was set from the number of
    threads, so at a few frequencies one and three threads gave results
    that differ at 2e-13; now it is a fixed count and the two agree to
    the last bit"""
    import numba
    if numba.config.NUMBA_NUM_THREADS < 2:
        pytest.skip("needs at least two numba threads")
    h = _layered_haldane()
    nk = 24
    ks = kubo._kmesh(h, nk)
    es, vs, cellvol, scale = kubo._bands_and_velocities(h, ks)
    pairs, dE, W, comps, trans = kubosimplex.pair_fields(es, vs, 0.01,
            1e-6*scale, True, True)
    pairs, dE, W = kubosimplex._prune(pairs, dE, W)
    args = (3, nk, pairs, dE, W, comps, trans, np.linspace(0., 4., 3),
            1e-2, cellvol)
    n0 = numba.get_num_threads()
    try:
        out = []
        for n in (1, min(3, numba.config.NUMBA_NUM_THREADS)):
            numba.set_num_threads(n)
            out.append(kubosimplex._sigma_from_fields(*args,
                    kernel="sources"))
    finally:
        numba.set_num_threads(n0)
    assert np.array_equal(out[0], out[1])
