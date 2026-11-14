"""The winding number of an inverse Green's function over a window of
frequencies (topologytk/greenwinding.py): against h.get_chern() and the
plaquette sum of topologytk/plaquette.py, with the two things the
two-derivative integrand of topologytk/green.py cannot do, a constant
non-unitary rescaling of G^-1 and a frequency-dependent self-energy, and
the contribution of a zero of G."""
import numpy as np
import pytest

from pyqula import geometry
from pyqula.topologytk import greenwinding as gw
from pyqula.topologytk.plaquette import chern_from_states

I2 = np.identity(2)


def _haldane(t2):
    g = geometry.honeycomb_lattice()
    h = g.get_hamiltonian(has_spin=False)
    h.add_haldane(t2)
    h.add_onsite(0.05) # a small shift, so zero is not a symmetry point
    return h


def _band_edges(f, h, nk=24):
    """(min, max) of every band of the matrix function f over a k-mesh"""
    es = np.array([np.linalg.eigvalsh(f(k)) for k in h.geometry.get_kmesh(nk=nk)])
    return np.array([es.min(axis=0), es.max(axis=0)]).T


@pytest.mark.parametrize("t2", [0.2, -0.2])
def test_lower_band_matches_get_chern(t2):
    h = _haldane(t2)
    hk = h.get_hk_gen()
    c0 = round(h.get_chern(nk=20, write=False))
    assert abs(c0) == 1 # a Chern insulator, so the sign is tested
    e = _band_edges(hk, h)
    window = [e[0, 0] - 0.5, (e[0, 1] + e[1, 0])/2.] # the lower band
    ginv = lambda z, q: z*I2 - hk(q)
    err = [abs(gw.window_chern(ginv, window, nq=nq, nz=10) - c0)
           for nq in (6, 12)]
    assert err[1] < 1e-4 # 4.6e-6 measured at nq=12
    assert err[1] < err[0]/10. # 1.9e-3 at nq=6: not quantized on a mesh
    # the central difference converges as 1/nq^2
    errc = [abs(gw.window_chern(ginv, window, nq=nq, nz=10,
                                derivative="central") - c0) for nq in (8, 16)]
    assert errc[1] < 0.1
    assert 3. < errc[0]/errc[1] < 5.
    # both bands
    both = [e[0, 0] - 0.5, e[1, 1] + 0.5]
    assert abs(gw.window_chern(ginv, both, nq=8, nz=10)) < 1e-8


def test_half_contour_equals_full_contour():
    h = _haldane(0.2)
    hk = h.get_hk_gen()
    ginv = lambda z, q: z*I2 - hk(q)
    c1 = gw.window_chern(ginv, [-3.5, 0.05], nq=6, nz=8)
    c2 = gw.window_chern(ginv, [-3.5, 0.05], nq=6, nz=8, hermitian=False)
    assert abs(c1 - c2) < 1e-10


def _old_integrand_sum(ginv, window, nq=6, nz=8):
    """The contour integral of the two-derivative integrand of
    topologytk/green.py, summed over the mesh, in its own normalization"""
    from pyqula.topologytk.green import berry_green_generator
    zs, dzs = gw.contour(window, nz=nz)
    f = lambda e=0., k=None: np.linalg.inv(ginv(e, k))
    gI = lambda e=0., k=None: ginv(e, k)
    out = 0.
    for q in [np.array([i/nq, j/nq, 0.]) for i in range(nq) for j in range(nq)]:
        fint = berry_green_generator(f, k=q, dk=1e-4, gI=gI)
        out += 2.*np.sum([fint(z)*dz for (z, dz) in zip(zs, dzs)]).real
    return out


def test_constant_rescaling_leaves_the_winding_unchanged():
    """G^-1 = D^dag (z - H) D with a non-unitary D: the integrand is the same
    at every q, while the integrand of topologytk/green.py changes"""
    h = _haldane(0.2)
    hk = h.get_hk_gen()
    rng = np.random.default_rng(1)
    D = rng.normal(size=(2, 2)) + 1j*rng.normal(size=(2, 2))
    g1 = lambda z, q: z*I2 - hk(q)
    g2 = lambda z, q: np.conj(D.T)@(z*I2 - hk(q))@D
    window = [-3.5, 0.05]
    d1 = gw.window_chern_density(g1, window, nq=8, nz=10)
    d2 = gw.window_chern_density(g2, window, nq=8, nz=10)
    assert abs(np.mean(d1) - 1.) < 1e-3
    assert np.max(np.abs(d1 - d2)) < 1e-9 # 3e-12 measured
    o1 = _old_integrand_sum(g1, window)
    o2 = _old_integrand_sum(g2, window)
    assert abs(o2/o1 - 1.) > 0.05 # it would not change if dzG^-1 were 1


def _bath():
    """A Haldane model with one flat bath level e_b per site, coupled by V,
    as the enlarged Hamiltonian and as the self-energy V^2/(z - e_b)"""
    h = _haldane(0.2)
    hk = h.get_hk_gen()
    eb, V = 0.1, 0.6
    hbig = lambda q: np.block([[hk(q), V*I2], [V*I2, eb*I2]])
    ginv = lambda z, q: z*I2 - hk(q) - V**2/(z - eb)*I2
    return h, hk, hbig, ginv, eb


def test_self_energy_window_gives_the_bands_of_the_enlarged_hamiltonian():
    h, hk, hbig, ginv, eb = _bath()
    cs = [chern_from_states(lambda q, b=b: np.linalg.eigh(hbig(q))[1].T[b:b+1],
                            nq=12) for b in range(4)]
    assert np.allclose(np.round(cs), [1, -1, 1, -1]) # not a trivial case
    e = _band_edges(hbig, h)
    ends = [e[0, 0] - 0.5] + [(e[i, 1] + e[i+1, 0])/2. for i in range(3)]
    for (a, b, has_eb) in [(0, 1, False), (1, 2, False), (2, 3, True),
                           (1, 3, True)]:
        window = [ends[a], ends[b]]
        assert (window[0] < eb < window[1]) == has_eb
        c = gw.window_chern(ginv, window, nq=12, nz=12)
        # the zero of G at e_b has the identity as residue, a trivial
        # bundle, so it adds nothing; 1.7e-5 at most measured
        assert abs(c - round(sum(cs[a:b]))) < 1e-3


def test_zero_of_the_green_function_counts_with_opposite_sign():
    """Sigma = Delta^2 P(q)/(z - e_b), with P(q) the projector on the lower
    Haldane band: a window that encloses the pole of G^-1 at e_b and no
    zero of G^-1 gives minus the Chern number of that band"""
    h = _haldane(0.2)
    hk = h.get_hk_gen()
    c0 = round(h.get_chern(nk=20, write=False))
    eb, delta = 0.1, 1.0
    def plow(q):
        e, u = np.linalg.eigh(hk(q))
        return np.outer(u[:, 0], np.conj(u[:, 0]))
    ginv = lambda z, q: z*I2 - hk(q) - delta**2*plow(q)/(z - eb)
    # zeros of the lower sector, (z - E1)(z - e_b) = delta^2, on both sides
    e1 = _band_edges(hk, h)[0]
    em = (e1 + eb)/2. - np.sqrt(((e1 - eb)/2.)**2 + delta**2)
    ep = (e1 + eb)/2. + np.sqrt(((e1 - eb)/2.)**2 + delta**2)
    assert np.max(em) < eb < np.min(ep)
    c = gw.window_chern(ginv, [np.max(em) + 0.3, eb + 0.12], nq=12, nz=12)
    assert abs(c + c0) < 1e-3 # -1.00006 measured: only the zero of G
    c = gw.window_chern(ginv, [np.max(em) + 0.3, np.max(ep) + 0.1], nq=12,
                        nz=12)
    assert abs(c) < 1e-2 # the zero cancels the band above it


def test_errors():
    h = _haldane(0.2)
    hk = h.get_hk_gen()
    ginv = lambda z, q: z*I2 - hk(q)
    with pytest.raises(ValueError, match="window"):
        gw.window_chern(ginv, [0.05, -3.5])
    with pytest.raises(ValueError, match="central, spectral"):
        gw.window_chern(ginv, [-3.5, 0.05], derivative="fft")
    not_hermitian = lambda z, q: z*I2 - hk(q) + 0.3j*I2
    with pytest.raises(ValueError, match="hermitian=False"):
        gw.window_chern(not_hermitian, [-3.5, 0.05], nq=4)
    with pytest.raises(ValueError, match="square matrix"):
        gw.window_chern(lambda z, q: np.ones((2, 3)), [-3.5, 0.05], nq=4)


def test_contour_integrates_a_pole():
    """The upper half returned by contour() and its mirror image close a
    counterclockwise loop: 2 pi i for a pole inside, 0 for one outside"""
    for height in (None, 0.3):
        zs, dzs = gw.contour([-1., 2.], nz=64, height=height)
        top = 1.5 if height is None else height
        assert np.isclose(np.max(zs.imag), top) and np.min(zs.imag) > 0.
        zf = np.concatenate([zs, np.conj(zs)])
        df = np.concatenate([dzs, -np.conj(dzs)])
        for (w0, inside) in [(0.4 + 0.0j, True), (1.7 - 0.05j, True),
                             (2.5 + 0.0j, False), (0.4 + 2.0j, False)]:
            total = np.sum(df/(zf - w0))
            assert abs(total - 2j*np.pi*inside) < 1e-8


def test_shift_moves_the_density_with_the_mesh():
    """A shift by one mesh step along one direction is a roll of the density
    along that axis, on an odd mesh, and a generic shift keeps the mean"""
    h = _haldane(0.2)
    hk = h.get_hk_gen()
    ginv = lambda z, q: z*I2 - hk(q)
    window, nq = [-3.5, 0.05], 7
    d0 = gw.window_chern_density(ginv, window, nq=nq, nz=8)
    d1 = gw.window_chern_density(ginv, window, nq=nq, nz=8,
                                 shift=(1./nq, 0.))
    d2 = gw.window_chern_density(ginv, window, nq=nq, nz=8,
                                 shift=(0., 2./nq))
    assert np.max(np.abs(d0 - np.roll(d0, -1, axis=0))) > 0.1 # not uniform
    assert np.max(np.abs(np.roll(d0, -1, axis=0)
                         - np.roll(d0, -1, axis=1))) > 0.1 # nor along axis 1
    assert np.max(np.abs(d1 - np.roll(d0, -1, axis=0))) < 1e-9
    assert np.max(np.abs(d2 - np.roll(d0, -2, axis=1))) < 1e-9
    d3 = gw.window_chern_density(ginv, window, nq=nq, nz=8,
                                 shift=(0.03, 0.07))
    assert np.max(np.abs(d3 - d0)) > 1e-2 # the shift is used
    assert abs(np.mean(d3) - 1.) < 5e-3 and abs(np.mean(d0) - 1.) < 5e-3


def test_analytic_z_derivative_and_height():
    """dginv_dz given in closed form agrees with the finite difference at
    every q, for a self-energy that depends on z, and the density does not
    depend on the height of the contour, which is where ginv is evaluated"""
    h, hk, hbig, ginv, eb = _bath()
    dginv = lambda z, q: (1. + 0.6**2/(z - eb)**2)*I2
    window = [-3.6, -0.9]
    assert not window[0] < eb < window[1]
    d0 = gw.window_chern_density(ginv, window, nq=6, nz=10)
    d1 = gw.window_chern_density(ginv, window, nq=6, nz=10, dginv_dz=dginv)
    assert np.max(np.abs(d0 - d1)) < 1e-7
    d2 = gw.window_chern_density(ginv, window, nq=6, nz=10,
                                 dginv_dz=lambda z, q: I2) # not its slope
    assert np.max(np.abs(d0 - d2)) > 1e-3
    seen = []
    def recording(z, q):
        seen.append(z)
        return ginv(z, q)
    out = []
    for height in (0.8, 2.0):
        del seen[:]
        out.append(gw.window_chern_density(recording, window, nq=6, nz=24,
                                           height=height))
        assert np.isclose(np.max(np.imag(seen)), height)
    assert np.max(np.abs(out[0] - out[1])) < 1e-7


def test_enlarged_hamiltonian_with_batched_frequencies():
    """The 4x4 form z - H_big, called with batched=True, gives the density
    of the 2x2 form with the self-energy only in the sum over the mesh, and
    the plaquette integers band by band"""
    h, hk, hbig, ginv, eb = _bath()
    I4 = np.identity(4)
    calls = []
    def gbig(zs, q):
        calls.append(len(np.atleast_1d(zs)))
        return np.asarray(zs)[:, None, None]*I4 - hbig(q)
    e = _band_edges(hbig, h)
    ends = [e[0, 0] - 0.5] + [(e[i, 1] + e[i+1, 0])/2. for i in range(3)]
    ends = ends + [e[3, 1] + 0.5]
    cs = [chern_from_states(lambda q, b=b: np.linalg.eigh(hbig(q))[1].T[b:b+1],
                            nq=12) for b in range(4)]
    for (a, b) in [(0, 1), (1, 2), (3, 4), (1, 3)]:
        c = gw.window_chern(gbig, [ends[a], ends[b]], nq=12, nz=12,
                            batched=True)
        assert abs(c - round(sum(cs[a:b]))) < 1e-3
    assert max(calls) >= 36 # the 36 nodes of the contour went in one call


def test_zero_with_a_residue_that_does_not_commute_with_h():
    """Sigma = Delta^2 P'(q)/(z - e_b), with P' the projector on the lower
    band of the Haldane model with the opposite flux, which does not commute
    with H. The poles of G are the eigenvalues of the 3x3 matrix
    [[H, Delta u'], [Delta u'^dag, e_b]], and the residue of each is the
    projector on the first two components of its eigenvector, which do not
    depend on the phase of u'. Every window gives the plaquette Chern
    numbers of those residues, minus the one of P' when e_b is inside"""
    h = _haldane(0.2)
    hk = h.get_hk_gen()
    hkp = _haldane(-0.2).get_hk_gen()
    eb, delta = 0.05, 0.5
    up = lambda q: np.linalg.eigh(hkp(q))[1][:, 0]
    def h3(q):
        u = up(q)[:, None]
        return np.block([[hk(q), delta*u],
                         [delta*np.conj(u.T), np.array([[eb]])]])
    def ginv(z, q):
        u = up(q)
        return z*I2 - hk(q) - delta**2*np.outer(u, np.conj(u))/(z - eb)
    q0 = np.array([0.1, 0.2, 0.])
    p0 = np.outer(up(q0), np.conj(up(q0)))
    assert np.max(np.abs(hk(q0)@p0 - p0@hk(q0))) > 0.1 # they do not commute
    cp = chern_from_states(lambda q: up(q)[None], nq=12)
    ca = [chern_from_states(lambda q, b=b: np.linalg.eigh(h3(q))[1][:2, b][None],
                            nq=16) for b in range(3)]
    assert abs(round(cp)) == 1 and abs(round(ca[1])) == 1 # nothing trivial
    e = _band_edges(h3, h)
    assert e[1, 0] - e[0, 1] > 0.3 and e[2, 0] - e[1, 1] > 0.3 # two gaps
    ends = [e[0, 0] - 0.5, (e[0, 1] + e[1, 0])/2., (e[1, 1] + e[2, 0])/2.,
            e[2, 1] + 0.5]
    for (a, b) in [(0, 1), (1, 2), (2, 3), (0, 2)]:
        window = [ends[a], ends[b]]
        expected = sum(ca[a:b]) - (cp if window[0] < eb < window[1] else 0.)
        c = gw.window_chern(ginv, window, nq=12, nz=12)
        assert abs(c - round(expected)) < 2e-2 # 4e-3 at most measured
    # the middle window holds one pole and the zero, which cancel
    assert ends[1] < eb < ends[2] and round(ca[1]) == round(cp)


def test_reflection_of_the_mesh_with_content_at_the_nyquist_mode():
    """G^-1(-q) has the density of G^-1(q) at the reflected mesh point, also
    when the mesh is so coarse that G^-1 has a harmonic at its Nyquist
    frequency, whose derivative the spectral rule sets to zero"""
    h = _haldane(0.2)
    hk = h.get_hk_gen()
    def extra(q): # a second harmonic, the Nyquist mode of a mesh of 4
        a, b = 0.3*np.cos(4.*np.pi*q[0]), 0.2*np.exp(4j*np.pi*q[1])
        return np.array([[a, b], [np.conj(b), -a]])
    g1 = lambda z, q: z*I2 - hk(q) - extra(q)
    g2 = lambda z, q: g1(z, -np.array(q))
    window, nq = [-4.5, 0.05], 4
    d1 = gw.window_chern_density(g1, window, nq=nq, nz=8)
    d2 = gw.window_chern_density(g2, window, nq=nq, nz=8)
    reflected = np.roll(d1[::-1, ::-1], (1, 1), axis=(0, 1)) # i -> -i mod nq
    assert np.max(np.abs(d1)) > 0.1
    assert np.max(np.abs(d2 - reflected)) < 1e-9
