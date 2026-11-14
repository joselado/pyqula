"""The vertex weights of the linear simplex integration (dmtk.simplexdm)
in closed form. The T=0 weights of a simplex cut by the Fermi level are
written out as products of the edge fractions, with no array and no
determinant. They are the weights of the sub-simplex decomposition with
barycentric vertices and determinant volumes that the module used before,
reproduced here in numpy, to rounding, also for degenerate vertex energies
and for a level exactly on a vertex. At T>0 the weights and the density of
states of the Bloechl term are read in one pass over the quadrature
nodes, and they agree with the charge kernel of the Fermi search, which
reads the same nodes."""
import numpy as np
import pytest

from pyqula.dmtk import simplexdm


def _decomposition(e, mu):
    """the T=0 weights from the sub-simplex decomposition: the corner of
    the lowest vertex, its complement, and in 3D the prism of three
    tetrahedra, each with volume |det| of its barycentric vertices"""
    p = len(e)
    out = np.zeros(p)
    if mu <= e[0]: return out
    if mu >= e[-1]: return np.full(p, 1./p)
    def vert(i):
        b = np.zeros(p)
        b[i] = 1.
        return b
    def cut(i, j):
        a = (mu-e[i])/(e[j]-e[i])
        b = np.zeros(p)
        b[i], b[j] = 1.-a, a
        return b
    def sub(c, *bs):
        m = np.array(bs)
        out[:] += c*abs(np.linalg.det(m))*m.sum(axis=0)/p
    if p == 2:
        sub(1., vert(0), cut(0, 1))
    elif p == 3:
        if mu < e[1]: sub(1., vert(0), cut(0, 1), cut(0, 2))
        else:
            out[:] = 1./p
            sub(-1., vert(2), cut(0, 2), cut(1, 2))
    else:
        if mu < e[1]: sub(1., vert(0), cut(0, 1), cut(0, 2), cut(0, 3))
        elif mu < e[2]:
            v0, v1 = vert(0), vert(1)
            p02, p03, p12, p13 = cut(0, 2), cut(0, 3), cut(1, 2), cut(1, 3)
            sub(1., v0, p02, p03, v1)
            sub(1., p02, p03, v1, p12)
            sub(1., p03, v1, p12, p13)
        else:
            out[:] = 1./p
            sub(-1., vert(3), cut(0, 3), cut(1, 3), cut(2, 3))
    return out


def _cases(p, rng):
    """sorted vertex energies: generic, with every pattern of ties, and
    with two energies far closer than the others"""
    out = [np.sort(rng.normal(size=p)) for _ in range(40)]
    for _ in range(10):
        x = np.sort(rng.normal(size=p))
        for i in range(p-1): # one tie
            y = x.copy(); y[i+1] = y[i]; out.append(y)
        y = x.copy(); y[:] = y[0]; out.append(y) # flat
        y = x.copy(); y[1] = y[0]+1e-9; out.append(np.sort(y)) # near tie
    if p == 4:
        for _ in range(10): # two pairs of ties
            a, b = np.sort(rng.normal(size=2))
            out.append(np.array([a, a, b, b]))
    return out


@pytest.mark.parametrize("p", [2, 3, 4])
def test_closed_form_matches_the_decomposition(p):
    rng = np.random.default_rng(p)
    out = np.zeros(p)
    for e in _cases(p, rng):
        mus = list(rng.uniform(e[0]-0.2, e[-1]+0.2, size=20))+list(e)
        for mu in mus: # also exactly on every vertex
            s = simplexdm._step_weights(e, p, mu, out)
            ref = _decomposition(e, mu)
            assert np.max(np.abs(out-ref)) < 1e-14
            assert abs(s-np.sum(ref)) < 1e-14
            assert abs(s-simplexdm._step_fraction(e, p, mu)) < 1e-14


@pytest.mark.parametrize("p", [2, 3, 4])
@pytest.mark.parametrize("T", [1e-7, 1e-3, 3e-2])
def test_finite_temperature_weights_and_dos_match_the_charge_kernel(p, T):
    """The occupied fraction (the sum of the weights) and the density of
    states that _weights_T reads in one pass are the ones of the charge
    kernel of the Fermi search, which reads the same knots and nodes"""
    rng = np.random.default_rng(10+p)
    w = np.zeros(p); tmp = np.zeros(p)
    kn = np.zeros(p+3+2*len(simplexdm._tbreaks))
    args = (simplexdm._gx, simplexdm._gw, simplexdm._tbreaks)
    for e in _cases(p, rng):
        for shift in list(rng.uniform(e[0]-0.1, e[-1]+0.1, size=6))+list(e):
            x = np.ascontiguousarray(e-shift) # measured from mu
            s, dos = simplexdm._weights_T(x, p, T, *args, w, tmp, kn)
            c, d = simplexdm._charge_dos_T(x, p, T, *args, kn)
            assert abs(s-np.sum(w)) < 1e-15
            assert abs(s-c) < 1e-13
            if x[-1] > x[0]: # a flat simplex has no Bloechl term
                assert abs(dos-d) <= 1e-13*max(1., abs(d))


@pytest.mark.parametrize("dim,nk", [(2, 12), (3, 6)])
def test_mesh_weights_at_zero_temperature_match_the_decomposition(dim, nk):
    """On a whole mesh, the weights of every state summed from the
    decomposition of every simplex, against simplex_weights"""
    from pyqula.dostk.simplexdos import simplices
    rng = np.random.default_rng(dim)
    nb = 2
    es = rng.normal(size=(nk**dim, nb))
    sim, frac = simplices(dim, nk)
    ref = np.zeros(es.shape)
    for s in sim:
        for n in range(nb):
            e = es[s, n]
            order = np.argsort(e, kind="stable")
            ref[s[order], n] += _decomposition(e[order], 0.)
    w = simplexdm.simplex_weights(dim, nk, es, mu=0., T=0., bloechl=False)
    assert np.max(np.abs(w-ref*frac)) < 1e-14


@pytest.mark.parametrize("p", [2, 3, 4])
@pytest.mark.parametrize("T", [1e-3, 3e-2, 0.3])
def test_finite_temperature_weights_are_the_smeared_decomposition(p, T):
    """Each vertex weight at T>0 is the T=0 weight convolved with -f'. In
    the variable phi = f(x) this is w_j(T) = int_0^1 w_j(x(phi), T=0) dphi,
    with x(phi) = T log((1 - phi)/phi) the level measured from mu. Here the
    integral is an adaptive quadrature of the sub-simplex decomposition,
    split at the vertex energies, against the eight nodes per piece of the
    module, whose error is below 3e-9 of the simplex volume. This pins
    every vertex weight, not only their sum"""
    from scipy.integrate import quad_vec
    from scipy.special import expit
    rng = np.random.default_rng(20+p)
    w = np.zeros(p); tmp = np.zeros(p)
    kn = np.zeros(p+3+2*len(simplexdm._tbreaks))
    args = (simplexdm._gx, simplexdm._gw, simplexdm._tbreaks)
    cases = [0.15*np.sort(rng.normal(size=p)) for _ in range(3)]
    tie = cases[0].copy(); tie[1] = tie[0]; cases.append(tie)
    for e in cases:
        for mu in (e[0]+0.3*(e[-1]-e[0]), e[0]+0.8*(e[-1]-e[0])):
            x = np.ascontiguousarray(e-mu) # measured from mu
            simplexdm._weights_T(x, p, T, *args, w, tmp, kn)
            def smeared(phi):
                return _decomposition(x, T*(np.log1p(-phi)-np.log(phi)))
            pts = [f for f in expit(-x/T) if 1e-12 < f < 1.-1e-12]
            ref = quad_vec(smeared, 0., 1., points=pts, epsabs=1e-12,
                    epsrel=0., limit=2000)[0]
            assert np.max(np.abs(ref)) > 1e-3 # a simplex with weight
            assert np.max(np.abs(w-ref)) < 1e-8, (e, mu)


@pytest.mark.parametrize("dim,nk", [(2, 8), (3, 4)])
@pytest.mark.parametrize("T", [0., 2e-2])
def test_mesh_weights_with_the_bloechl_term_assemble_the_simplices(dim, nk,
        T):
    """On a whole mesh with the Bloechl term, the weight of a state is the
    sum over the simplices that share its k-point of the weight of each
    one plus c_dim D (sum_i e_i - (dim+1) e_j), with D the density of
    states of the simplex, here taken from the charge kernel of the Fermi
    search (at T=0, the one of the step)"""
    from pyqula.dostk.simplexdos import simplices
    rng = np.random.default_rng(30+dim)
    nb, p = 2, dim+1
    es = 0.3*rng.normal(size=(nk**dim, nb))
    sim, frac = simplices(dim, nk)
    c = simplexdm.bloechl_coefficient(dim)
    w = np.zeros(p); tmp = np.zeros(p)
    kn = np.zeros(p+3+2*len(simplexdm._tbreaks))
    args = (simplexdm._gx, simplexdm._gw, simplexdm._tbreaks)
    ref = np.zeros(es.shape)
    for s in sim:
        for n in range(nb):
            order = np.argsort(es[s, n], kind="stable")
            x = np.ascontiguousarray(es[s, n][order])
            if T > 0.:
                simplexdm._weights_T(x, p, T, *args, w, tmp, kn)
                D = simplexdm._charge_dos_T(x, p, T, *args, kn)[1]
            else:
                simplexdm._step_weights(x, p, 0., w)
                D = simplexdm._step_dos(x, p, 0.)
            ref[s[order], n] += w+c*D*(np.sum(x)-p*x)
    got = simplexdm.simplex_weights(dim, nk, es, mu=0., T=T, bloechl=True)
    assert np.max(np.abs(got-ref*frac)) < 1e-14
    plain = simplexdm.simplex_weights(dim, nk, es, mu=0., T=T, bloechl=False)
    assert np.max(np.abs(got-plain)) > 1e-6 # the term is there
