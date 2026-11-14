"""The finite-temperature weights of the linear simplex integration
(dmtk.simplexdm) when no vertex energy is within the cutoff of the Fermi
function, 40 T of the chemical potential. The T=0 weight is then one
polynomial of degree dim+1 over the whole window of -f', so its
convolution with -f' is the Sommerfeld series, which ends at the fourth
moment,
  w(T) = w + (pi^2 T^2/6) w'' + (7 pi^4 T^4/360) w'''',
and the three-node Gauss rule of the weight -f' gives it to rounding. The
pieces of Gauss-Legendre quadrature in f(x), about 200 evaluations, gave it
with an error that grows as T^2, 1e-11 of the simplex volume at T = 1/100
of the energy spread of the simplex. The weights, their sum in the charge
kernel of the Fermi search and the density of states take the same rule on
the same simplices, so the density matrix holds the charge of the
search."""
import numpy as np
import pytest

from pyqula.dmtk import simplexdm

CUT = simplexdm._tcut


def _clear_simplices(p, rng, ratios):
    """sorted vertex energies around mu = 0 and a temperature, with no
    vertex within the cutoff of the Fermi function"""
    out = []
    for ratio in ratios:
        found = 0
        while found < 20:
            e = np.sort(rng.uniform(-0.5, 0.5, size=p))
            T = ratio*(e[-1]-e[0])
            if e[0] < 0. < e[-1] and np.min(np.abs(e)) > CUT*T:
                out.append((e, T))
                found += 1
    return out


def _step(e, x):
    out = np.zeros(len(e))
    simplexdm._step_weights(e, len(e), x, out)
    return out


def _sommerfeld(values, width):
    """int (-f'(x)) g(x) dx for a polynomial g of degree 4, given at the
    five points width*T*(-1, -1/2, 0, 1/2, 1) of x: its coefficients times the
    moments 1, pi^2 T^2/3 and 7 pi^4 T^4/15 of -f'"""
    xs = np.linspace(-1., 1., 5)*width
    c = np.polynomial.polynomial.polyfit(xs, values, 4) # x in units of T
    return c[0]+c[2]*np.pi**2/3.+c[4]*7.*np.pi**4/15.


def _weights(e, T):
    p = len(e)
    out, tmp = np.zeros(p), np.zeros(p)
    kn = np.zeros(p+3+2*len(simplexdm._tbreaks))
    frac, dos = simplexdm._weights_T(e, p, T, simplexdm._gx, simplexdm._gw,
                                     simplexdm._tbreaks, out, tmp, kn)
    return out, frac, dos


def _charge(e, T):
    p = len(e)
    kn = np.zeros(p+3+2*len(simplexdm._tbreaks))
    return simplexdm._charge_dos_T(e, p, T, simplexdm._gx, simplexdm._gw,
                                   simplexdm._tbreaks, kn)


@pytest.mark.parametrize("p", [2, 3, 4])
def test_a_clear_window_gives_the_sommerfeld_series(p):
    """the weights, the occupied fraction and the density of states, to
    rounding, at temperatures up to the largest one that leaves the window
    clear, 1/80 of the energy spread"""
    rng = np.random.default_rng(10+p)
    cases = _clear_simplices(p, rng, [1e-6, 1./300., 1./150., 1./100.])
    xs = np.linspace(-1., 1., 5)*30.
    for e, T in cases:
        w, frac, dos = _weights(e, T)
        ref = _sommerfeld(np.array([_step(e, x*T) for x in xs]), 30.)
        assert np.max(np.abs(w-ref)) < 1e-13
        assert abs(frac-np.sum(ref)) < 1e-13
        dref = _sommerfeld(np.array([simplexdm._step_dos(e, p, x*T)
                                     for x in xs]), 30.)
        assert abs(dos-dref) < 1e-10*abs(dref)
        # the charge kernel of the Fermi search, on the same rule
        c, d = _charge(e, T)
        assert abs(c-frac) < 1e-14
        assert abs(d-dos) < 1e-13*abs(dos)


@pytest.mark.parametrize("p", [2, 3, 4])
def test_the_two_rules_meet_at_the_edge_of_the_window(p):
    """with the nearest vertex just inside the window the pieces are used,
    and just outside it the three nodes. The two differ by the error of
    the pieces there, so the charge has no jump that the Fermi search, at
    1e-12 per state, could see: a simplex is 1/nk^dim of the zone or less"""
    rng = np.random.default_rng(20+p)
    done = 0
    while done < 100:
        e = np.sort(rng.uniform(-0.5, 0.5, size=p))
        if not e[0] < 0. < e[-1]: continue
        T = np.min(np.abs(e))/CUT # the nearest vertex on the edge
        inside, fi, di = _weights(e, T*(1.+1e-9))
        outside, fo, do = _weights(e, T*(1.-1e-9))
        assert np.max(np.abs(inside-outside)) < 1e-10
        assert abs(fi-fo) < 1e-10
        assert abs(di-do) < 1e-6*abs(do)
        done += 1


def _rules(sim, es, mu, T):
    """how many (simplex, band) pairs that the window of the Fermi function
    cuts take the three nodes, and how many take the pieces"""
    e = np.sort(es[sim], axis=1)-mu # (simplex, vertex, band)
    L = CUT*T
    cut = (e[:, 0, :] < L) & (e[:, -1, :] > -L)
    clear = np.all(np.abs(e) >= L, axis=1)
    return int(np.sum(cut & clear)), int(np.sum(cut & ~clear))


@pytest.mark.parametrize("dim,nk", [(1, 40), (2, 24), (3, 10)])
def test_the_mesh_weights_hold_the_charge_of_the_search(dim, nk):
    """On a mesh the weights and the charge kernel of the Fermi search sort
    the vertices of each simplex on their own, so that they take the same
    rule on every simplex is checked here, simplex by simplex: the weights
    of a simplex sum to the charge that the charge kernel gives for it, to
    rounding, where the two rules differ by 1e-13 to 1e-11 of its volume.
    This is done at T = 1e-7, where every cut simplex takes the three
    nodes, at T = 2e-2, where every one takes the pieces, and at a
    temperature chosen from the mesh so that both rules are in use, about
    half of the simplices that hold the Fermi level on each. The level
    that the search finds holds the filling at each of them. In 1D the
    occupied fraction of a segment is linear in mu, so both rules give the
    same charge there and only the 2D and 3D cases can see a mismatch
    (8e-12 and 2e-13 when the charge kernel is put on the pieces alone)"""
    from pyqula.dostk.simplexdos import uniform_mesh, simplices
    ks = uniform_mesh(dim, nk)[:, :dim]
    es = np.zeros((len(ks), 2))
    es[:, 0] = -np.sum(np.cos(2.*np.pi*ks), axis=1)
    es[:, 1] = 0.7+0.5*np.sum(np.cos(2.*np.pi*ks+0.3), axis=1)
    sim, frac = simplices(dim, nk)
    # the temperature with both rules: 40 T is the median distance from
    # the Fermi level to the nearest vertex of the simplices that hold it
    mu0 = simplexdm.fermi_from_energies(dim, nk, es, 0.37, T=1e-7)
    e = np.sort(es[sim], axis=1)-mu0
    holds = (e[:, 0, :] < 0.) & (e[:, -1, :] > 0.)
    Tmix = np.median(np.min(np.abs(e), axis=1)[holds])/CUT
    used = {}
    for T in [1e-7, Tmix, 2e-2]:
        mu = simplexdm.fermi_from_energies(dim, nk, es, 0.37, T=T)
        used[T] = _rules(sim, es, mu, T)
        w = simplexdm.simplex_weights(dim, nk, es, mu=mu, T=T)
        c = simplexdm.simplex_charge(dim, nk, es, mu=mu, T=T)
        assert abs(np.sum(w)-c) < 1e-12
        assert abs(c-0.37*2) < 1e-11
        out = np.zeros((len(sim), es.shape[1], sim.shape[1]))
        simplexdm._simplex_weights_jit(sim, es, mu, T, 0., simplexdm._gx,
                simplexdm._gw, simplexdm._tbreaks, out)
        per = np.sum(out, axis=(1, 2)) # per simplex, in units of its volume
        for it in range(len(sim)):
            ci = simplexdm._charge_dos_jit(sim[it:it+1], es, mu, T,
                    simplexdm._gx, simplexdm._gw, simplexdm._tbreaks)[0]
            assert abs(per[it]-ci) < 2e-14
    assert used[1e-7][0] > 0 and used[1e-7][1] == 0 # the three nodes only
    assert used[Tmix][0] > 0 and used[Tmix][1] > 0 # both rules
    assert used[2e-2][0] == 0 and used[2e-2][1] > 0 # the pieces only
