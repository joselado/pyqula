"""The superfluid weight in three dimensions, on models whose tensor is not
simply a multiple of the identity.

A cubic lattice gives D = d*I, so comparing it against the finite
difference only compares zeros on the off-diagonals, and an error confined
to the off-diagonal four-point stencil or to the off-diagonal entries of the
nd=3 paramagnetic/diamagnetic loops would pass.  The oracles here have
off-diagonals of order 0.1-0.3, or reduce exactly to a two-dimensional
result:

- a cubic lattice with anisotropic hoppings, rotated by an arbitrary
  rotation R, checked against R D0 R^T and against a hand-written
  single-band grand potential differentiated with its own stencil;
- two-dimensional layers stacked with no interlayer hopping, where D_zz and
  the xz/yz entries must vanish and c*D[:2,:2] must equal the 2D weight;
- the conventional/geometric decomposition and the closed-form conventional
  part on three-dimensional cells.
"""
import numpy as np
import pytest

from pyqula import geometry
from pyqula import superfluid as sf
from pyqula.sctk import superfluidweight as sw


# ---------------------------------------------------------------------------
# rotated anisotropic cubic lattice
# ---------------------------------------------------------------------------

_TS = [1.0, 0.6, 0.3]   # hoppings along the three cubic axes
_MU = -0.8
_DELTA = 0.4


def _rotation(axis, angle):
    axis = np.array(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    k = np.array([[0., -axis[2], axis[1]],
                  [axis[2], 0., -axis[0]],
                  [-axis[1], axis[0], 0.]])
    return np.identity(3) + np.sin(angle)*k + (1.-np.cos(angle))*k@k


_R = _rotation([1., 2., 3.], 0.7)


def _rotated_cubic(R):
    """Cubic lattice with lattice vectors R e_i and hopping _TS[i] along
    them.  Returns the Hamiltonian and the (rotated) lattice vectors."""
    g = geometry.cubic_lattice()
    g.a1 = R@np.array([1., 0., 0.])
    g.a2 = R@np.array([0., 1., 0.])
    g.a3 = R@np.array([0., 0., 1.])
    axes = [g.a1, g.a2, g.a3]

    def tij(r1, r2):
        dr = r2 - r1
        if abs(np.linalg.norm(dr)-1.) > 1e-4:
            return 0.
        for (a, t) in zip(axes, _TS):
            if abs(abs(dr.dot(a))-1.) < 1e-4:
                return t
        return 0.
    h = g.get_hamiltonian(tij=tij)
    h.add_onsite(_MU)
    h.add_swave(_DELTA)
    return h, axes


def _band(k, axes):
    """Normal-state dispersion of _rotated_cubic at Cartesian momenta k
    (rows), in pyqula's convention: hopping t enters as +t."""
    return sum(2.*t*np.cos(k@a) for (a, t) in zip(axes, _TS)) + _MU


def _brute_grand_potential(axes, Q, nk):
    """Grand potential per cell of the single-band s-wave BdG model with
    Cooper-pair momentum 2Q, written out by hand: each k gives the 2x2 block
    [[eps(k+Q), D], [D, -eps(k-Q)]] with eigenvalues a +- sqrt(s^2+D^2), and
    the T=0 grand potential is the sum of the negative ones.  The spin
    doubling of pyqula's 4x4 Nambu basis is compensated by its factor 1/2."""
    A = np.array(axes)
    B = 2.*np.pi*np.linalg.inv(A).T      # rows are the reciprocal vectors
    f = np.arange(nk)/nk
    frac = np.array(np.meshgrid(f, f, f, indexing="ij")).reshape(3, -1).T
    k = frac@B
    ep = _band(k+Q, axes)
    em = _band(k-Q, axes)
    a = (ep-em)/2.
    r = np.sqrt(((ep+em)/2.)**2 + _DELTA**2)
    E = np.concatenate([a+r, a-r])
    return np.sum(E[E < 0.])/len(k)


def _brute_superfluid_weight(axes, nk, dQ):
    om = lambda Q: _brute_grand_potential(axes, np.array(Q), nk)
    e = np.identity(3)
    o0 = om(np.zeros(3))
    D = np.zeros((3, 3))
    for i in range(3):
        D[i, i] = (om(dQ*e[i]) - 2.*o0 + om(-dQ*e[i]))/dQ**2
        for j in range(i+1, 3):
            D[i, j] = (om(dQ*(e[i]+e[j])) - om(dQ*(e[i]-e[j]))
                       - om(dQ*(e[j]-e[i])) + om(-dQ*(e[i]+e[j])))/(4.*dQ**2)
            D[j, i] = D[i, j]
    return D   # the cell volume is 1


@pytest.mark.parametrize("T", [0.0, 0.2])
def test_rotated_anisotropic_cubic_has_the_right_off_diagonals(T):
    """At T=0 only the diamagnetic term survives in a one-band s-wave model;
    at T=0.2 the thermal paramagnetic term is finite too, so both nd=3
    loops are exercised off the diagonal."""
    nk = 6
    h, axes = _rotated_cubic(_R)
    D = sf.superfluid_weight(h, nk=nk, T=T)
    # the test is only worth something if the off-diagonals are large
    off = D[~np.eye(3, dtype=bool)]
    assert np.max(np.abs(off)) > 0.1, D
    assert np.max(np.abs(D-D.T)) < 1e-12
    # the tensor rotates with the crystal
    h0, _ = _rotated_cubic(np.identity(3))
    D0 = sf.superfluid_weight(h0, nk=nk, T=T)
    assert np.max(np.abs(D0-np.diag(np.diag(D0)))) < 1e-12, D0
    assert np.max(np.abs(D - _R@D0@_R.T)) < 1e-10, (D, _R@D0@_R.T)
    # and agrees with pyqula's own finite difference of the grand potential
    Df = sf.superfluid_weight(h, nk=nk, T=T, mode="fd", dQ=1e-3)
    scale = np.max(np.abs(D))
    assert np.max(np.abs(D-Df))/scale < 1e-5, (D, Df)
    if T == 0.:
        # and with an independent, hand-written one: first check that the
        # hand-written model is the model pyqula builds
        Q = np.array([.01, .02, -.03])
        assert abs(sf.grand_potential(h, Q=Q, nk=nk)
                   - _brute_grand_potential(axes, Q, nk)) < 1e-12
        Db = _brute_superfluid_weight(axes, nk, 1e-3)
        assert np.max(np.abs(D-Db))/scale < 1e-5, (D, Db)


def test_rotated_cubic_decomposition_and_closed_form():
    """One band per spin: the decomposition is purely conventional and adds
    up to Kubo.  The closed-form conventional part (Liang et al. Eq. (21))
    only converges to it on meshes too dense for a fast 3D test (that
    convergence is checked in 2D in test_decomposition.py), but on any mesh
    it must rotate with the crystal, which fixes its off-diagonals."""
    nk = 6
    T = 0.2
    h, _ = _rotated_cubic(_R)
    h0, _ = _rotated_cubic(np.identity(3))
    out = sf.superfluid_weight(h, nk=nk, T=T, decompose=True)
    D = sf.superfluid_weight(h, nk=nk, T=T)
    assert np.max(np.abs(out["total"]-D)) < 1e-12
    assert np.max(np.abs(out["geometric"])) < 1e-12
    cc = sw.superfluid_weight_conventional_closed(h, nk=nk, T=T)
    cc0 = sw.superfluid_weight_conventional_closed(h0, nk=nk, T=T)
    assert np.max(np.abs(cc0-np.diag(np.diag(cc0)))) < 1e-12, cc0
    assert np.max(np.abs(cc[0, 1])) > 0.1, cc
    assert np.max(np.abs(cc - _R@cc0@_R.T)) < 1e-10, (cc, _R@cc0@_R.T)


# ---------------------------------------------------------------------------
# decoupled layers
# ---------------------------------------------------------------------------

_C = 3.0   # interlayer distance, larger than the first-neighbour distance


def _stack(g2):
    g = g2.copy()
    g.a3 = np.array([0., 0., _C])
    g.dimensionality = 3
    return g


def _square(g):
    h = g.get_hamiltonian()
    h.add_onsite(-0.7)
    h.add_swave(0.3)
    return h


def _honeycomb_rashba(g):
    h = g.get_hamiltonian()
    h.add_onsite(0.4)
    h.add_rashba(0.3)
    h.add_swave(0.35)
    return h


def _triangular_zeeman(g):
    h = g.get_hamiltonian()
    h.add_onsite(-1.0)
    h.add_zeeman([0., 0., 0.2])
    h.add_swave(0.6)
    return h


@pytest.mark.parametrize("lattice,build", [
    (geometry.square_lattice, _square),
    (geometry.honeycomb_lattice, _honeycomb_rashba),
    (geometry.triangular_lattice, _triangular_zeeman),
    ])
def test_decoupled_layers_reproduce_the_two_dimensional_weight(lattice,
                                                               build):
    nk = 6
    T = 0.05
    h2 = build(lattice())
    h3 = build(_stack(lattice()))
    D2 = sf.superfluid_weight(h2, nk=nk, T=T)
    D3 = sf.superfluid_weight(h3, nk=nk, T=T)
    # nothing disperses along z, so there is no stiffness there
    assert np.max(np.abs(D3[2, :])) < 1e-12, D3
    assert np.max(np.abs(D3[:, 2])) < 1e-12, D3
    # per unit volume, the in-plane block is the 2D weight over c
    assert np.max(np.abs(_C*D3[:2, :2]-D2)) < 1e-10, (_C*D3[:2, :2], D2)


def test_decoupled_layers_decomposition_and_closed_form():
    """The same reduction for the decomposition (on a honeycomb lattice, so
    the geometric part is finite) and for the closed-form conventional
    part.  nk=8 keeps the Dirac points, where the decomposition is ill
    defined, off the mesh."""
    nk = 8
    T = 0.05
    # uniform s-wave with time-reversal symmetry: the decomposition applies
    h2 = _square(geometry.honeycomb_lattice())
    h3 = _square(_stack(geometry.honeycomb_lattice()))
    out2 = sf.superfluid_weight(h2, nk=nk, T=T, decompose=True)
    out3 = sf.superfluid_weight(h3, nk=nk, T=T, decompose=True)
    assert np.max(np.abs(out3["total"]
                         - sf.superfluid_weight(h3, nk=nk, T=T))) < 1e-12
    assert np.max(np.abs(out2["geometric"])) > 1e-3   # two orbitals per cell
    out2["closed"] = sw.superfluid_weight_conventional_closed(h2, nk=nk, T=T)
    out3["closed"] = sw.superfluid_weight_conventional_closed(h3, nk=nk, T=T)
    for part in ["conventional", "geometric", "closed"]:
        assert np.max(np.abs(out3[part][2, :])) < 1e-12, part
        assert np.max(np.abs(out3[part][:, 2])) < 1e-12, part
        assert np.max(np.abs(_C*out3[part][:2, :2]-out2[part])) < 1e-10, part


# ---------------------------------------------------------------------------
# genuinely three-dimensional multi-orbital cells
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("lattice,mu", [
    (geometry.diamond_lattice, 0.5),
    (geometry.pyrochlore_lattice, -0.5),
    ])
def test_cubic_multiorbital_cells(lattice, mu):
    """Diamond and pyrochlore have cubic point groups, so the tensor is
    isotropic; the decomposition must add up to Kubo, and Kubo must match
    the finite difference with several orbitals per cell."""
    h = lattice().get_hamiltonian()
    h.add_onsite(mu)
    h.add_swave(0.4)
    nk = 3
    T = 0.05
    D = sf.superfluid_weight(h, nk=nk, T=T)
    scale = np.max(np.abs(D))
    assert scale > 1e-2, D
    Df = sf.superfluid_weight(h, nk=nk, T=T, mode="fd", dQ=3e-4)
    assert np.max(np.abs(D-Df))/scale < 1e-5, (D, Df)
    out = sf.superfluid_weight(h, nk=nk, T=T, decompose=True)
    assert np.max(np.abs(out["total"]-D))/scale < 1e-10
    D6 = sf.superfluid_weight(h, nk=6, T=T)
    assert np.max(np.abs(D6-D6[0, 0]*np.identity(3)))/D6[0, 0] < 1e-6, D6
