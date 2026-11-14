import numpy as np
import pytest

from pyqula import geometry
from pyqula.kondolattice import KondoLatticeHamiltonian
from pyqula.kondotk import spinonspectrum

# The local response of the moments of a Kondo lattice, q=None, takes the
# states at k+q from one diagonalization of the k-mesh, since k+q is a
# point of the same mesh. It must be the average of the response at each q
# of the mesh, which diagonalizes H(k+q) directly, bare and dressed with
# the residual interaction, in two dimensions, for a spin-degenerate
# mean field, in a field along z and in a field along another direction.


@pytest.fixture(scope="module")
def hsquare():
    g = geometry.square_lattice().get_supercell((2, 1))
    h = KondoLatticeHamiltonian(g.get_hamiltonian(has_spin=True))
    n = len(h._kondo_pairs)
    return h.get_mean_field_hamiltonian(J=4.0, filling=0.5, nk=12,
            mf=([2.0+0j]*n, [0.0]*n), maxerror=1e-7, maxite=2000)


def Uq(q):
    """A residual interaction that depends on q, so that every q of the
    mesh must be dressed with its own one"""
    return -0.5*np.cos(2*np.pi*q[0]) - 0.3*np.cos(2*np.pi*q[1])


def turning(r):
    """A field that turns from site to site, so that the Hamiltonian has
    two spin directions and takes the general route of the bubble. A
    uniform field in any direction takes the collinear one, in its own
    spin frame."""
    return [0.15, 0., 0.] if r[0] > 0. else [0., 0., 0.2]


FIELDS = [None, [0., 0., 0.2], [0.1, 0.05, 0.2], turning]
FIELD_IDS = ["none", "z", "tilted", "turning"]


@pytest.mark.parametrize("field", FIELDS, ids=FIELD_IDS)
@pytest.mark.parametrize("U", [None, -0.5, Uq],
                         ids=["bare", "number", "function"])
def test_local_moment_response_is_the_average_over_the_mesh(hsquare, field, U):
    h = hsquare.copy()
    if field is not None: h.add_zeeman(field)
    if field is turning:
        from pyqula.chitk import localspinchi
        assert localspinchi._route(h) == localspinchi._SPINFUL
    kw = dict(energies=np.linspace(0., 5., 6), nk=3, delta=0.1, T=0.05, U=U)
    e, out = h.get_moment_spinchi(**kw)
    ref = np.mean([h.get_moment_spinchi(q=q, **kw)[1]
                   for q in h.geometry.get_kmesh(nk=3)], axis=0)
    assert out.shape == (6, 2, 2) and np.max(np.abs(ref)) > 0.01
    assert np.allclose(out, ref, atol=1e-12)


@pytest.mark.parametrize("field", FIELDS, ids=FIELD_IDS)
def test_bare_moment_response_is_the_combination_of_the_spin_blocks(hsquare,
                                                                     field):
    """Without the residual interaction only S^+ is contracted; the result
    is chi^xx + chi^yy - i chi^xy + i chi^yx of the full bubble of the f
    sites, with S^+ on the first index"""
    from pyqula.chitk import localspinchi
    h = hsquare.copy()
    if field is not None: h.add_zeeman(field)
    fsites = [fi for (ci, fi) in h._kondo_pairs]
    kw = dict(q=[0.25, 0.5, 0.], energies=np.linspace(0., 5., 6), nk=3,
              delta=0.1, T=0.05)
    out = h.get_moment_spinchi(**kw)[1]
    chi = localspinchi.bare_spin_chi(h, sites=fsites, **kw)[1]
    x, y = slice(0, 2), slice(2, 4)
    ref = chi[:, x, x] + chi[:, y, y] - 1j*chi[:, x, y] + 1j*chi[:, y, x]
    assert np.max(np.abs(ref)) > 0.01
    assert np.allclose(out, ref, atol=1e-12)


def test_spectral_components_of_a_degenerate_hamiltonian_from_one_spin():
    """For a Hamiltonian that is the same for the two spins the occupied
    and the empty spectral weight come from the bands of one spin, each
    one twice. It is the first component of the general route."""
    g = geometry.triangular_lattice().get_supercell((3, 2))
    h = g.get_hamiltonian(has_spin=True)
    h.add_onsite(0.2)
    sites = [0, 3, 5]
    kw = dict(nk=4, T=0.05, de=0.01)
    n0, occ, emp = spinonspectrum.local_spectral_components(h, sites,
            degenerate=True, **kw)
    m0, occ4, emp4 = spinonspectrum.local_spectral_components(h, sites,
            degenerate=False, **kw)
    assert n0 == m0 and occ.shape[1] == 1 and occ4.shape[1] == 4
    assert np.allclose(occ[:, 0], occ4[:, 0], atol=1e-12)
    assert np.allclose(emp[:, 0], emp4[:, 0], atol=1e-12)
    assert np.allclose(np.sum(occ + emp, axis=2), 2., atol=1e-10) # two spins
    assert np.max(np.abs(occ4[:, 1:])) < 1e-12 # no spin density
