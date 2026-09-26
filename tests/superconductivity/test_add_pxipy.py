import numpy as np
import pytest

from pyqula import geometry
from pyqula import superconductivity
from pyqula.multihopping import MultiHopping

# superconductivity.add_pxipy (and its alias add_pwave) built a d-vector per
# bond and handed that length-3 list to add_pairing, which indexes a 2x2
# pairing matrix, so every call died in a TypeError; with the default
# r1=None it died earlier, on len(None). It is now the chiral px+ipy
# pairing with the d-vector along z, the electron-hole part of what
# add_pairing(mode="chiral_pwave",d=[0,0,1]) builds.


def _by_hand(g, delta):
    """A Nambu Hamiltonian whose pairing is assembled from add_pxipy, with
    the dagger of each block added the way add_pairing does it"""
    h = g.get_hamiltonian()
    h.setup_nambu_spinor()
    dd = dict(h.get_multihopping().get_dict())
    def add(R, m):
        R = tuple(int(x) for x in R)
        dd[R] = dd.get(R, 0.*m) + m
    r = g.r
    m = superconductivity.add_pxipy(delta, is_sparse=True, r1=r, r2=r)
    add((0, 0, 0), m + m.getH())
    for R in g.neighbor_directions():
        if R.dot(R) < 1e-4: continue
        m = superconductivity.add_pxipy(delta, is_sparse=True, r1=r,
                                        r2=g.replicas(d=R))
        add(R, m)
        add(-np.array(R), m.getH())
    h.set_multihopping(MultiHopping(dd))
    return h


@pytest.mark.parametrize("lattice", [geometry.chain,
                                     geometry.triangular_lattice,
                                     geometry.honeycomb_lattice])
def test_add_pxipy_is_the_chiral_pwave_of_add_pairing(lattice):
    g = lattice()
    h1 = _by_hand(g, 0.3)
    h2 = g.get_hamiltonian()
    h2.add_pairing(delta=0.3, mode="chiral_pwave", d=[0., 0., 1.])
    hk1 = h1.get_hk_gen()
    hk2 = h2.get_hk_gen()
    for k in np.random.default_rng(0).random((4, 3)):
        assert np.allclose(hk1(k), hk2(k), atol=1e-12)


def test_add_pxipy_on_a_chain_is_the_px_pairing():
    g = geometry.chain()
    h1 = _by_hand(g, lambda r: 0.3) # a callable amplitude is accepted
    h2 = g.get_hamiltonian()
    h2.add_pairing(delta=0.3, mode="pwave", d=[0., 0., 1.])
    for k in [0.1, 0.37]:
        assert np.allclose(h1.get_hk_gen()([k, 0., 0.]),
                           h2.get_hk_gen()([k, 0., 0.]), atol=1e-12)


def test_add_pwave_is_an_alias_and_dense_is_the_default():
    g = geometry.triangular_lattice()
    r = g.r
    r2 = g.replicas(d=np.array([1, 0, 0]))
    ms = superconductivity.add_pxipy(0.3, is_sparse=True, r1=r, r2=r2)
    md = superconductivity.add_pwave(0.3, r1=r, r2=r2)
    assert isinstance(md, np.ndarray)
    assert np.allclose(md, ms.toarray())
    assert np.max(np.abs(md)) > 0.1


def test_add_pxipy_needs_the_positions():
    with pytest.raises(ValueError, match="r1 and r2"):
        superconductivity.add_pxipy(0.3)
