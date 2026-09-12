import numpy as np

from pyqula import specialgeometry
from pyqula.specialhopping import twisted_matrix


def _graphene_bn(mass):
    """Mismatched graphene/BN bilayer at the library's default (5,4)
    commensurate replica pair. The z<0 layer is the boron nitride: a
    sublattice imbalance of amplitude `mass` makes it a wide-gap insulator,
    while the z>0 graphene layer stays a semimetal."""
    g = specialgeometry.mismatched_lattice(5, 4)
    h = g.get_hamiltonian(is_sparse=True, has_spin=False, is_multicell=False,
                          mgenerator=twisted_matrix(ti=0.4, lambi=5.0))

    def fm(r):
        if r[2] < 0.0:
            return mass
        return 0.0

    h.add_sublattice_imbalance(fm)
    return h


def test_graphene_bn_low_energy_states_live_on_the_graphene_layer(tmp_path,
                                                                  monkeypatch):
    """The point of putting graphene on boron nitride is that only graphene
    contributes at low energy: the BN layer's sublattice imbalance pushes
    its states out to |E| ~ mass, so every state near the Fermi level sits
    on the graphene layer. The two layers are at z = +-1.5, so <zposition>
    reads off which layer a state belongs to.

    Replaces a recorded sum(e) constant. add_sublattice_imbalance puts +f on
    one sublattice and -f on the other, so it drops out of Tr H exactly, and
    sum(e) = sum_k Tr H(k) = 0 for mass = 3.5, for mass = 0.35 and for mass
    = 0."""
    monkeypatch.chdir(tmp_path)  # get_bands writes BANDS.OUT to cwd
    mass = 3.5
    h = _graphene_bn(mass)
    (k, e, z) = h.get_bands(nk=20, operator="zposition")
    e, z = np.array(e), np.array(z)
    assert np.min(np.abs(e)) < 0.1  # graphene keeps the semimetallic point
    assert np.all(z[np.abs(e) < 1.0] > 1.0)  # ... and all of it is graphene

    # the BN-polarised states are pushed out by their own sublattice gap
    bn = z < -1.0
    assert np.sum(bn) > 0
    assert np.min(np.abs(e[bn])) > 0.5 * mass
