import numpy as np

from pyqula import specialgeometry
from pyqula.specialhopping import twisted_matrix


def test_graphene_bn_mismatched_lattice_bands_match_reference(tmp_path, monkeypatch):
    """Regression check for a mismatched graphene/BN bilayer (specialgeometry
    .mismatched_lattice at the library's default (5,4) commensurate replica
    pair) with a sublattice imbalance: the band energy sum must match the
    value recorded from a known-good run."""
    monkeypatch.chdir(tmp_path)  # get_bands writes BANDS.OUT to cwd
    g = specialgeometry.mismatched_lattice(5, 4)
    h = g.get_hamiltonian(is_sparse=True, has_spin=False, is_multicell=False,
                           mgenerator=twisted_matrix(ti=0.4, lambi=5.0))

    def fm(r):
        if r[2] < 0.0:
            return 3.5
        return 0.0

    h.add_sublattice_imbalance(fm)
    (k, e) = h.get_bands(nk=20)
    assert np.isclose(np.sum(e), 2.842170943040401e-14, atol=1e-6)
