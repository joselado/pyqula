import numpy as np
import pytest

from pyqula import specialhamiltonian
from pyqula import specialgeometry
from pyqula import specialhopping


@pytest.mark.slow
def test_tbg_two_construction_modes_match_reference(tmp_path, monkeypatch):
    """Regression check for the two ways of building a twisted bilayer
    graphene Hamiltonian (specialhamiltonian.twisted_bilayer_graphene vs.
    specialgeometry.twisted_bilayer + get_hamiltonian with a twisted_matrix
    generator), at the smallest commensurate moire index (n=1). The two
    modes use different defaults (e.g. spin, hopping cutoff) so they are
    NOT expected to agree with each other -- each band energy sum must
    independently match the value recorded from a known-good run. Marked
    slow: building two twisted-hopping matrices keeps this a few seconds
    even at the smallest moire cell."""
    monkeypatch.chdir(tmp_path)  # get_bands writes BANDS.OUT to cwd
    n = 1
    ti = 0.3

    h1 = specialhamiltonian.twisted_bilayer_graphene(n=n, ti=ti)
    h1.set_filling(0.5, nk=2)
    (k1, e1) = h1.get_bands(num_bands=20)

    g = specialgeometry.twisted_bilayer(n)
    h2 = g.get_hamiltonian(mgenerator=specialhopping.twisted_matrix(ti=ti))
    h2.set_filling(0.5, nk=2)
    (k2, e2) = h2.get_bands(num_bands=20)

    assert np.isclose(np.sum(e1), -192.46480225751023, atol=1e-6)
    assert np.isclose(np.sum(e2), 133.582765059469, atol=1e-6)
