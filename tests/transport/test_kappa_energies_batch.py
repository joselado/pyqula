"""`HT.get_kappa(energies=[...])` is documented next to `temp=`, but at the
default temp=0 the array form used to collide with get_kappa's own
`energies=[energy]` and raise TypeError -- it worked only at temp!=0."""
import numpy as np
import pytest

from pyqula import geometry, heterostructures


def _junction():
    g = geometry.chain()
    h1 = g.get_hamiltonian()
    h1.setup_nambu_spinor()
    h2 = g.get_hamiltonian()
    h2.add_swave(0.1)
    return heterostructures.build(h1, h2)


def test_batched_kappa_matches_the_scalar_calls_it_replaces():
    ht = _junction()
    es = [0.0, 0.01, 0.02]
    batched = ht.get_kappa(energies=es)
    assert np.array(batched).shape == (3,)
    one_by_one = [ht.get_kappa(energy=e) for e in es]
    assert np.allclose(batched, one_by_one, atol=1e-6), (batched, one_by_one)


def test_the_two_spellings_are_mutually_exclusive():
    """The reference says "mutually exclusive with energy"; make it true."""
    ht = _junction()
    with pytest.raises(ValueError, match="not both"):
        ht.get_kappa(energy=0.01, energies=[0.0, 0.01])
