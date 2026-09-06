import numpy as np
import pytest

from pyqula import geometry

# The d-vector routines read the pairing out of a 4x4 spin x electron-hole
# block per site (sctk.extract.extract_triplet_pairing, nr = dim//4). On
# anything else that arithmetic still runs and returns an array of the
# wrong length instead of raising: zeros of half the length for a spinful
# non-Nambu Hamiltonian, an empty array (and nan means) for a spinless
# Nambu one.


def _chain(nsites=4, **kwargs):
    g = geometry.chain().supercell(nsites)
    return g, g.get_hamiltonian(**kwargs)


def test_spinful_non_nambu_is_refused():
    g, h = _chain()
    assert h.has_spin and not h.has_eh
    with pytest.raises(ValueError):
        h.get_dvector_non_unitarity(nk=2)
    with pytest.raises(ValueError):
        h.get_average_dvector(nk=2)


def test_spinless_nambu_is_refused():
    g, h = _chain(has_spin=False)
    h.setup_nambu_spinor()
    assert (not h.has_spin) and h.has_eh
    with pytest.raises(ValueError):
        h.get_dvector_non_unitarity(nk=2)
    with pytest.raises(ValueError):
        h.get_average_dvector(nk=2)


def test_spinful_nambu_still_works():
    nsites = 6
    g, h = _chain(nsites)
    h.add_zeeman([0., 0., 0.3])
    h.add_swave(0.0)
    h.add_pairing(mode="pwave", delta=0.3)
    out = np.array(h.get_dvector_non_unitarity(nk=4))
    # one non-unitarity vector per site, not per 4x4 Nambu block
    assert out.shape == (nsites, 3)
    assert np.array(h.get_average_dvector(nk=4)).shape == (3,)
