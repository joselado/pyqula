import numpy as np

from pyqula import geometry
from pyqula.htk.extract import site_slice, site_dof, local_hamiltonian


def _finite_chain(n, has_spin, has_eh=False):
    g = geometry.chain().get_supercell(n)
    g.dimensionality = 0
    h = g.get_hamiltonian(has_spin=has_spin)
    if has_eh:
        h.add_swave(0.1)
    return h


def test_negative_index_matches_equivalent_positive_index():
    """site_slice/local_hamiltonian must support Python-style negative
    site indices, the same way the pre-refactor spinless branch got this
    for free from numpy's m[i,i] -- a bare slice(i*dof,(i+1)*dof) does not,
    since e.g. slice(-1,0) is empty rather than wrapping around."""
    for has_spin, has_eh in [(False, False), (True, False), (False, True), (True, True)]:
        h = _finite_chain(5, has_spin, has_eh)
        m = h.intra
        assert site_slice(h, -1) == site_slice(h, 4)
        assert np.allclose(local_hamiltonian(h, m, i=-1),
                            local_hamiltonian(h, m, i=4))


def test_out_of_range_negative_index_raises():
    h = _finite_chain(5, has_spin=False)
    try:
        site_slice(h, -100)
        assert False, "expected an IndexError"
    except IndexError:
        pass
