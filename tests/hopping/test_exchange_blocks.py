import numpy as np
import pytest

from pyqula import geometry


def _onsite_spin_term(g, ms):
    """Oracle: sum over sites of |i><i| x (m_i . sigma), built densely"""
    pauli = [np.array([[0., 1.], [1., 0.]]), np.array([[0., -1j], [1j, 0.]]),
            np.array([[1., 0.], [0., -1.]])]
    out = np.zeros((2*len(g.r), 2*len(g.r)), dtype=np.complex128)
    for i, m in enumerate(ms):
        out[2*i:2*i+2, 2*i:2*i+2] = sum(m[a]*pauli[a] for a in range(3))
    return out


@pytest.mark.parametrize("sparse", [False, True])
def test_exchange_is_the_local_field_on_each_spin_block(sparse):
    """add_exchange, add_zeeman and add_antiferromagnetism build their
    block-diagonal term from coordinates in one pass, linear in the number
    of sites, where they used to assemble an n x n list of 2 x 2 blocks
    (90 s and a 2 GB peak at 10,000 sites); the term is still m_i . sigma
    on the spin block of each site i"""
    g = geometry.honeycomb_lattice().get_supercell(2)
    h0 = g.get_hamiltonian(has_spin=True, is_sparse=sparse)
    rng = np.random.default_rng(3)
    ms = rng.random((len(g.r), 3)) - 0.5
    def term(h): # what the call added to the Hamiltonian
        m = h.intra - h0.intra
        return m.toarray() if hasattr(m, "toarray") else np.asarray(m)
    h = h0.copy(); h.add_exchange(ms)
    assert np.max(np.abs(term(h) - _onsite_spin_term(g, ms))) < 1e-14
    h = h0.copy(); h.add_zeeman(lambda r: [r[0], 0.2, r[1]])
    expected = _onsite_spin_term(g, [[r[0], 0.2, r[1]] for r in g.r])
    assert np.max(np.abs(term(h) - expected)) < 1e-14
    h = h0.copy(); h.add_antiferromagnetism(0.3)
    expected = _onsite_spin_term(g, [[0., 0., 0.3*s] for s in g.sublattice])
    assert np.max(np.abs(term(h) - expected)) < 1e-14


def test_exchange_with_the_wrong_number_of_components_is_refused():
    """A field with two components per site used to stop on an IndexError
    inside the loop over sites; reshaped into a block-diagonal build it
    could be read as a different field, so it is refused by name"""
    from pyqula.magnetism import exchange_matrix
    with pytest.raises(ValueError, match="three components"):
        exchange_matrix(np.ones((3, 2)))
