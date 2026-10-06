import numpy as np
from scipy.sparse import issparse

from pyqula import geometry, meanfield
from pyqula.multihopping import MultiHopping


def test_random_guess_of_a_sparse_hamiltonian_stays_sparse():
    """For a sparse Hamiltonian the random guess is drawn on its sparsity
    pattern and on the onsite block of every site, rather than as a dense
    random matrix, which made the first mean-field Hamiltonian of a large
    system dense; it is still Hermitian, and it still seeds every onsite
    channel (charge, the three magnetizations and, with Nambu, pairing)"""
    g = geometry.square_lattice().get_supercell(4)
    g.dimensionality = 0
    for nambu in [False, True]:
        h = g.get_hamiltonian(has_spin=True, is_sparse=True)
        if nambu: h.setup_nambu_spinor()
        mf = meanfield.guess(h, mode="random")
        assert MultiHopping(mf).is_hermitian()
        m = mf[(0, 0, 0)]
        assert issparse(m)
        b = 4 if nambu else 2 # orbitals per site
        pattern = np.zeros(h.intra.shape, dtype=bool) # stored entries of h
        c = h.intra.tocoo() ; pattern[c.row, c.col] = True
        for i in range(len(g.r)):
            pattern[b*i:b*i+b, b*i:b*i+b] = True
        mm = m.toarray()
        assert np.all(mm[~pattern] == 0.) # nothing outside the pattern
        for i in range(len(g.r)): # every onsite entry seeded
            assert np.all(np.abs(mm[b*i:b*i+b, b*i:b*i+b]) > 0.)
