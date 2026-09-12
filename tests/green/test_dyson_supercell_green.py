import numpy as np

from pyqula import geometry, algebra, embedding
from pyqula.dyson import dyson, dyson1d, dyson1d_hkgen
from pyqula.greentk.selfenergy import bloch_selfenergy

# pyqula.dyson builds the Green's function of an nsuper-cell supercell of a
# periodic Hamiltonian by Brillouin-zone summation. There are two 1D
# routes: dyson1d, fed the explicit intra/inter matrices of a
# first-neighbour-cell Hamiltonian, and dyson1d_hkgen, fed the Bloch
# generator, which is what dyson() falls back to when the Hamiltonian
# couples cells further apart (so get_no_multicell refuses it). The second
# one used to call the jitted kernel with 7 of its 11 arguments, passing
# the generator where the intracell matrix belongs, so every call raised
# TypeError -- reachable from Embedding(h).get_gf(nsuper=2) on any 1D
# Hamiltonian with beyond-nearest-cell hopping.

ENERGY, DELTA, NK = 0.3, 0.1, 30


def test_generator_and_matrix_paths_agree_on_a_nearest_neighbour_chain():
    """On a Hamiltonian both routes can handle, they must return the same
    supercell Green's function, entry by entry. The cell here has two
    inequivalent sites, so the intercell hopping is not symmetric and the
    (0,1) and (1,0) blocks of the answer differ -- a k-convention mismatch
    between the two routes would swap them rather than cancel out."""
    g = geometry.chain().get_supercell(2)
    h = g.get_hamiltonian().get_no_multicell().get_dense()
    intra = algebra.todense(h.intra)
    inter = algebra.todense(h.inter)
    assert np.max(np.abs(inter-algebra.dagger(inter))) > 0.1 # not symmetric
    hkgen = h.get_hk_gen()
    ez = ENERGY + 1j*DELTA
    for nsuper in [1, 2, 3]:
        gm = dyson1d(intra, inter, nsuper, NK, ez)
        gg = dyson1d_hkgen(hkgen, nsuper, NK, ez)
        assert np.array(gg).shape == np.array(gm).shape
        assert np.max(np.abs(np.array(gg)-np.array(gm))) < 1e-10


def long_range_chain():
    """A 1D Hamiltonian coupling cells two apart, so get_no_multicell
    refuses it and dyson() has to use the generator route."""
    return geometry.chain().get_hamiltonian(tij=[1.0, 0.3])


def test_long_range_1d_diagonal_blocks_are_the_bulk_green_function():
    """Every diagonal block of the supercell Green's function is the
    R=0 bulk Green's function of the unit cell, which bloch_selfenergy's
    mode='full' computes independently as a plain uniform-mesh average of
    inv(E+i*delta-H(k)) over the same mesh. The blocks must also all be
    equal to each other: the system is translationally invariant, and
    which cell of the supercell you sit in cannot matter."""
    h = long_range_chain()
    n = h.intra.shape[0]
    gref, _ = bloch_selfenergy(h, energy=ENERGY, delta=DELTA, nk=NK,
                               mode="full")
    gref = np.array(algebra.todense(gref))
    for nsuper in [2, 3]:
        gs = np.array(dyson(h, [nsuper, 1], NK, ENERGY+1j*DELTA))
        assert gs.shape == (n*nsuper, n*nsuper)
        for i in range(nsuper):
            blk = gs[n*i:n*(i+1), n*i:n*(i+1)]
            assert np.max(np.abs(blk-gref)) < 1e-10


def test_embedding_get_gf_works_beyond_nearest_neighbour_cells():
    """The public path the missing arguments crashed. The embedded
    Green's function of a *defect-free* embedding is the clean one, so it
    must reproduce the same bulk Green's function on its diagonal blocks
    and give a positive local density of states."""
    h = long_range_chain()
    n = h.intra.shape[0]
    eb = embedding.Embedding(h)
    gs = np.array(algebra.todense(eb.get_gf(nsuper=2, energy=ENERGY,
                                            delta=DELTA, nk=NK)))
    assert gs.shape == (2*n, 2*n)
    gref, _ = bloch_selfenergy(h, energy=ENERGY, delta=DELTA, nk=NK,
                               mode="full")
    gref = np.array(algebra.todense(gref))
    for i in range(2):
        blk = gs[n*i:n*(i+1), n*i:n*(i+1)]
        assert np.max(np.abs(blk-gref)) < 1e-8
    assert -np.trace(gs).imag > 0.
