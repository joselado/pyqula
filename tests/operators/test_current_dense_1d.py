"""The 1d current operator on a dense Hamiltonian.

current.derivative inherits the type of h.intra, an np.matrix on a dense
Hamiltonian (after add_rashba, for one), so the operator returned a (1,n)
matrix for a vector, and get_vev("current") raised a shape error in the
sum over kpoints. The storage of the Hamiltonian cannot change the answer.
"""

import numpy as np

from pyqula import geometry


def _rashba_chain():
    g = geometry.chain().get_supercell(2)
    g.dimensionality = 1
    h = g.get_hamiltonian(has_spin=True)
    h.add_rashba(0.4)
    h.add_exchange([0.3, 0.2, 0.1])
    return h


def test_current_operator_returns_the_shape_of_the_vector():
    h = _rashba_chain()
    assert not h.is_sparse
    op = h.get_operator("current")
    w = np.random.default_rng(1).normal(size=h.intra.shape[0])+0j
    u = op(w, k=[0.17, 0., 0.])
    assert isinstance(u, np.ndarray) and u.shape == w.shape


def test_current_vev_does_not_depend_on_the_storage():
    h = _rashba_chain()
    hs = h.copy()
    hs.turn_sparse()
    assert not h.is_sparse and hs.is_sparse
    # a metal at a temperature that makes the occupation smooth in k, so
    # the sum over the mesh converges exponentially in nk
    kw = dict(nk=60, fermi=0.3, T=0.25)
    jd = h.get_vev("current", **kw)
    js = hs.get_vev("current", **kw)
    assert np.max(np.abs(jd-js)) < 1e-10
    sd = h.get_single_vev("current", **kw)
    ss = hs.get_single_vev("current", **kw)
    assert abs(sd-ss) < 1e-10
    # in equilibrium the total current vanishes, by the Bloch theorem: the
    # sum over k of f(E) dE/dk is the integral of a total derivative. The
    # exchange breaks time reversal, so this is not a k to -k cancellation
    assert abs(sd) < 1e-6
