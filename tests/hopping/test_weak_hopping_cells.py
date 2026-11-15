"""A weak hopping keeps its cells.

The parametric builders behind g.get_hamiltonian(tij=...) dropped every
cell whose matrix had sum|m| <= 1e-5, an absolute threshold. A weak hopping
kept its bonds inside the cell and lost all its bonds to the other cells,
so the Bloch Hamiltonian was not the one of the hopping. A cell is now
dropped only when its matrix is exactly zero.
"""

import numpy as np

from pyqula import geometry
from pyqula.specialhopping import NNG


def first_neighbor(t):
    """A hopping t between sites at distance 1"""
    def f(r1,r2):
        d = np.linalg.norm(np.array(r1)-np.array(r2))
        return t*float(abs(d-1.)<1e-6)
    return f


def test_a_weak_hopping_reaches_the_next_cell():
    """On the chain of one site every first-neighbor bond goes to the next
    cell, so H(k=0) = 2t. With t=1e-6 the two cells were dropped and
    H(k=0) was zero, for a function and for a generator"""
    g = geometry.chain()
    t = 1e-6
    for tij in (first_neighbor(t),NNG(g,[t])):
        h = g.get_hamiltonian(has_spin=False,tij=tij)
        hk = np.array(h.get_hk_gen()([0.,0.,0.]))
        assert abs(hk[0,0]-2.*t)<1e-15
