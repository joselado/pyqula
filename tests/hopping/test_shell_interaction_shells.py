"""The neighbor shells of an interaction given as a list of couplings,
built by specialhopping.shell_interaction for the mean field, the RPA and
the Bethe-Salpeter kernel. Every coupling of the list must reach its shell.
"""

import numpy as np

from pyqula import geometry
from pyqula import specialhopping


def test_every_shell_of_the_list_is_taken():
    """On the chain of one site the n-th shell is the pair n cells away.
    shell_interaction asked neighbor_distances for its default of four
    shells, so a fifth coupling was dropped with no warning"""
    g = geometry.chain()
    v = specialhopping.shell_interaction(g,[0.,0.,0.,0.,0.7])
    for d in [(5,0,0),(-5,0,0)]:
        assert d in v
        assert abs(v[d][0,0]-0.7)<1e-12
    total = sum(np.sum(m) for m in v.values()) # the k=0 coupling
    assert abs(total-1.4)<1e-12
