"""The dense BSE solver refuses keywords it does not read.

BSE takes **kwargs for the options of the iterative and quantics solvers,
and the dense branch used to drop them without a word, so a misspelt nk or
kernel ran the default calculation instead. metal= is refused as well: the
exciton kernel has no occupancy filter, so honouring it would build pairs
between two occupied or two empty states."""
import numpy as np
import pytest

from pyqula.bsetk.interaction import density_interaction
from testutils import gapped_ionic_chain


def _system():
    h = gapped_ionic_chain()
    return h, density_interaction(h, U=1.0, V1=0.3)


@pytest.mark.parametrize("extra", [dict(nkk=40), dict(kernal="none"),
                                   dict(metal=True)])
def test_dense_refuses_unknown_keywords(extra):
    h, W = _system()
    name = list(extra)[0]
    with pytest.raises(TypeError, match=name):
        h.get_bse(V=W, nk=4, **extra)
    with pytest.raises(TypeError, match=name):
        h.get_exciton_energies(V=W, nk=4, **extra)


def test_exciton_bands_refuses_unknown_keywords():
    h, W = _system()
    with pytest.raises(TypeError, match="nkk"):
        h.get_exciton_bands(V=W, nk=4, nq=2, nkk=40)


def test_iterative_still_takes_its_own_options():
    """The guard is on the dense branch only: the iterative solver's own
    options still reach it, and it agrees with the dense one"""
    h, W = _system()
    ref = h.get_bse(V=W, nk=16, tda=True).get_energies(n=4).real
    got = h.get_bse(V=W, nk=16, tda=True, solver="iterative", neig=4,
                    tol=1e-12, maxiter=800).get_energies().real
    assert np.max(np.abs(np.sort(got) - np.sort(ref))) < 1e-8
