import warnings

import numpy as np
import pytest

from pyqula import geometry, heterostructures
from pyqula.greentk.rg import (green_renormalization_python,
                               green_renormalization_jit)

# The pure-Python decimation stopped only when the decimated couplings
# fell below the threshold. On a lead whose decimation overflows the
# couplings become NaN, NaN < error is never true, and the loop ran
# forever; the numba twin raised only because its solve refuses a
# non-finite matrix. The lead is the topological Rashba wire of the
# notebook quantum_transport/04_noncollinear_nambu_basis, at the energy
# of its Zeeman map where the S-matrix route hung.

E_HANG = 0.23164556962025312


def _wire():
    h = geometry.chain().get_hamiltonian(has_spin=True)
    h.add_rashba(0.6)
    h.add_zeeman(list(0.15*np.array([-0.6, 0., 0.8])))
    h.add_onsite(2.0)
    h.add_swave(0.1)
    return h


def _blocks(h):
    d = h.get_multihopping().get_dict()
    inter = [v for k, v in d.items()
             if tuple(int(x) for x in k) == (1, 0, 0)][0]
    return d[(0, 0, 0)], inter


def test_both_backends_raise_on_a_diverging_decimation():
    intra, inter = _blocks(_wire())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # the overflow on the way to NaN
        with pytest.raises(ValueError, match="diverged"):
            green_renormalization_python(intra, inter, energy=E_HANG,
                                         delta=1e-12)
        with pytest.raises(ValueError):
            green_renormalization_jit(intra, inter, energy=E_HANG,
                                      delta=1e-12)


def test_the_smatrix_escalates_its_broadening_instead_of_hanging():
    h = geometry.chain().get_hamiltonian(has_spin=True)
    h1 = h.copy()
    h1.add_rashba(0.6)
    h1.add_zeeman([0.3, 0.2, 0.4])
    ht = heterostructures.build(h1, _wire())
    ht.scale_lc = 0.3
    ht.delta = 1e-6
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        g = ht.didv(energy=E_HANG)
    assert any("could not be resolved" in str(w.message) for w in caught)
    assert np.isfinite(g) and 0. <= g <= 4.
