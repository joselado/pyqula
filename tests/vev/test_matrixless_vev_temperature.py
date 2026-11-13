"""An operator with no matrix is read at the temperature of the loop.

A self-consistent Hamiltonian carries the temperature its loop converged at
(h.temperature), and densitymatrix.full_dm reads it at that temperature by
default. An operator defined only by its action goes through
vev.kresolved_orbital_vev instead, which used to fall back to the fixed
smearing delta_dm, so one observable gave two numbers depending on whether
the operator had a matrix. On a metal the difference is of order one.
"""

import numpy as np

from pyqula import geometry
from pyqula.operators import Operator


def _metal_at_temperature(T):
    """A spinful honeycomb metal with every spin component polarized, and
    the temperature stored as a mean-field loop stores it"""
    g = geometry.honeycomb_lattice()
    h = g.get_hamiltonian(has_spin=True)
    h.add_rashba(0.3)
    h.add_exchange([0.3, 0.5, 0.2])
    h.add_onsite(0.4) # away from half filling, a Fermi surface
    h.temperature = T # what every mean-field loop stores
    return h


def _blind(h, name):
    """The same operator, given only by its action"""
    m = h.get_operator(name).get_matrix()
    out = Operator(lambda v, k=None: m @ v)
    assert out.matrix is None
    return out


def test_matrixless_operator_uses_the_stored_temperature():
    h = _metal_at_temperature(0.3)
    nk = 8
    for name in ["sx", "sy", "sz"]:
        a = h.get_single_vev(name, nk=nk)
        b = h.get_single_vev(_blind(h, name), nk=nk)
        assert abs(a-b) < 1e-10
        # and both are the explicit reading at the stored temperature
        c = h.get_single_vev(_blind(h, name), nk=nk, T=0.3)
        assert abs(b-c) < 1e-10
        # site resolved too
        pa = h.get_vev(name, nk=nk)
        pb = h.get_vev(_blind(h, name), nk=nk)
        assert np.max(np.abs(pa-pb)) < 1e-10
    # the temperature matters here, so the agreement has teeth
    cold = h.get_single_vev("sy", nk=nk, T=1e-6)
    assert abs(h.get_single_vev("sy", nk=nk)-cold) > 1e-3


def test_matrixless_operator_unchanged_without_a_temperature():
    """A Hamiltonian that is not a mean-field output has temperature 0 and
    is read at delta_dm by both routes, as before"""
    h = _metal_at_temperature(0.)
    from pyqula.densitymatrix import delta_dm
    a = h.get_single_vev(_blind(h, "sy"), nk=8)
    b = h.get_single_vev(_blind(h, "sy"), nk=8, T=delta_dm)
    c = h.get_single_vev("sy", nk=8)
    assert abs(a-b) < 1e-12
    assert abs(a-c) < 1e-10
