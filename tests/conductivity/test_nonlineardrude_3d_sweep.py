"""The component sweep of the nonlinear Drude conductivity in 3D.

On the cubic lattice with isotropic hopping, a d-wave spin splitting
s J sin k_a sin k_b in the xz plane is the one in the xy plane with y and z
exchanged, and the Gamma-centered mesh is mapped onto itself by that
exchange. So the selection-rule sweep, a maximum over every component of
each order, must give the same numbers for the two magnets, and the xz one
must respond at first order like the xy one.
"""
from itertools import combinations_with_replacement

import numpy as np

from pyqula import geometry
from pyqula import conductivity
from pyqula.multihopping import MultiHopping


def _dwave(a, b, J=0.3):
    """Spinful cubic lattice, hopping -1, onsite 0.5, with the spin
    splitting s J sin k_a sin k_b = s (J/2) [cos(k_a-k_b) - cos(k_a+k_b)]"""
    sz = np.diag([1., -1.]).astype(complex)
    s0 = np.eye(2, dtype=complex)
    h = geometry.cubic_lattice().get_hamiltonian(has_spin=True)
    dd = {(0, 0, 0): 0.5*s0}
    for i in range(3):
        R = np.zeros(3, dtype=int)
        R[i] = 1
        dd[tuple(R)] = -1.*s0
        dd[tuple(-R)] = -1.*s0
    for sign, c in [(-1, J/4.), (1, -J/4.)]: # k_a - k_b, then k_a + k_b
        R = np.zeros(3, dtype=int)
        R[a] = 1
        R[b] = sign
        dd[tuple(R)] = c*sz
        dd[tuple(-R)] = c*sz
    h.set_multihopping(MultiHopping(dd))
    return h


def test_xz_and_xy_dwave_magnets_give_the_same_orders():
    kw = dict(lmax=3, nk=24, T=0.1)
    oxy = conductivity.nonlinear_drude_orders(_dwave(0, 1), **kw)
    oxz = conductivity.nonlinear_drude_orders(_dwave(0, 2), **kw)
    assert oxy[1] > 1e-3 # a d-wave magnet responds at first order
    assert np.allclose(oxz, oxy, rtol=1e-8, atol=1e-12*oxy[1])


def test_3d_components_match_the_single_calls():
    """Every key of the 3D sweep is the component of the single-component
    routine, the z ones included: at order l the ways to split l field
    powers between x, y and z (3, 6 and 10 of them up to third order) times
    the three currents. The first order of the xz magnet sits in
    sigma^{x;z} and sigma^{z;x}, and the third order is finite too, so the
    comparison of the values is not one of zeros"""
    h = _dwave(0, 2)
    kw = dict(nk=16, T=0.1)
    comps = dict()
    for l in (1, 2, 3):
        fields = ["".join(c) for c in combinations_with_replacement("xyz", l)]
        comps[l] = conductivity.nonlinear_drude_components(h, l, **kw)
        assert set(comps[l]) == {f+";"+b for f in fields for b in "xyz"}
        for key, value in comps[l].items():
            field, current = key.split(";")
            ref = conductivity.nonlinear_drude_conductivity(h, field=field,
                    current=current, **kw)
            assert abs(value-ref) <= 1e-14*max(abs(ref), 1.)
    assert abs(comps[1]["x;z"]) > 1e-3
    assert abs(comps[1]["x;z"]-comps[1]["z;x"]) < 1e-12
    assert max(abs(v) for v in comps[3].values()) > 1e-4


def test_the_current_runs_over_z_too():
    """The xz magnet on the cubic lattice is symmetric under the exchange
    of x and z, so at every order the largest component with the current
    along z equals the largest one with the current along x. A sweep whose
    fields run over z but whose current stops at y fails this"""
    h = _dwave(0, 2)
    for l in (1, 2, 3):
        comps = conductivity.nonlinear_drude_components(h, l, nk=16, T=0.1)
        top = {b: max(abs(v) for k, v in comps.items() if k.endswith(";"+b))
                for b in "xyz"}
        assert np.isclose(top["z"], top["x"], rtol=1e-8, atol=1e-14)


def test_the_keys_below_3d_are_the_ones_of_x_and_y():
    """In 1D and 2D the sweep keeps the keys it always had: the l field
    powers split between x and y, times the currents along x and y, and
    no z"""
    chain = geometry.chain().get_hamiltonian(has_spin=True)
    square = geometry.square_lattice().get_hamiltonian(has_spin=True)
    for h in (chain, square):
        for l in (1, 2, 3):
            comps = conductivity.nonlinear_drude_components(h, l,
                    channel="charge", nk=12, T=0.1)
            assert set(comps) == {"x"*l1+"y"*(l-l1)+";"+b
                    for l1 in range(l+1) for b in "xy"}
