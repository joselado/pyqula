"""The qtci DOS does not depend on the zero of energy.

A constant added to every onsite energy moves the DOS rigidly. The qtci
path interpolates each band energy with a tolerance relative to its largest
value, so raw band energies near 100 are resolved 30 times worse than band
energies near 3, and the DOS at a small broadening reads that error. Each
band must enter the interpolation measured from a centre of its own, so
that its tolerance is relative to its width.
"""
import random

import numpy as np
import pytest

from pyqula import geometry
from pyqula.dostk import simplexdos


def _graphene(c):
    """Honeycomb lattice with sublattice imbalance 0.3 and every onsite
    energy shifted by c"""
    h = geometry.honeycomb_lattice().get_hamiltonian(has_spin=False)
    h.add_sublattice_imbalance(0.3)
    if c!=0.: h.add_onsite(c)
    return h


@pytest.mark.parametrize("c", [100., 1e4])
def test_a_constant_shift_does_not_loosen_the_qtci_dos(c):
    es = np.linspace(-3.5,3.5,101)+0.0037
    kw = dict(nk=128,delta=1e-3)
    t0 = simplexdos.dos_triangle_counted(_graphene(0.),es,**kw)[1]
    tc = simplexdos.dos_triangle_counted(_graphene(c),es+c,**kw)[1]
    scale = np.max(np.abs(t0))
    # the triangle sum on the full mesh moves rigidly, up to the rounding
    # of the shifted energies against the broadening
    assert np.max(np.abs(tc-t0)) < 1e-6*scale
    # the pivot search of the interpolation draws from the random module,
    # so each curve is compared with its triangle and not with the other
    random.seed(0)
    q0 = simplexdos.dos_qtci_counted(_graphene(0.),es,**kw)[1]
    qc = simplexdos.dos_qtci_counted(_graphene(c),es+c,**kw)[1]
    # 2e-6 to 1.4e-5 was measured over three seeds and three shifts
    assert np.max(np.abs(q0-t0)) < 1e-4*scale
    assert np.max(np.abs(qc-tc)) < 1e-4*scale


@pytest.mark.parametrize("c", [0., 5.])
def test_a_flat_band_is_not_interpolated(c):
    """Measured from its own centre, the flat band of the Lieb lattice is
    rounding noise. One interpolation per field normalizes a field by its
    own largest value: the noise would be of order one and take the full
    rank, 64 on this mesh, and with it every point of the mesh. The band
    is recognized as a constant against the size of the band energies
    instead, with any shift of the onsite energies"""
    h = geometry.lieb_lattice().get_hamiltonian(has_spin=False)
    if c!=0.: h.add_onsite(c)
    es = np.linspace(-3.2,3.2,81)+0.0037+c
    kw = dict(nk=64,delta=1e-2)
    dt = simplexdos.dos_triangle_counted(h,es,**kw)[1]
    random.seed(0)
    _,dq,ndiag,ranks = simplexdos.dos_qtci_counted(h,es,**kw)
    assert ranks[1] <= 1 # the middle band is the flat one
    assert ranks[0] > 1 and ranks[2] > 1 # the other two are interpolated
    assert np.max(np.abs(dq-dt)) < 1e-5*np.max(np.abs(dt)) # 1.4e-7 measured


def test_a_band_far_from_the_mean_is_resolved_against_its_width():
    """Onsite energy 20 on one sublattice and 0 on the other: the lower
    band lies within 0.45 of zero, and 10 below the mean of the band
    energies. Measured from that mean its tolerance would be relative to
    10.45, 23 times its width, and the DOS at a small broadening reads the
    error (2e-4 to 3e-4 of the largest value). Each band is measured from
    the middle of its own range instead, so the position of a band does
    not enter its tolerance"""
    h = geometry.honeycomb_lattice().get_hamiltonian(has_spin=False)
    h.add_sublattice_imbalance(10.)
    h.add_onsite(10.)
    es = np.linspace(-0.5,0.05,111)+0.0037
    kw = dict(nk=128,delta=1e-3)
    t = simplexdos.dos_triangle_counted(h,es,**kw)[1]
    random.seed(0)
    q = simplexdos.dos_qtci_counted(h,es,**kw)[1]
    # 2e-6 to 4e-6 was measured over three seeds
    assert np.max(np.abs(q-t)) < 3e-5*np.max(np.abs(t))
