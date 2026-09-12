"""h.get_average_spin_splitting had no test at all. Two invariants pin it
here, both taken from code that already lives in this repo rather than
from a recorded number: the splitting of a uniform Zeeman field, which is
known exactly and cannot depend on how big a cell the same crystal is
described in, and the collinearity guard that its sibling
spin_splitting_vs_energy already refuses non-collinear Hamiltonians with."""

import numpy as np
import pytest

from pyqula import geometry


def test_average_spin_splitting_is_intensive():
    """A uniform Zeeman field of h_z splits every band by exactly 2*h_z,
    at every k. So the AVERAGE splitting is 2*h_z whatever the cell is:
    describing the same crystal in an n-fold supercell folds the bands but
    changes no physics. It used to SUM over bands instead of averaging, so
    the answer grew as the number of bands (1.2 / 4.8 / 10.8 on the
    primitive, 2x and 3x cells of this very system)."""
    for ns in [1, 2, 3]:
        g = geometry.honeycomb_lattice()
        if ns > 1: g = g.get_supercell(ns)
        h = g.get_hamiltonian()
        h.add_zeeman([0., 0., 0.3])
        assert abs(h.get_average_spin_splitting(nk=4) - 0.6) < 1e-8, ns


def test_average_spin_splitting_matches_the_spectrum():
    """Second oracle, independent of the field being uniform: the same
    quantity read straight off the two spin-resolved spectra."""
    g = geometry.honeycomb_lattice()
    h = g.get_hamiltonian()
    h.add_zeeman([0., 0., 0.2])
    h.add_sublattice_imbalance(0.3)
    hup = h.copy() ; hup.remove_spin(channel="up")
    hdn = h.copy() ; hdn.remove_spin(channel="dn")
    from pyqula.klist import kmesh
    ks = kmesh(h.geometry.dimensionality, nk=4)
    ref = np.mean([np.mean(np.abs(
        np.sort(np.linalg.eigvalsh(np.array(hup.get_hk_gen()(k))))
        - np.sort(np.linalg.eigvalsh(np.array(hdn.get_hk_gen()(k))))))
        for k in ks])
    assert abs(h.get_average_spin_splitting(nk=4) - ref) < 1e-8


def test_spin_splitting_refuses_a_non_collinear_hamiltonian():
    """Both routines build their two channels with remove_spin, which
    drops the spin off-diagonal block silently. On a Rashba Hamiltonian
    they used to return 0.0 -- 'no spin splitting' for a Hamiltonian whose
    off-diagonal element is 0.77 -- while spin_splitting_vs_energy, which
    has the guard, refuses it. All three now agree."""
    g = geometry.honeycomb_lattice()
    h = g.get_hamiltonian()
    h.add_rashba(0.3)
    with pytest.raises(ValueError):
        h.get_average_spin_splitting(nk=4)
    with pytest.raises(ValueError):
        h.get_spin_splitting_density(nk=4)
    with pytest.raises(ValueError):
        h.get_spin_splitting_vs_energy(nk=4)  # the sibling, already guarded
