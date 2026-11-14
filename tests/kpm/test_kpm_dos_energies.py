import numpy as np
import pytest

from pyqula import geometry


def _gapped_honeycomb():
    """Spinless honeycomb lattice with a sublattice mass 0.2: a smooth
    density of states at E=0.5, away from the gap and the van Hove peak"""
    h = geometry.honeycomb_lattice().get_hamiltonian(has_spin=False)
    h.add_sublattice_imbalance(0.2)
    return h


def _kpm_dos(h, energies, delta=0.05, **kwargs):
    """The KPM DOS on a fixed k-mesh, with the random vectors of the
    stochastic trace drawn from the same seed on every call"""
    np.random.seed(7)
    return h.get_dos(mode="KPM", energies=energies, nk=20, random=False,
            delta=delta, write=False, **kwargs)


def test_kpm_dos_at_an_energy_does_not_depend_on_the_other_energies():
    """The KPM DOS at one energy is the same number whether that energy is
    asked alone, among three energies or on a dense grid.

    dos_kpm used to evaluate the Chebyshev series on ne=len(energies)
    points spread over max|E| and interpolate from them, so one energy
    gave exactly zero and three energies gave a DOS several times too
    small (0.05 against 0.15 at E=0.5 on this lattice)"""
    h = _gapped_honeycomb()
    dense = np.linspace(-4., 4., 801)
    i = np.argmin(np.abs(dense-0.5))
    (_, d_dense) = _kpm_dos(h, dense)
    (_, d_three) = _kpm_dos(h, [dense[i], 2.0, 2.5])
    (_, d_one) = _kpm_dos(h, [dense[i]])
    assert d_dense[i] > 0.1 # the band is there, so the test can fail
    assert abs(d_three[0]-d_dense[i]) < 1e-10
    assert abs(d_one[0]-d_dense[i]) < 1e-10


def test_kpm_dos_takes_its_resolution_from_delta_alone():
    """The number of polynomials follows from scale and delta and is not
    an input, so npol= is refused with a message that says to pass delta.
    It used to crash with a TypeError of a keyword given twice, since npol
    was computed from delta and passed a second time"""
    h = _gapped_honeycomb()
    es = np.linspace(-3., 3., 61)
    (_, d_fine) = _kpm_dos(h, es)
    (_, d_coarse) = _kpm_dos(h, es, delta=0.5)
    assert np.max(np.abs(d_coarse-d_fine)) > 1e-3 # delta is really used
    with pytest.raises(TypeError, match="pass delta"):
        _kpm_dos(h, es, npol=40)
