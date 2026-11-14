# delta_discard of the diagonalization LDOS (ldostk/ldoswaves.py) drops the
# states that are farther than delta_discard*delta from the energies of the
# maps. The window was built from the lowest energy on both edges, so with
# several energies the states at the upper ones were dropped, and their
# maps lost their largest contributions. The window now runs from the
# lowest to the highest energy, so a dropped state is far from every
# energy and only its Lorentzian tail is lost.
import numpy as np

from pyqula import geometry
from pyqula import algebra
from pyqula.ldostk.ldoswaves import ldos_waves_from_eigsystem


def _eigensystem():
    h = geometry.honeycomb_lattice().get_supercell(3).get_hamiltonian(
            has_spin=False)
    k = np.array([0.137,0.291,0.])
    return np.linalg.eigh(algebra.todense(h.get_hk_gen()(k)))


def test_a_discarded_state_is_far_from_every_energy():
    """A dropped state is farther than delta_discard*delta from every
    energy, so it changes each map by less than its Lorentzian tail,
    1/(pi delta_discard^2 delta) times its weight on the orbital, and the
    weights of all the states on one orbital add up to one"""
    eig,vs = _eigensystem()
    delta,dd = 0.05,10.
    energies = np.array([eig[4],eig[12]]) # one state sits at each energy
    outside = (eig<energies[0]-dd*delta)|(eig>energies[1]+dd*delta)
    assert np.sum(outside) > 0 # there are states to drop
    full = ldos_waves_from_eigsystem(eig,vs.T,energies,delta)
    cut = ldos_waves_from_eigsystem(eig,vs.T,energies,delta,
            delta_discard=dd)
    diff = np.max(np.abs(full-cut),axis=1) # one number per energy
    assert np.all(diff > 0.) # something was dropped
    assert np.all(diff < 1./(np.pi*dd**2*delta))


def test_the_window_of_one_energy_is_centred_on_it():
    """With one energy the map with a discard is the sum over the states
    inside the window around it, computed here with numpy"""
    eig,vs = _eigensystem()
    delta,dd = 0.05,10.
    e = eig[12]
    cut = ldos_waves_from_eigsystem(eig,vs.T,[e],delta,delta_discard=dd)[0]
    keep = np.abs(eig-e)<dd*delta
    assert 0 < np.sum(keep) < len(eig)
    L = delta/((e-eig[keep])**2+delta**2)/np.pi
    ref = (np.abs(vs[:,keep])**2)@L
    assert np.max(np.abs(cut-ref)) < 1e-12*np.max(np.abs(ref))
