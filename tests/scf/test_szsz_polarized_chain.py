import numpy as np
import pytest

from pyqula import geometry, meanfield
from pyqula.scftk import spinspin


def _polarized_chain(nk, T):
    """The fully polarized chain of the exchange guide page: J1=-6 at
    filling=0.2, from a guess with an exchange field of 1"""
    h = geometry.chain().get_hamiltonian(has_spin=True)
    mf = meanfield.guess(h, "ferroZ", fun=1.0)
    scf = spinspin.SzSz(h, J1=-6.0, filling=0.2, mf=mf, nk=nk, mix=0.3,
            maxite=1000, T=T, verbose=0, write=False, load_mf=False)
    occ = np.diag(scf.dm[(0, 0, 0)]).real # n_up, n_down of the site
    return scf, occ


@pytest.mark.parametrize("nk", [60, 100])
def test_polarized_chain_converges_with_a_small_temperature(nk):
    """With 0.4*nk electrons in the polarized band (always an even number)
    the two states at +k_F and -k_F sit exactly at the Fermi level, half
    occupied. A complex hopping mean field (a current) splits them, and
    at T close to zero their occupations jump to 1 and 0, which feeds back
    the opposite current, so the plain loop oscillates forever at the
    default T. A small T smooths the occupations: the loop must converge
    to the same fully polarized state as at nk=40, where the default T
    converges, with the same moment and energy per site."""
    ref, occ_ref = _polarized_chain(40, 1e-7)
    scf, occ = _polarized_chain(nk, 1e-2)
    assert ref.converged and scf.converged
    assert np.allclose(occ_ref, [0.0, 0.4], atol=1e-6) # fully polarized
    assert np.allclose(occ, occ_ref, atol=1e-6)
    # the meshes differ by an O(1/nk^2) error of the k-sum
    assert abs(scf.total_energy - ref.total_energy) < 2e-3
    assert scf.total_energy < -0.645 # below the unpolarized solution


@pytest.mark.parametrize("nk", [60, 100])
def test_polarized_chain_observables_read_at_the_loop_temperature(nk):
    """The converged Hamiltonian has its zero of energy at the chemical
    potential found at T=1e-2, which leaves the pair of states at +k_F and
    -k_F slightly below zero rather than at it. Read at the default
    smearing of the density matrix (1e-6) both were counted as occupied,
    one electron too many on the mesh, and the moment came out as -0.4024
    at nk=60 and -0.4100 at nk=100. Read at the temperature of the loop,
    which the Hamiltonian stores, the moment and the filling are exactly
    those of the self-consistent density matrix, 0.4 electrons of spin
    down per site."""
    scf, occ = _polarized_chain(nk, 1e-2)
    h = scf.hamiltonian
    assert scf.converged
    assert h.temperature == 1e-2 # stored by the loop, like h.nk
    mz = h.get_magnetization(nk=nk)[0, 2]
    assert abs(mz - (occ[0] - occ[1])) < 1e-6 # the density matrix's
    assert abs(mz + 0.4) < 1e-6
    assert abs(h.get_vev("sz", nk=nk)[0] + 0.4) < 1e-6
    assert abs(h.get_filling(nk=nk) - 0.2) < 1e-6
    # an explicit smearing still wins over the stored one, and at the old
    # default the pair is counted twice again
    mz_sharp = h.get_magnetization(nk=nk, T=1e-6)[0, 2]
    assert mz_sharp < -0.4 - 1e-3 # -0.4024 at nk=60, -0.4100 at nk=100


def test_polarized_chain_non_degenerate_readout_unchanged():
    """At nk=48 the polarized band holds 19 electrons, an odd number, so
    no state sits at the Fermi level and the temperature of the readout
    does not matter: the stored T=1e-2 gives the same moment and filling
    as the old default smearing and the T=0 count"""
    from pyqula import densitymatrix
    nk = 48
    scf, occ = _polarized_chain(nk, 1e-2)
    h = scf.hamiltonian
    assert scf.converged
    mz = h.get_magnetization(nk=nk)[0, 2]
    mz0 = h.get_magnetization(nk=nk, T=densitymatrix.delta_dm)[0, 2]
    assert abs(mz - mz0) < 1e-8
    assert abs(mz + 19./48) < 1e-6
    assert abs(h.get_filling(nk=nk) - h.get_filling(nk=nk, T=0.)) < 1e-8


def test_hand_built_hamiltonian_readout_unchanged():
    """A Hamiltonian that did not come out of a self-consistent loop keeps
    the old default smearing for the moment and the step for the filling"""
    from pyqula import geometry, densitymatrix
    h = geometry.honeycomb_lattice().get_hamiltonian(has_spin=True)
    h.add_exchange([0.1, 0.2, 0.3])
    h.add_onsite(0.4)
    assert h.temperature == 0.
    nk = 12
    m = h.get_magnetization(nk=nk)
    m0 = h.get_magnetization(nk=nk, T=densitymatrix.delta_dm)
    assert np.allclose(m, m0, atol=1e-12)
    assert h.get_filling(nk=nk) == h.get_filling(nk=nk, T=0.)
