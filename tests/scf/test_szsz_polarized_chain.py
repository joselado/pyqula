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
