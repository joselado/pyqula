"""A filling of one occupies every state and a filling of zero none, at any
temperature, in the Fermi search and in the density matrix built from it.

The T=0 cut puts the Fermi level 1e-5 above the top state (below the
bottom one when empty). At T >> 1e-5 the Fermi-Dirac occupation of that
state is then about 1/2, so the density matrix missed half an electron. The
mean field with integration="ed" and the ultracell both take their Fermi
level from spectrum.get_fermi_energy_T, so this is checked there and in
the density matrix of the "ed" route."""
import numpy as np
import pytest
from scipy.special import expit

from pyqula import geometry
from pyqula import densitymatrix as dmm
from pyqula.spectrum import get_fermi_energy_T


@pytest.mark.parametrize("T", [1e-2, 0.3])
def test_fermi_level_holds_every_state_or_none(T):
    """The Fermi-Dirac count at the Fermi level of filling 1 is the number
    of states, and at filling 0 it is zero, to 1e-9 of an electron"""
    es = np.array([-1.3, -0.2, 0.4, 0.4, 2.1])
    for filling, nexp in [(1.0, len(es)), (0.0, 0)]:
        mu = get_fermi_energy_T(es, filling, T=T)
        assert abs(np.sum(expit(-(es-mu)/T)) - nexp) < 1e-9


def test_full_filling_density_matrix_holds_the_charge():
    """Tr(dm) of the onsite density matrix is the number of orbitals per
    cell at filling 1 and zero at filling 0, at T=1e-2. Gapped graphene on
    a 4x4 mesh has its top state alone at Gamma, which used to be half
    empty, so the trace was short by about 1/32 of an electron"""
    h = geometry.honeycomb_lattice().get_hamiltonian(has_spin=False)
    h.add_sublattice_imbalance(0.3)
    n = h.intra.shape[0]
    pairs = {(0, 0, 0): (np.arange(n), np.arange(n))}
    for filling, nexp in [(1.0, n), (0.0, 0)]:
        dm, fermi = dmm.full_dm_accumulate_sparse_with_fermi(h, pairs,
                filling, nk=4, delta=1e-2)
        assert abs(np.trace(dm[(0, 0, 0)]).real - nexp) < 1e-9
