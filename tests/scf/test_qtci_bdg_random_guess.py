"""A Bogoliubov-de Gennes mean field with integration="qtci" from a random
guess. The qtci density matrix (qtcitk/densitymatrix_qtci.py) interpolates
the fields of the entries with a complex cross interpolation, so an onsite
diagonal entry, which is real, came out with an imaginary part of the size
of the interpolation error (5e-9 to 8e-8 at tolerance=1e-6). The Hartree
term carried it into the onsite mean field, about 12 times larger there for
V1 on the spinful square lattice, and the Hermiticity check of
scftk.superscf.get_mf_bdg (a norm of 1e-7) stopped the loop with
"Non-Hermitian BdG mean field" after 228 to 311 iterations. Now the onsite
diagonal entries keep only their real part, so the density matrix is
Hermitian by construction, and the loop converges to the state of "ed"
(the energies from three seeds are in future_development/
tetrahedron_density_matrix.md; the whole qtci loop takes about 500
iterations, too long for the suite, so it is checked on the converged
state of "ed" here)."""
import random

import numpy as np
import pytest

from pyqula import geometry
from pyqula import specialhopping
from pyqula.multihopping import MultiHopping
from pyqula.qtcitk.densitymatrix_qtci import get_dm_qtci
from pyqula.scftk.densitydensity import get_mf, Vinteraction


def _h0():
    h = geometry.square_lattice().get_hamiltonian(has_spin=True)
    h.setup_nambu_spinor()
    return h


def _v1_interaction_dict(h, V1):
    """The spin-doubled first-neighbor interaction Vinteraction builds"""
    nd = h.geometry.neighbor_distances()
    mgenerator = specialhopping.distance_hopping_matrix([V1/2., 0., 0.],
            nd[0:3])
    hv = h.geometry.get_hamiltonian(has_spin=False, is_multicell=True,
            mgenerator=mgenerator)
    v = hv.get_hopping_dict()
    for d in list(v.keys()):
        m = v[d]; n = m.shape[0]
        m1 = np.zeros((2*n, 2*n), dtype=np.complex128)
        for i in range(n):
            for j in range(n):
                m1[2*i:2*i+2, 2*j:2*j+2] = m[i, j]
        v[d] = m1
    return v


def _scf(integration, seed):
    """The mean field from the random guess of the seed. The pivots of the
    cross interpolation are drawn with the random module of Python, so it
    is seeded too, for the same qtci run every time"""
    np.random.seed(seed)
    random.seed(seed)
    scf = Vinteraction(_h0(), V1=-1.5, filling=0.3, nk=8, mf=None,
            integration=integration, load_mf=False, write=False, verbose=0)
    assert scf.converged
    return scf


def test_qtci_bdg_density_matrix_is_hermitian():
    """The qtci density matrix of the paired state that "ed" reaches from
    the random guess of seed 0, a state whose stacked fields have a rank
    of 51 on the 8x8 mesh: its onsite diagonal is exactly real, and the BdG
    mean field built from it is Hermitian to rounding. Before, the
    imaginary part of that diagonal was 7e-10 to 1e-9 here, depending on
    the random pivots, and the non-Hermitian part of the mean field 1.7e-8"""
    h = _scf("ed", 0).hamiltonian
    v = _v1_interaction_dict(h, V1=-1.5)
    dm = get_dm_qtci(h, v, nk=8)
    diag = dm[(0, 0, 0)].diagonal()
    assert np.all(diag.imag == 0.)
    mf = MultiHopping(get_mf(v, dm, has_eh=True))
    assert (mf-mf.get_dagger()).norm() < 1e-12

