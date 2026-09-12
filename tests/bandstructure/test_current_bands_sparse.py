"""current_bands batched H(k) across kpoints by hand, with
np.array([...],dtype=np.complex128), which cannot coerce a scipy sparse
matrix: the same Hamiltonian ran dense and died with "must be real
number, not csc_matrix" sparse. Every batched site has to go through
htk.eigenvectors.hk_matrix_batch, which densifies each H(k) first."""

import numpy as np

from pyqula import geometry
from pyqula.bandstructure import current_bands


def test_current_bands_agrees_between_the_dense_and_sparse_forms(tmp_path,
                                                                 monkeypatch):
    """The two Hamiltonians are numerically identical -- one is the
    other's sparse storage -- so anything that differs between them is a
    representation bug rather than physics."""
    monkeypatch.chdir(tmp_path)
    g = geometry.chain().supercell(4)
    h = g.get_hamiltonian(has_spin=False)
    h.add_onsite(lambda r: 0.3*np.cos(np.pi*2.*r[0]/4.))
    klist = np.linspace(0., 1., 8)
    current_bands(h, klist=klist)
    dense = np.genfromtxt("BANDS.OUT")
    hs = h.copy() ; hs.turn_sparse()
    assert hs.is_sparse
    current_bands(hs, klist=klist)   # used to raise TypeError here
    sparse = np.genfromtxt("BANDS.OUT")
    assert np.array_equal(dense[:, :2], sparse[:, :2])  # kpoint and energy
    assert np.allclose(dense[:, 2], sparse[:, 2], atol=1e-14)  # the current
