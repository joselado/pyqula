import numpy as np

from pyqula import geometry
from pyqula.scftk.spinspin import VJinteraction


def test_vjinteraction_kpm_holds_the_requested_filling_at_finite_T(
        monkeypatch, tmp_path):
    """VJinteraction(integration="kpm") builds its density matrix with the
    Fermi-Dirac weight at T, so the Fermi level must be located at the same
    T: a T=0 step count holds a different number of electrons wherever the
    density of states is not symmetric about the Fermi level. The oracle is
    the electron count of the returned Hamiltonian from an exact
    diagonalization density matrix at the same T and k-mesh, which shares
    no code with the KPM Fermi search."""
    monkeypatch.chdir(tmp_path) # the loop saves MF.pkl on convergence
    h0 = geometry.chain().get_hamiltonian(has_spin=True)
    nk, T, filling = 30, 0.05, 0.1
    scf = VJinteraction(h0, U=0.2, filling=filling, T=T, nk=nk,
            integration="kpm", npol=300, mix=0.5, maxite=100, verbose=0)
    assert scf.converged
    dm = scf.hamiltonian.get_density_matrix(ds=[(0,0,0)], nk=nk, T=T)
    nelec = np.trace(dm[(0,0,0)]).real
    assert abs(nelec - 2*filling) < 1e-3, nelec
