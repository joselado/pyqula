import numpy as np

from pyqula import geometry
from pyqula.scftk.spinspin import VJinteraction
from pyqula.kpmtk.densitymatrix_kpm import get_fermi4filling_kpm


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


def test_kpm_fermi_level_is_not_pinned_to_the_energy_grid():
    """A temperature far below the spacing of the KPM energy grid cannot
    be resolved, so the Fermi level at T=1e-7, the default of every KPM
    mean field, must be the T=0 one, and it must follow a rigid onsite
    shift of the bands. A Fermi-Dirac weight evaluated on the grid itself
    is a step at that T, which pinned the Fermi level to a grid point: it
    stayed put while the bands moved, then jumped by a whole grid step
    (2*0.99*scale/(4*npol-1), 0.03 here), and the KPM mean-field loop
    cycled at a floor instead of converging. The scale is fixed so that
    the shift moves the bands and not the grid."""
    h = geometry.triangular_lattice().get_hamiltonian()
    h.add_exchange([0.3, 0., 0.2])
    follow = []
    for eps in np.linspace(0., 0.02, 6):
        hs = h.copy()
        hs.add_onsite(eps)
        e0 = get_fermi4filling_kpm(hs, 0.3, nk=12, npol=200, scale=12., T=0.)
        e1 = get_fermi4filling_kpm(hs, 0.3, nk=12, npol=200, scale=12., T=1e-7)
        assert abs(e1 - e0) < 1e-10, (eps, e0, e1)
        follow.append(e1 - eps)
    assert np.ptp(follow) < 2e-3, follow
