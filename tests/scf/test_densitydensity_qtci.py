import numpy as np
import pytest

from pyqula import geometry
from pyqula import specialhopping
from pyqula.qtcitk.densitymatrix_qtci import get_dm_qtci
from pyqula.selfconsistency.densitydensity import get_mf, get_dc_energy


def _v1_interaction_dict(h, V1=1.0):
    """Build the same spin-doubled first-neighbor interaction dictionary
    Vinteraction builds internally, for a frozen-Hamiltonian cross-check
    (bypassing the SCF loop entirely) -- mirrors
    tests/scf/test_densitydensity_kpm.py's helper of the same purpose."""
    nd = h.geometry.neighbor_distances()
    mgenerator = specialhopping.distance_hopping_matrix([V1/2., 0., 0.], nd[0:3])
    hv = h.geometry.get_hamiltonian(has_spin=False, is_multicell=True,
            mgenerator=mgenerator)
    v = hv.get_hopping_dict()
    for d in list(v.keys()):
        m = v[d]; n = m.shape[0]
        m1 = np.zeros((2*n, 2*n), dtype=np.complex128)
        for i in range(n):
            for j in range(n):
                m1[2*i, 2*j] = m[i, j]
                m1[2*i+1, 2*j] = m[i, j]
                m1[2*i, 2*j+1] = m[i, j]
                m1[2*i+1, 2*j+1] = m[i, j]
        v[d] = m1
    return v


def test_get_dm_qtci_matches_full_dm_for_v1_interaction():
    """get_dm_qtci's per-element BZ integration must reproduce the same
    mean field / double-counting energy as exact diagonalization's k-mesh
    average, for a first-neighbor (V1) interaction on a frozen periodic 2D
    Hamiltonian -- isolating get_dm_qtci's own correctness from any SCF
    convergence-path sensitivity."""
    g = geometry.honeycomb_lattice()
    h = g.get_hamiltonian(has_spin=True)
    h.add_sublattice_imbalance(0.2)  # seed a nontrivial charge pattern
    v = _v1_interaction_dict(h)

    dm_ed = h.get_density_matrix(ds=list(v.keys()), nk=8)
    dm_qtci = get_dm_qtci(h, v, nk=8)

    mf_ed = get_mf(v, dm_ed)
    mf_qtci = get_mf(v, dm_qtci)
    for d in v:
        diff = np.max(np.abs(mf_ed[d]-mf_qtci[d]))
        assert diff < 1e-2, f"direction {d}: |mf_ed-mf_qtci|={diff}"
    ediff = abs(get_dc_energy(v, dm_ed) - get_dc_energy(v, dm_qtci))
    assert ediff < 1e-2


def test_get_dm_qtci_handles_symmetry_protected_zero_entries():
    """Some required (direction,i,j) entries are identically zero over the
    whole BZ for symmetry reasons (e.g. a spin-off-diagonal density-matrix
    element in a spin-conserving Hamiltonian, as in this onsite-only, no
    spin-mixing case). qutecipy's TensorCI2 refuses to start if its single
    default seed point (k=(0,0)) happens to sample zero
    ("maxsamplevalue is zero!"), so get_dm_qtci must detect and special-case
    identically-zero entries instead of crashing."""
    g = geometry.honeycomb_lattice()
    h = g.get_hamiltonian(has_spin=True)
    n = h.intra.shape[0]
    v = {(0, 0, 0): np.eye(n, dtype=np.complex128)*0.5}  # no spin mixing
    dm_qtci = get_dm_qtci(h, v, nk=8)  # must not raise
    off_diag_spin = np.array([dm_qtci[(0, 0, 0)][2*i, 2*i+1] for i in range(n//2)])
    assert np.max(np.abs(off_diag_spin)) < 1e-10


def test_get_dm_qtci_rejects_non_2d():
    """get_dm_qtci integrates over a 2D BZ (kx,ky in [0,1]x[0,1]); anything
    else must fail loudly rather than silently integrating over the wrong
    domain."""
    h = geometry.chain().get_hamiltonian()
    v = {(0, 0, 0): np.zeros((1, 1), dtype=np.complex128)}
    with pytest.raises(NotImplementedError):
        get_dm_qtci(h, v, nk=4)


@pytest.mark.slow
def test_mean_field_hamiltonian_qtci_smoke():
    """End-to-end wiring check: get_mean_field_hamiltonian(integration=
    "qtci") must run through Vinteraction -> densitydensity ->
    generic_densitydensity -> get_dm -> get_dm_qtci and converge to *some*
    self-consistent Hamiltonian. Not compared bit-for-bit against the
    exact-diagonalization SCF trajectory: two independent SCF trajectories
    can settle into distinct, individually valid (near-degenerate) fixed
    points even when both density-matrix backends are correct -- the
    frozen-Hamiltonian tests above already isolate get_dm_qtci's own
    correctness from that path-dependence."""
    g = geometry.honeycomb_lattice()
    np.random.seed(1)
    h, e = g.get_hamiltonian().get_mean_field_hamiltonian(
            U=2.0, filling=0.5, mf="random", nk=4, maxerror=1e-3,
            verbose=0, return_total_energy=True, integration="qtci")
    assert h is not None
    assert np.isfinite(e)
