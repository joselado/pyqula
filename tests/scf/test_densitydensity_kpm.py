import numpy as np
import pytest

from pyqula import geometry
from pyqula import islands
from pyqula import specialhopping
from pyqula import meanfield
from pyqula.kpmtk.densitymatrix_kpm import get_dm_kpm
from pyqula.selfconsistency.densitydensity import get_mf, get_dc_energy


def _v1_interaction_dict(h, V1=1.0):
    """Build the same spin-doubled first-neighbor interaction dictionary
    Vinteraction/Vinteraction_kpm build internally, for a frozen-Hamiltonian
    cross-check (bypassing the SCF loop entirely)."""
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


def test_get_dm_kpm_matches_full_dm_for_v1_interaction():
    """get_dm_kpm's selectively-computed density matrix must reproduce the
    same mean field / double-counting energy as exact diagonalization's
    full dense density matrix, for a first-neighbor (V1) interaction on a
    frozen Hamiltonian -- isolating get_dm_kpm's own correctness from any
    SCF convergence-path sensitivity (two independent SCF trajectories can
    settle into distinct, individually valid fixed points even when both
    density-matrix backends are correct)."""
    h = islands.get_geometry(name="honeycomb", n=2, nedges=3).get_hamiltonian()
    h.add_sublattice_imbalance(0.2)  # seed a nontrivial charge pattern
    v = _v1_interaction_dict(h)

    dm_ed = h.get_density_matrix(ds=list(v.keys()), nk=1)
    dm_kpm = get_dm_kpm(h, v, nk=1, npol=300, scale=None)

    mf_ed = get_mf(v, dm_ed)
    mf_kpm = get_mf(v, dm_kpm)
    for d in v:
        diff = np.max(np.abs(mf_ed[d]-mf_kpm[d]))
        assert diff < 1e-2, f"direction {d}: |mf_ed-mf_kpm|={diff}"
    ediff = abs(get_dc_energy(v, dm_ed) - get_dc_energy(v, dm_kpm))
    assert ediff < 1e-2


def test_hubbard_kpm_transverse_mean_field_matches_ed():
    """Regression test: hubbard_kpm builds the onsite Hubbard interaction
    v[(0,0,0)] asymmetrically (mirroring densitydensity.hubbard's own
    convention, which relies on exact diagonalization always returning a
    dense dm block regardless of v's sparsity pattern). If
    required_elements ever again requests only the (i,j) entries where v
    is nonzero -- instead of also requesting the transposed (j,i) entry
    normal_term_ij actually reads for a self-paired onsite direction -- the
    entire transverse-spin (non-collinear) part of the Hubbard mean field
    is silently computed from a zero density matrix instead of its true
    value. A Hamiltonian with a non-collinear (x-direction) exchange field
    gives a density matrix with a genuine nonzero transverse-spin
    component, exercising exactly that path."""
    g = islands.get_geometry(name="honeycomb", n=2, nedges=3)
    h = g.get_hamiltonian()
    h.add_exchange([0.3, 0., 0.])  # transverse (non-z) exchange field
    n = h.intra.shape[0]

    v = dict()
    zero = np.zeros((n, n), dtype=np.complex128)
    for i in range(n//2): zero[2*i, 2*i+1] = 1.5  # hubbard_kpm's own v construction
    v[(0, 0, 0)] = zero

    dm_ed = h.get_density_matrix(ds=[(0, 0, 0)], nk=1)
    dm_kpm = get_dm_kpm(h, v, nk=1, npol=300, scale=None)

    mf_ed = get_mf(v, dm_ed)
    mf_kpm = get_mf(v, dm_kpm)

    transverse_ed = np.array([mf_ed[(0, 0, 0)][2*i+1, 2*i] for i in range(n//2)])
    assert np.max(np.abs(transverse_ed)) > 1e-3, \
        "test setup didn't actually produce a nonzero transverse mean field"
    transverse_kpm = np.array([mf_kpm[(0, 0, 0)][2*i+1, 2*i] for i in range(n//2)])
    diff = np.max(np.abs(transverse_ed - transverse_kpm))
    assert diff < 1e-2, \
        f"KPM transverse-spin Hubbard mean field diverges from ED: {diff}"


def test_get_dm_kpm_rejects_spinless_bdg():
    """get_dm_kpm's Nambu index mapping (required_elements_eh/
    _local_nambu_index) assumes a spinful Nambu Hamiltonian (4 Nambu slots
    per site); a spinless BdG Hamiltonian uses a different (2 slots per
    site) convention it does not implement, and must fail loudly rather
    than silently computing wrong density-matrix entries."""
    h = geometry.chain().get_hamiltonian(has_spin=False)
    h.setup_nambu_spinor()  # spinless_nambu: has_eh=True, has_spin=False
    assert h.has_eh and not h.has_spin
    v = {(0, 0, 0): np.zeros((1, 1), dtype=np.complex128)}
    with pytest.raises(NotImplementedError):
        get_dm_kpm(h, v, nk=4)


def test_get_dm_kpm_matches_full_dm_for_bdg_hamiltonian():
    """required_elements_eh's Nambu-reordering index arithmetic must
    reproduce exact diagonalization's dense (2n)x(2n) density matrix for a
    (spinful) BdG/Nambu Hamiltonian, on a frozen Hamiltonian -- isolating
    get_dm_kpm's has_eh path from SCF convergence-path sensitivity."""
    h = geometry.chain().get_hamiltonian()
    h.turn_nambu()
    n = h.intra.shape[0]//2  # electron-sector (v-space) size
    v = dict()
    zero = np.zeros((n, n), dtype=np.complex128)
    zero[0, 1] += 0.6/2.
    zero[1, 0] += 0.6/2.
    v[(0, 0, 0)] = zero

    dm_ed = h.get_density_matrix(ds=list(v.keys()), nk=8)
    dm_kpm = get_dm_kpm(h, v, nk=8, npol=400, scale=None)

    mf_ed = get_mf(v, dm_ed, has_eh=True)
    mf_kpm = get_mf(v, dm_kpm, has_eh=True)
    for d in v:
        diff = np.max(np.abs(mf_ed[d]-mf_kpm[d]))
        assert diff < 1e-2, f"direction {d}: |mf_ed-mf_kpm|={diff}"


def test_get_dm_kpm_temperature_smearing_moves_toward_ed():
    """T is forwarded from the KPM SCF loop into get_dm_kpm for
    finite-temperature (Fermi-Dirac) smearing, matching what
    densitymatrix.py's full_dm(h,T=...) does for the exact-diagonalization
    path -- it must not be silently ignored, so passing an explicit
    (non-tiny) T has to actually change the resulting density matrix
    relative to the effectively-zero-temperature default."""
    h = islands.get_geometry(name="honeycomb", n=2, nedges=3).get_hamiltonian()
    v = {(0, 0, 0): np.eye(h.intra.shape[0], dtype=np.complex128)*0.5}
    dm_cold = get_dm_kpm(h, v, nk=1, npol=300, scale=None, T=1e-7)
    dm_warm = get_dm_kpm(h, v, nk=1, npol=300, scale=None, T=2.0)
    diff = np.max(np.abs(dm_cold[(0, 0, 0)] - dm_warm[(0, 0, 0)]))
    assert diff > 1e-3, "changing T had no effect on get_dm_kpm's output"
