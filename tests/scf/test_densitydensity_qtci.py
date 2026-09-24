import numpy as np
import pytest

from pyqula import geometry
from pyqula import specialhopping
from pyqula.qtcitk.densitymatrix_qtci import get_dm_qtci
from pyqula.scftk.densitydensity import get_mf, get_dc_energy


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


def _low_symmetry_metal():
    """Square-lattice metal with Rashba coupling and a tilted exchange
    field: these lift the spin and k-space degeneracies, so that on the
    Gauss-Kronrod node grid every level is (nearly) its own step in the
    electron count."""
    h = geometry.square_lattice().get_hamiltonian(has_spin=True)
    h.add_rashba(0.5)
    h.add_exchange([0.4, 0., 0.1])
    return h


def _largest_level_weight(h, nk):
    """Largest weight carried by one (degenerate) level on the node grid
    get_dm_qtci integrates on -- the resolution of its electron count."""
    from pyqula.qtcitk.densitymatrix_qtci import gk_node_grid
    kx, ky, w = gk_node_grid(nk)
    hk = h.get_hk_gen()
    es = np.array([np.linalg.eigvalsh(hk([x, y, 0.])) for x, y in zip(kx, ky)])
    E = es.ravel(); W = np.repeat(w, es.shape[1])
    o = np.argsort(E); E, W = E[o], W[o]
    groups = np.split(W, np.nonzero(np.diff(E) > 1e-10)[0]+1)
    return max(np.sum(x) for x in groups)


@pytest.mark.parametrize("nk", [8, 32])
def test_qtci_density_matrix_holds_requested_filling_in_a_metal(nk):
    """In a metal the Fermi level used by the qtci density matrix must be
    found on the Gauss-Kronrod nodes that density matrix is integrated on:
    one taken from the uniform mesh held 0.03-0.05 electrons too few out of
    0.6. The trace must match 2*filling to half a level's weight, and beat
    the uniform-mesh Fermi level."""
    from pyqula.qtcitk.densitymatrix_qtci import get_fermi4filling_qtci
    from pyqula.scftk.densitydensity import get_dm
    filling, T = 0.3, 1e-7
    v = {(0, 0, 0): np.array([[0, .5], [.5, 0]], dtype=np.complex128)}
    h0 = _low_symmetry_metal()
    target = filling*h0.intra.shape[0]
    h = h0.copy()
    h.shift_fermi(-get_fermi4filling_qtci(h, filling, nk=nk, T=T))
    n_new = np.trace(get_dm(h, v, nk=nk, integration="qtci", T=T)[(0, 0, 0)]).real
    assert abs(n_new-target) <= 0.5*_largest_level_weight(h, nk) + 1e-8
    hold = h0.copy()
    hold.shift_fermi(-hold.get_fermi4filling(filling, nk=nk, T=T))
    n_old = np.trace(get_dm(hold, v, nk=nk, integration="qtci", T=T)[(0, 0, 0)]).real
    assert abs(n_new-target) < abs(n_old-target)


def test_qtci_fermi_level_at_finite_temperature_is_exact():
    """With a smearing T comparable to the level spacing the count is
    continuous, and the qtci density matrix holds the filling exactly."""
    from pyqula.qtcitk.densitymatrix_qtci import get_fermi4filling_qtci
    from pyqula.scftk.densitydensity import get_dm
    h = _low_symmetry_metal()
    v = {(0, 0, 0): np.array([[0, .5], [.5, 0]], dtype=np.complex128)}
    h.shift_fermi(-get_fermi4filling_qtci(h, 0.3, nk=8, T=0.05))
    n = np.trace(get_dm(h, v, nk=8, integration="qtci", T=0.05)[(0, 0, 0)]).real
    assert abs(n-0.6) < 1e-6


def test_qtci_scf_loop_density_matrix_holds_filling():
    """The same invariant through the SCF wiring: the density matrix the
    Vinteraction loop computes under integration="qtci" holds 2*filling
    (to the node-grid resolution, about 0.03 electrons here)."""
    from pyqula.scftk.densitydensity import Vinteraction
    dms = []
    def capture(dm):
        dms.append(dm)
        return dm
    np.random.seed(0)
    Vinteraction(_low_symmetry_metal(), U=1.0, filling=0.3, nk=8,
            mf=None, load_mf=False, maxite=1, verbose=0,
            integration="qtci", callback_dm=capture)
    assert len(dms) > 0
    for dm in dms:
        assert abs(np.trace(dm[(0, 0, 0)]).real-0.6) < 0.015
