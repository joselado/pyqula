"""integration="qtci" of the mean-field density matrix
(qtcitk/densitymatrix_qtci.py): the triangle weights of dmtk/simplexdm.py on
a 2^R x 2^R mesh whose band energies and band projectors come from quantics
cross interpolations. On the same mesh it must agree with
integration="triangle" to the interpolation tolerance, in a metal as in an
insulator."""
import numpy as np
import pytest

from pyqula import geometry
from pyqula import specialhopping
from pyqula.dmtk import simplexdm
from pyqula.qtcitk import densitymatrix_qtci
from pyqula.qtcitk.densitymatrix_qtci import get_dm_qtci
from pyqula.scftk.densitydensity import get_mf, get_dc_energy, Vinteraction


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
    """In a gap every triangle is full or empty, so the triangle density
    matrix is the one of exact diagonalization on the same mesh, and the
    qtci one must reproduce its mean field and double-counting energy for a
    first-neighbor (V1) interaction on a frozen 2D insulator, to the
    interpolation tolerance."""
    g = geometry.honeycomb_lattice()
    h = g.get_hamiltonian(has_spin=True)
    h.add_sublattice_imbalance(0.2)  # a gap of 0.4 at the Fermi level
    v = _v1_interaction_dict(h)

    dm_ed = h.get_density_matrix(ds=list(v.keys()), nk=8)
    dm_qtci = get_dm_qtci(h, v, nk=8)

    mf_ed = get_mf(v, dm_ed)
    mf_qtci = get_mf(v, dm_qtci)
    for d in v:
        diff = np.max(np.abs(mf_ed[d]-mf_qtci[d]))
        assert diff < 1e-6, f"direction {d}: |mf_ed-mf_qtci|={diff}"
    ediff = abs(get_dc_energy(v, dm_ed) - get_dc_energy(v, dm_qtci))
    assert ediff < 1e-6


def test_get_dm_qtci_handles_symmetry_protected_zero_entries():
    """Some required (direction,i,j) entries are identically zero over the
    whole BZ for symmetry reasons (e.g. a spin-off-diagonal density-matrix
    element in a spin-conserving Hamiltonian, as in this onsite-only, no
    spin-mixing case). The cross interpolation must not crash on them and
    must give zero."""
    g = geometry.honeycomb_lattice()
    h = g.get_hamiltonian(has_spin=True)
    n = h.intra.shape[0]
    v = {(0, 0, 0): np.eye(n, dtype=np.complex128)*0.5}  # no spin mixing
    dm_qtci = get_dm_qtci(h, v, nk=8)  # must not raise
    off_diag_spin = np.array([dm_qtci[(0, 0, 0)][2*i, 2*i+1] for i in range(n//2)])
    assert np.max(np.abs(off_diag_spin)) < 1e-10


def test_get_dm_qtci_rejects_non_2d():
    """get_dm_qtci interpolates on a 2D mesh; anything else must fail
    loudly rather than silently integrating over the wrong domain."""
    h = geometry.chain().get_hamiltonian()
    v = {(0, 0, 0): np.zeros((1, 1), dtype=np.complex128)}
    with pytest.raises(NotImplementedError):
        get_dm_qtci(h, v, nk=4)


@pytest.mark.slow
def test_mean_field_hamiltonian_qtci_smoke():
    """End-to-end wiring check: get_mean_field_hamiltonian(integration=
    "qtci") must run through Vinteraction -> densitydensity ->
    generic_densitydensity -> get_dm -> get_dm_qtci and converge to *some*
    self-consistent Hamiltonian."""
    g = geometry.honeycomb_lattice()
    np.random.seed(1)
    h, e = g.get_hamiltonian().get_mean_field_hamiltonian(
            U=2.0, filling=0.5, mf="random", nk=4, maxerror=1e-3,
            verbose=0, return_total_energy=True, integration="qtci")
    assert h is not None
    assert np.isfinite(e)


def _low_symmetry_metal():
    """Square-lattice metal with Rashba coupling and a tilted exchange
    field: complex projectors that vary along every band, and no spin
    symmetry left"""
    h = geometry.square_lattice().get_hamiltonian(has_spin=True)
    h.add_rashba(0.5)
    h.add_exchange([0.4, 0., 0.1])
    return h


@pytest.mark.parametrize("nk", [8, 64])
def test_qtci_matches_the_triangle_on_the_same_mesh_in_a_metal(nk):
    """The qtci density matrix is the triangle one on the 2^R mesh with
    interpolated fields, so every entry, with a Bloch phase and complex
    values, must agree with integration="triangle" at the same nk to the
    interpolation tolerance. The Fermi level comes from the qtci search,
    so the density matrix of the shifted Hamiltonian reuses its
    eigenpairs."""
    from pyqula.qtcitk.densitymatrix_qtci import (get_fermi4filling_qtci,
            dm_entries_qtci)
    h = _low_symmetry_metal().get_multicell()
    h.shift_fermi(-get_fermi4filling_qtci(h, 0.3, nk=nk, T=1e-3))
    ds = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)]
    needed = {(d, i, j) for d in ds for i in range(2) for j in range(2)}
    dq = dm_entries_qtci(h, needed, nk=nk, T=1e-3)
    ref = simplexdm.get_dm(h, ds, nk=nk, T=1e-3)
    err = max(abs(dq[d][i, j]-ref[d][i, j]) for (d, i, j) in needed)
    assert err < 1e-6
    assert abs(ref[(1, 0, 0)][0, 1]) > 1e-2 # a nontrivial bond entry


def test_qtci_reuses_the_eigenpairs_of_the_fermi_search_only_for_a_shift():
    """The density matrix of the Hamiltonian of the Fermi search shifted
    by a constant diagonalizes no point again, and gives what a fresh
    computation gives; any other change of the Hamiltonian starts from
    scratch."""
    from pyqula.qtcitk.densitymatrix_qtci import (get_fermi4filling_qtci,
            dm_entries_qtci)
    needed = {((0, 0, 0), 0, 1), ((1, 0, 0), 1, 0), ((0, 0, 0), 0, 0)}
    h = _low_symmetry_metal().get_multicell()
    h.shift_fermi(-get_fermi4filling_qtci(h, 0.3, nk=16, T=1e-3))
    dm, ndiag, ranks = dm_entries_qtci(h, needed, nk=16, T=1e-3, counted=True)
    assert ndiag == 0
    densitymatrix_qtci._last_mesh = None # forget the Fermi search
    dm2, ndiag2, ranks = dm_entries_qtci(h, needed, nk=16, T=1e-3,
            counted=True)
    assert ndiag2 > 0
    for (d, i, j) in needed:
        assert abs(dm[d][i, j]-dm2[d][i, j]) < 1e-12
    get_fermi4filling_qtci(h, 0.3, nk=16, T=1e-3)
    h2 = h.copy()
    h2.add_exchange([0., 0., 0.01]) # not a shift
    dm3, ndiag3, ranks = dm_entries_qtci(h2, needed, nk=16, T=1e-3,
            counted=True)
    assert ndiag3 > 0


@pytest.mark.parametrize("nk", [8, 32])
def test_qtci_density_matrix_holds_requested_filling_in_a_metal(nk):
    """In a metal the charge of the triangle weights is continuous in the
    Fermi level, so the Fermi level found on the reconstructed mesh holds
    the filling to the interpolation tolerance, while the one of the
    uniform-mesh eigenvalue count does not."""
    from pyqula.qtcitk.densitymatrix_qtci import get_fermi4filling_qtci
    from pyqula.scftk.densitydensity import get_dm
    filling, T = 0.3, 1e-7
    v = {(0, 0, 0): np.array([[0, .5], [.5, 0]], dtype=np.complex128)}
    h0 = _low_symmetry_metal()
    target = filling*h0.intra.shape[0]
    h = h0.copy()
    h.shift_fermi(-get_fermi4filling_qtci(h, filling, nk=nk, T=T))
    n_new = np.trace(get_dm(h, v, nk=nk, integration="qtci", T=T)[(0, 0, 0)]).real
    assert abs(n_new-target) < 1e-6
    hold = h0.copy()
    hold.shift_fermi(-hold.get_fermi4filling(filling, nk=nk, T=T))
    n_old = np.trace(get_dm(hold, v, nk=nk, integration="qtci", T=T)[(0, 0, 0)]).real
    assert abs(n_new-target) < abs(n_old-target)


def test_qtci_fermi_level_at_finite_temperature_is_exact():
    """With a smearing T comparable to the level spacing the qtci density
    matrix holds the filling too."""
    from pyqula.qtcitk.densitymatrix_qtci import get_fermi4filling_qtci
    from pyqula.scftk.densitydensity import get_dm
    h = _low_symmetry_metal()
    v = {(0, 0, 0): np.array([[0, .5], [.5, 0]], dtype=np.complex128)}
    h.shift_fermi(-get_fermi4filling_qtci(h, 0.3, nk=8, T=0.05))
    n = np.trace(get_dm(h, v, nk=8, integration="qtci", T=0.05)[(0, 0, 0)]).real
    assert abs(n-0.6) < 1e-6


def test_qtci_scf_loop_density_matrix_holds_filling():
    """The same invariant through the SCF wiring: the density matrix the
    Vinteraction loop computes under integration="qtci" holds 2*filling."""
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
        assert abs(np.trace(dm[(0, 0, 0)]).real-0.6) < 1e-6


def test_get_dm_qtci_matches_dense_mesh_in_a_gapped_insulator():
    """In a gap the triangle density matrix is the uniform-mesh one, which
    converges exponentially: every required entry agrees with a dense
    nk=80 mesh to 1e-3 at nk=8 (measured 2e-4) and to 1e-5 at nk=16."""
    from pyqula.kpmtk.densitymatrix_kpm import required_elements
    g = geometry.honeycomb_lattice()
    h = g.get_hamiltonian(has_spin=True)
    h.add_sublattice_imbalance(0.5)
    v = _v1_interaction_dict(h)
    need = required_elements(v)
    ds = sorted({d for (d, i, j) in need})
    ref = h.get_density_matrix(ds=ds, nk=80)
    for nk, tol in ((8, 1e-3), (16, 1e-5)):
        dq = get_dm_qtci(h, v, nk=nk)
        err = max(abs(dq[d][i, j]-ref[d][i, j]) for (d, i, j) in need)
        assert err < tol


def test_get_dm_qtci_nambu_matches_dense_mesh():
    """The BdG path (required_elements_eh): a frozen spinful s-wave
    Hamiltonian's qtci density matrix agrees with a dense mesh on every
    entry the anomalous mean field reads, and so does that mean field."""
    from pyqula.kpmtk.densitymatrix_kpm import required_elements_eh
    h = geometry.square_lattice().get_hamiltonian()
    h.shift_fermi(-1.0)
    h.setup_nambu_spinor()
    h.add_swave(0.4)
    v = {(0, 0, 0): np.zeros((2, 2), dtype=np.complex128)}
    v[(0, 0, 0)][0, 1] = v[(0, 0, 0)][1, 0] = -1.5
    need = required_elements_eh(v)
    ds = sorted({d for (d, i, j) in need})
    ref = h.get_density_matrix(ds=ds, nk=80)
    dq = get_dm_qtci(h, v, nk=16)
    assert max(abs(dq[d][i, j]-ref[d][i, j]) for (d, i, j) in need) < 1e-2
    mq = get_mf(v, dq, has_eh=True)
    mr = get_mf(v, ref, has_eh=True)
    assert max(np.max(np.abs(mq[d]-mr[d])) for d in mr) < 3e-2
    assert max(np.max(np.abs(mr[d])) for d in mr) > 0.1 # nontrivial pairing


def test_full_dm_qtci_agrees_with_get_dm_qtci_and_fills_every_entry():
    """full_dm_qtci (what scf.dm reports under integration="qtci") is the
    same construction as get_dm_qtci: equal on the entries get_dm_qtci
    computes, and nonzero where get_dm_qtci left a zero placeholder."""
    from pyqula.qtcitk.densitymatrix_qtci import full_dm_qtci
    g = geometry.honeycomb_lattice()
    h = g.get_hamiltonian(has_spin=True)
    h.add_sublattice_imbalance(0.5)
    n = h.intra.shape[0]
    v = {(0, 0, 0): np.eye(n, dtype=np.complex128)*0.5} # onsite only
    dq = get_dm_qtci(h, v, nk=8)
    full = full_dm_qtci(h, [(0, 0, 0)], nk=8)
    for i in range(n):
        assert abs(dq[(0, 0, 0)][i, i]-full[(0, 0, 0)][i, i]) < 1e-7
    assert dq[(0, 0, 0)][0, 2] == 0. # placeholder: v does not read it
    ref = h.get_density_matrix(ds=[(0, 0, 0)], nk=80)[(0, 0, 0)]
    assert np.max(np.abs(full[(0, 0, 0)]-ref)) < 1e-3


@pytest.mark.slow
def test_qtci_scf_dm_is_complete():
    """scf.dm under integration="qtci" must be the full density matrix, not
    the loop's partial one: the inter-sublattice coherence the Hubbard mean
    field never reads used to be reported as exactly zero."""
    g = geometry.honeycomb_lattice()
    scf = Vinteraction(g.get_hamiltonian(has_spin=True), U=1.0, filling=0.5,
            mf="antiferro", nk=6, maxerror=1e-5, integration="qtci",
            load_mf=False, verbose=0)
    ref = scf.hamiltonian.get_density_matrix(ds=[(0, 0, 0)], nk=80)[(0, 0, 0)]
    dm = scf.dm[(0, 0, 0)]
    assert abs(dm[0, 2]) > 0.2 # A-up/B-up hopping coherence
    assert np.max(np.abs(dm-ref)) < 1e-2


def test_qtci_total_energy_of_a_metal_is_the_triangle_one():
    """Under integration="qtci" the total energy of a nearly bare metal is
    the band energy of the triangle weights with the Bloechl term on the
    reconstructed mesh: it equals integration="triangle" at the same nk to
    the interpolation tolerance, and a dense exact sum to 1e-3 at nk=8,
    where the Gauss-Kronrod rule of the earlier qtci missed it by 4.8e-3."""
    from pyqula.klist import kmesh
    h0 = geometry.square_lattice().get_hamiltonian(has_spin=True)
    nk = 8
    kw = dict(U=1e-6, filling=0.3, nk=nk, mf="ferro", T=1e-7,
            return_total_energy=True, maxerror=1e-9, verbose=0)
    h, e = h0.copy().get_mean_field_hamiltonian(integration="qtci", **kw)
    ht, et = h0.copy().get_mean_field_hamiltonian(integration="triangle", **kw)
    assert abs(e-et) < 1e-6
    nkd = 400 # dense reference: sum the 0.6*nkd^2 lowest eigenvalues
    hk = h0.get_hk_gen()
    es = np.sort(np.array([np.linalg.eigvalsh(hk(k))
        for k in kmesh(2, nk=nkd)]).ravel())
    eref = es[:int(round(0.6*nkd**2))].sum()/nkd**2
    assert abs(e-eref) < 1e-3


@pytest.mark.parametrize("nk", [8, 16])
def test_qtci_fermi_level_holds_the_filling_in_a_high_symmetry_metal(nk):
    """On the bare square lattice the levels of a coarse mesh are highly
    degenerate, and a Fermi level cut between levels missed a 0.3 filling
    by 0.02. On the triangles the charge is continuous, so the density
    matrix holds the filling, and the Fermi level lands near the
    dense-mesh one (-1.059)."""
    from pyqula.qtcitk.densitymatrix_qtci import (get_fermi4filling_qtci,
            full_dm_qtci)
    h = geometry.square_lattice().get_hamiltonian(has_spin=False)
    mu = get_fermi4filling_qtci(h, 0.3, nk=nk, T=1e-7)
    h.shift_fermi(-mu)
    n = np.trace(full_dm_qtci(h, [(0, 0, 0)], nk=nk, T=1e-7)[(0, 0, 0)]).real
    assert abs(n - 0.3) < 1e-6
    assert abs(mu - (-1.059)) < 0.1


def test_integrate_robust_skips_a_pivot_that_underflows():
    """A density-matrix entry that is zero up to rounding (a spin-off-
    diagonal entry of a collinear ferromagnet) can hold a subnormal value at
    a node, 1e-322 on the honeycomb ferromagnet. That value is nonzero, but
    times the Gauss-Kronrod weight it underflows to exactly zero, and the
    cross interpolation stopped with "maxsamplevalue is zero!". It has to be
    skipped like a zero, and the rule then gives zero. integrate_robust is
    what topology.chern_qtci integrates with."""
    from pyqula.qtcitk.gkintegrate import integrate_robust, gkorder_from_nk
    from pyqula.qutecipytk.gausskronrod import kronrod
    GKorder = gkorder_from_nk(16)
    x, _, _ = kronrod(GKorder//2, -1, 1)
    kc = (x[len(x)//2]+1)/2 # the central node, the first diagonal hit
    tiny = np.nextafter(0., 1.) # the smallest subnormal
    def f(k):
        return tiny if (k[0] == kc and k[1] == kc) else 0.
    assert integrate_robust(np.complex128, f, GKorder, 1e-6) == 0.


def _scf_pair(model, nk, other="triangle"):
    """the qtci mean field of one model and the one of another integration
    on the same mesh, from the same guess"""
    if model == "square": # the partly polarized Hubbard ferromagnet
        h = geometry.square_lattice().get_hamiltonian(has_spin=True)
        kw = dict(U=6.0, filling=0.4, mf="ferro", T=1e-3)
    else: # a sublattice-imbalance insulator at half filling
        h = geometry.honeycomb_lattice().get_hamiltonian(has_spin=True)
        h.add_sublattice_imbalance(0.5)
        kw = dict(U=1.0, filling=0.5, mf="ferro", T=1e-7)
    out = []
    for integration in ("qtci", other):
        np.random.seed(1)
        out.append(Vinteraction(h.copy(), nk=nk, integration=integration,
            solver="broyden_mixing", maxerror=1e-8, load_mf=False,
            write=False, verbose=0, **kw))
    return out


def _magnetization(dm):
    n = dm.shape[0]
    return sum((dm[2*i, 2*i]-dm[2*i+1, 2*i+1]).real for i in range(n//2))


def test_qtci_scf_reaches_the_triangle_state_on_the_square_ferromagnet():
    """The square-lattice Hubbard ferromagnet (U=6, filling 0.4, T=1e-3) is
    a metal with both spin Fermi surfaces. The Gauss-Kronrod rule of the
    earlier qtci did not resolve them and converged to m=0.005 instead of
    m=0.474. The qtci mean field must now reach the triangle state of the
    same mesh: the same magnetization, Fermi level and total energy, with
    the filling exact."""
    sq, st = _scf_pair("square", 8)
    assert sq.converged and st.converged
    mq, mt = _magnetization(sq.dm[(0, 0, 0)]), _magnetization(st.dm[(0, 0, 0)])
    assert abs(mt) > 0.4 # the ferromagnet, not the paramagnet
    assert abs(mq-mt) < 1e-6
    assert abs(sq.hamiltonian.fermi-st.hamiltonian.fermi) < 1e-6
    assert abs(sq.total_energy-st.total_energy) < 1e-6
    assert abs(np.trace(sq.dm[(0, 0, 0)]).real-0.8) < 1e-6


def test_qtci_scf_matches_ed_in_an_insulator():
    """In an insulator every triangle is full or empty, so the qtci mean
    field must be the one of exact diagonalization on the same mesh."""
    sq, se = _scf_pair("insulator", 8, other="ed")
    assert abs(sq.total_energy-se.total_energy) < 1e-7
    assert np.max(np.abs(sq.dm[(0, 0, 0)]-se.dm[(0, 0, 0)])) < 1e-6
