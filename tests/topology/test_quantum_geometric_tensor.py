import numpy as np

from pyqula import geometry
from pyqula import topology


def _haldane_model(has_spin=False, t2=0.2):
    g = geometry.honeycomb_lattice()
    h = g.get_hamiltonian(has_spin=has_spin)
    h.add_haldane(t2)
    # For t2=0.2 the (unshifted) direct gap is exactly [-0.9,0.9] at every
    # k, closing only in the t2->0 limit; shift_fermi(0.3) leaves a safe
    # 0.6 margin to either edge everywhere in the BZ. (A larger shift such
    # as 0.9 would put E=0 exactly AT the lower band's own edge at the K
    # point -- fragile: whether that borderline k-point then counts as
    # occupied or not is left to floating-point noise.)
    h.shift_fermi(0.3) # put the Fermi level safely mid-gap
    return h


def test_qgt_chern_matches_wilson_loop(tmp_path, monkeypatch):
    """Integrating the xy component of the Berry curvature obtained from
    the new sum-over-states quantum geometric tensor over the BZ must
    reproduce the Chern number of the (already tested) independent
    Fukui-Hatsugai-Suzuki Wilson-loop implementation, topology.chern --
    both in the trivial (C=0) and Haldane-gapped (C=+-1) cases.

    Uses the same 2x2 supercell as test_haldane_chern.py's trivial case:
    on the bare (un-supercelled) honeycomb lattice with no Haldane flux the
    two bands touch at the Dirac points, which are exactly degenerate
    (gapless) -- both the Wilson-loop and the sum-over-states Kubo formula
    are ill-defined there, so this is not actually a fair trivial-model
    comparison; the supercell sidesteps that (as the existing Haldane
    Chern-number test already relies on)."""
    monkeypatch.chdir(tmp_path) # topology.chern writes *.OUT files to cwd
    g = geometry.honeycomb_lattice().get_supercell(2)
    occ_idxs = list(range(8)) # bands 0-7, the spinful lower manifold
                               # (passed explicitly here, not relying on
                               # the E<0 default)

    h_trivial = g.get_hamiltonian() # has_spin=True, as in test_haldane_chern.py
    c_wilson_triv = topology.chern(h_trivial, nk=8)
    c_qgt_triv = topology.chern_from_qgt(h_trivial, nk=8, occ_idxs=occ_idxs)
    assert abs(round(c_wilson_triv)) == 0
    assert abs(c_qgt_triv) < 1e-2

    h = g.get_hamiltonian()
    h.add_haldane(0.2)
    c_wilson = topology.chern(h, nk=8)
    c_qgt = topology.chern_from_qgt(h, nk=8, occ_idxs=occ_idxs)
    assert abs(round(c_wilson) - c_wilson) < 1e-6
    assert round(c_wilson) != 0
    assert np.isclose(c_qgt, c_wilson, atol=1e-2)


def test_qgt_chern_default_occ_idxs_matches_explicit():
    """chern_from_qgt(h) (occ_idxs=None, the call a user reaches for first)
    must give the same result as passing occ_idxs explicitly. This is the
    default-filling code path through quantum_geometric_tensor_mesh, which
    resolves "the E<0 bands" once from a single reference k-point and
    reuses that fixed set everywhere (see _resolve_occ_idxs in qgt.py) --
    exercised here since none of the other tests call chern_from_qgt
    without occ_idxs. Unlike test_qgt_chern_matches_wilson_loop, this test
    never calls topology.chern (only chern_from_qgt, which does no file
    I/O), so it needs no tmp_path/monkeypatch.chdir."""
    h = _haldane_model(t2=0.2)
    c_default = topology.chern_from_qgt(h, nk=10)
    c_explicit = topology.chern_from_qgt(h, nk=10, occ_idxs=[0])
    assert np.isclose(c_default, c_explicit)
    assert np.isclose(c_default, 1.0, atol=1e-2)


def test_qgt_geometric_bounds_hold_pointwise():
    """Model-independent quantum-geometric bounds (see e.g. Roy, PRB 90,
    165139 (2014), and the PythTB quantum-geometric-tensor tutorial, which
    verifies the same two inequalities as a self-consistency check on a
    Haldane model): for every k-point and every band,

      weak bound:   |Omega_xy(k)| <= Tr g(k)
      strong bound: (1/4) Omega_xy(k)^2 <= det g(k)

    must hold. For an isolated band of a two-band model the strong bound
    is additionally known to saturate (equality), since the single
    occupied-band quantum geometric tensor has rank 1."""
    h = _haldane_model(t2=0.2)
    from pyqula import klist
    ks = klist.kmesh(2, nk=12)
    for k in ks:
        Q = topology.quantum_geometric_tensor(h, k=k, occ_idxs=[0])
        g_ = topology.quantum_metric_from_qgt(Q)
        omega = topology.berry_curvature_from_qgt(Q)
        omega_xy = omega[0, 1].real
        tr_g = (g_[0, 0] + g_[1, 1]).real
        det_g = (g_[0, 0]*g_[1, 1] - g_[0, 1]*g_[1, 0]).real
        assert abs(omega_xy) <= tr_g + 1e-8 # weak bound
        assert 0.25*omega_xy**2 <= det_g + 1e-8 # strong bound
        assert np.isclose(0.25*omega_xy**2, det_g, atol=1e-6) # saturated


def test_qgt_nonabelian_matches_abelian_trace():
    """The band-trace ("Abelian") quantum geometric tensor must equal the
    trace over the band-pair-resolved ("non-Abelian") tensor for the same
    band subspace -- a basis-independent identity that must hold exactly
    regardless of which orthonormal basis the diagonalization routine
    picks inside a degenerate subspace."""
    h = _haldane_model(has_spin=True, t2=0.2) # 4 bands: 2 exactly spin-degenerate pairs
    for k in ([0.31, 0.17, 0.], [0.0, 0.0, 0.], [0.5, 0.2, 0.]):
        Q_ab = topology.quantum_geometric_tensor(h, k=k, occ_idxs=[0, 1])
        Q_na = topology.quantum_geometric_tensor(h, k=k, occ_idxs=[0, 1],
                                                   non_abelian=True)
        assert np.allclose(Q_na[:, :, 0, 0] + Q_na[:, :, 1, 1], Q_ab)


def test_qgt_nonabelian_spin_degenerate_block_diagonal():
    """With no spin-orbit coupling or Zeeman splitting the Haldane
    Hamiltonian is block diagonal in spin and the two spin channels are
    identical copies of the same spinless problem. The non-Abelian quantum
    geometric tensor of the resulting exactly spin-degenerate pair of
    occupied bands must therefore: (i) vanish between the two spin
    channels (no coupling between them), and (ii) have equal diagonal
    blocks, each matching the single-band spinless result exactly -- this
    is precisely the situation ordinary single-band Kubo formulas cannot
    represent (degenerate energy denominator), which is the point of the
    non-Abelian/multiband generalization implemented here."""
    h_spinful = _haldane_model(has_spin=True, t2=0.2)
    h_spinless = _haldane_model(has_spin=False, t2=0.2)
    k = [0.31, 0.17, 0.]
    Q_na = topology.quantum_geometric_tensor(h_spinful, k=k, occ_idxs=[0, 1],
                                              non_abelian=True)
    Q_ref = topology.quantum_geometric_tensor(h_spinless, k=k, occ_idxs=[0])
    assert np.allclose(Q_na[:, :, 0, 1], 0.0, atol=1e-8) # no cross-spin coupling
    assert np.allclose(Q_na[:, :, 1, 0], 0.0, atol=1e-8)
    assert np.allclose(Q_na[:, :, 0, 0], Q_ref)
    assert np.allclose(Q_na[:, :, 1, 1], Q_ref)


_PAULI = {
    "x": np.array([[0, 1], [1, 0]], dtype=complex),
    "y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "z": np.array([[1, 0], [0, -1]], dtype=complex),
}


def _bloch_vector(hk):
    """Pauli decomposition H(k) = d0*I + d.sigma of a 2x2 Bloch Hamiltonian."""
    return np.array([np.real(np.trace(hk @ _PAULI[a]))/2. for a in "xyz"])


def test_qgt_matches_analytic_two_band_formula():
    """Independent analytic benchmark, computed without using any of this
    module's code: for a two-band Bloch Hamiltonian H(k) = d0(k) I +
    d(k).sigma (exactly the spinless Haldane model here), the lower band's
    quantum metric and Berry curvature have the closed forms (see e.g.
    Xiao, Chang & Niu, Rev. Mod. Phys. 82, 1959 (2010), Sec. II.B, mapping
    a two-level Hamiltonian to a spin in an effective field d(k))

      g_ij  =  (1/4) (d_i-hat) . (d_j-hat)
      Omega_xy = (1/2) d_hat . (d_x-hat x d_y-hat)

    with d_hat = d/|d|. This pins the *absolute scale* of the quantum
    metric (unlike the Chern-number and geometric-bound checks above,
    which are both invariant under an overall rescaling Q -> lambda^2 Q,
    so neither would catch e.g. a missing/duplicated prefactor). d(k) is
    obtained directly from h.get_hk_gen() and differentiated with a plain
    central finite difference here, entirely independent of qgt.py's
    exact analytic multicell derivative."""
    h = _haldane_model(t2=0.2)
    hkgen = h.get_hk_gen()
    dk = 1e-5
    def dhat(k):
        d = _bloch_vector(hkgen(np.array(k, dtype=float)))
        return d/np.linalg.norm(d)
    for k in ([0.1, 0.2, 0.], [0.31, 0.17, 0.], [0.05, 0.4, 0.]):
        k = np.array(k, dtype=float)
        ex, ey = np.array([1., 0., 0.]), np.array([0., 1., 0.])
        dx = (dhat(k+dk*ex) - dhat(k-dk*ex))/(2*dk)
        dy = (dhat(k+dk*ey) - dhat(k-dk*ey))/(2*dk)
        g_analytic = 0.25*np.array([[np.dot(dx, dx), np.dot(dx, dy)],
                                     [np.dot(dy, dx), np.dot(dy, dy)]])
        omega_analytic = 0.5*np.dot(dhat(k), np.cross(dx, dy))

        Q = topology.quantum_geometric_tensor(h, k=k, occ_idxs=[0])
        g_num = topology.quantum_metric_from_qgt(Q).real
        omega_num = topology.berry_curvature_from_qgt(Q)[0, 1].real

        assert np.allclose(g_num, g_analytic, atol=1e-5)
        assert np.isclose(omega_num, omega_analytic, atol=1e-5)


def test_qgt_nonabelian_berry_curvature_trace_matches_abelian():
    """The non-Abelian Berry curvature (and quantum metric) must trace
    down to the Abelian ones too, not just the raw tensor Q checked by
    test_qgt_nonabelian_matches_abelian_trace -- exercising the
    berry_curvature_from_qgt/quantum_metric_from_qgt conversion itself
    in non-Abelian mode, which the other tests never call."""
    h = _haldane_model(has_spin=True, t2=0.2)
    k = [0.31, 0.17, 0.]
    Q_ab = topology.quantum_geometric_tensor(h, k=k, occ_idxs=[0, 1])
    Q_na = topology.quantum_geometric_tensor(h, k=k, occ_idxs=[0, 1],
                                              non_abelian=True)
    omega_ab = topology.berry_curvature_from_qgt(Q_ab)
    omega_na = topology.berry_curvature_from_qgt(Q_na, non_abelian=True)
    g_ab = topology.quantum_metric_from_qgt(Q_ab)
    g_na = topology.quantum_metric_from_qgt(Q_na, non_abelian=True)
    assert np.allclose(omega_na[:, :, 0, 0] + omega_na[:, :, 1, 1], omega_ab)
    assert np.allclose(g_na[:, :, 0, 0] + g_na[:, :, 1, 1], g_ab)


def test_qgt_default_occ_idxs_follows_fermi_level_not_band_count():
    """occ_idxs=None must select bands by E<0 (the Fermi-level convention
    h.get_chern() and the rest of topology.py use, tracking
    h.shift_fermi(...)), not simply "the lower half of the bands" --
    those two choices coincide for the usual half-filled _haldane_model
    fixture (which is exactly why no other test here would catch a
    regression to band-count halving). Push the Fermi level below both
    bands of the (two-band) spinless Haldane model so *all* bands are
    occupied: the subspace then has no complement to project onto, so the
    tensor must come out identically zero -- band-count halving would
    instead (wrongly) still treat only the lower band as occupied and
    return a nonzero tensor."""
    g = geometry.honeycomb_lattice()
    h = g.get_hamiltonian(has_spin=False)
    h.add_haldane(0.2)
    h.shift_fermi(-2.0) # push spectrum well below the E=0 reference
    k = [0.31, 0.17, 0.]
    es = np.linalg.eigvalsh(h.get_hk_gen()(k))
    assert np.all(es < 0.0) # sanity: both bands now occupied
    Q = topology.quantum_geometric_tensor(h, k=k) # occ_idxs=None
    assert np.allclose(Q, 0.0)


def test_qgt_degenerate_subspace_without_gap_raises():
    """Selecting a subspace that is not gapped from its complement (e.g.
    only one of the two exactly spin-degenerate occupied bands) makes the
    sum-over-states denominator singular; this must fail loudly rather
    than silently return a wrong number."""
    h = _haldane_model(has_spin=True, t2=0.2)
    import pytest
    with pytest.raises(ValueError):
        topology.quantum_geometric_tensor(h, k=[0.31, 0.17, 0.], occ_idxs=[0])
