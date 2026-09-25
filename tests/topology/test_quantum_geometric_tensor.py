import numpy as np
import pytest

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


# --- independent oracles ------------------------------------------------
# Everything below builds the Bloch Hamiltonian of either gauge from
# h.get_hk_gen() and the geometry alone, without any of qgt.py's code.

def _orbital_fractions(h):
    """Fractional coordinates of every orbital along the periodic lattice
    vectors, from a least-squares solve of r = sum_i f_i a_i (the site index
    is the slowest one in pyqula's basis)"""
    g = h.geometry
    A = np.array([g.a1, g.a2, g.a3][:h.dimensionality])
    f = np.linalg.lstsq(A.T, np.array(g.r).T, rcond=None)[0].T
    return np.repeat(f, h.intra.shape[0]//len(g.r), axis=0)


def _bloch_generator(h, gauge):
    """k -> H(k) in the requested gauge: hk_gen itself for "lattice", and
    D^dag H D with D = diag(exp(2 pi i k.f_j)) for "atomic", i.e. every
    hopping carrying the full bond vector R + f_j - f_i in its phase"""
    hk = h.get_hk_gen()
    from pyqula import algebra
    if gauge == "lattice":
        return lambda k: np.asarray(algebra.todense(hk(k)))
    f = _orbital_fractions(h)
    dim = h.dimensionality
    def hka(k):
        ph = np.exp(2j*np.pi*(f@np.array(k, dtype=float)[:dim]))
        return np.conj(ph)[:, None]*np.asarray(algebra.todense(hk(k)))*ph[None, :]
    return hka


def _projector_fd(h, k0, occ, gauge, dk=1e-5):
    """P d_iP d_jP P by a central finite difference of the projector on the
    bands occ, in reduced k, in the requested gauge"""
    hk = _bloch_generator(h, gauge)
    def P(k):
        w = np.linalg.eigh(hk(k))[1][:, occ]
        return w@w.conj().T
    dim = h.dimensionality
    k0 = np.array(k0, dtype=float)
    dP = [(P(k0 + dk*e) - P(k0 - dk*e))/(2*dk) for e in np.eye(3)[:dim]]
    P0 = P(k0)
    return np.array([[P0@dP[i]@dP[j]@P0 for j in range(dim)]
                     for i in range(dim)])


def _cartesian_qgt(h, K, occ, gauge="atomic"):
    """Abelian tensor at the Cartesian k-point K, in Cartesian components:
    k_i = a_i.K/(2 pi) and Q_cart = J Q_red J^T with J[a,i] = a_i[a]/(2 pi)"""
    g = h.geometry
    dim = h.dimensionality
    A = np.array([g.a1, g.a2, g.a3][:dim])[:, :dim]
    kr = list(A@np.array(K)/(2*np.pi)) + [0.]*(3 - dim)
    J = A.T/(2*np.pi)
    Q = topology.quantum_geometric_tensor(h, k=kr, occ_idxs=occ, gauge=gauge)
    return J@Q@J.T


# --- models with degenerate multiplets ----------------------------------

def _rashba_exchange():
    """Occupied pair mixing spin, no degeneracy"""
    h = _haldane_model(has_spin=True, t2=0.2)
    h.add_rashba(0.3)
    h.add_exchange([0.1, 0.2, 0.1])
    return h, [0, 1], [[0.31, 0.17, 0.], [0.62, 0.05, 0.]]


def _kane_mele():
    """Inversion and time reversal: every band is an exact Kramers pair at
    every k, so the solver picks an arbitrary basis of the occupied pair
    everywhere, high-symmetry points included"""
    h = geometry.honeycomb_lattice().get_hamiltonian(has_spin=True)
    h.add_kane_mele(0.1)
    ks = [[0.31, 0.17, 0.], [0., 0., 0.], [0.5, 0., 0.], [1/3., 1/3., 0.]]
    return h, [0, 1], ks


def _threefold():
    """Three orbitals per site, with an exactly threefold degenerate lower
    multiplet at every k, and a different random unitary mixing the three
    orbitals of each site, so that the multiplet has no preferred basis"""
    h1 = geometry.honeycomb_lattice().get_hamiltonian(has_spin=False)
    h1.add_haldane(0.2)
    h1.add_sublattice_imbalance(0.1)
    hm = h1.get_multicell()
    rng = np.random.default_rng(7)
    U = [np.linalg.qr(rng.normal(size=(3, 3)) + 1j*rng.normal(size=(3, 3)))[0]
         for s in range(2)]
    V = np.block([[U[0], np.zeros((3, 3))], [np.zeros((3, 3)), U[1]]])
    lift = lambda m: V@np.kron(np.asarray(m), np.eye(3))@V.conj().T
    hm.intra = lift(hm.intra)
    for t in hm.hopping: t.m = lift(t.m)
    return hm, [0, 1, 2], [[0.31, 0.17, 0.], [1/3., 1/3., 0.]]


def _crossing():
    """Two copies with different bandwidths, shifted so that their lower
    bands cross exactly at k=(0.21,0.13) inside the chosen subspace, where
    the individual eigenvectors are not smooth but the projector is"""
    h1 = geometry.honeycomb_lattice().get_hamiltonian(has_spin=False)
    h1.add_haldane(0.2)
    h1.add_sublattice_imbalance(0.2)
    k0 = [0.21, 0.13, 0.]
    c = 0.5*np.linalg.eigvalsh(h1.get_hk_gen()(k0))[0]
    hm = h1.get_multicell()
    lift = lambda m, s: (np.kron(np.asarray(m), np.diag([1., 0.5]))
                         + s*np.kron(np.eye(2), np.diag([0., 1.])))
    hm.intra = lift(hm.intra, c)
    for t in hm.hopping: t.m = lift(t.m, 0.)
    es = np.linalg.eigvalsh(hm.get_hk_gen()(k0))
    assert abs(es[0] - es[1]) < 1e-12 # the premise: an exact crossing
    return hm, [0, 1], [k0]


def _diamond():
    """A three-dimensional two-orbital cell"""
    h = geometry.diamond_lattice_minimal().get_hamiltonian(has_spin=False)
    h.add_sublattice_imbalance(0.5)
    return h, [0], [[0.13, 0.41, 0.27], [0.5, 0.2, 0.7]]


_MODELS = {"rashba": _rashba_exchange, "kane_mele": _kane_mele,
           "threefold": _threefold, "crossing": _crossing,
           "diamond": _diamond}


# --- tests ---------------------------------------------------------------

@pytest.mark.parametrize("gauge", ["atomic", "lattice"])
def test_qgt_chern_matches_wilson_loop(tmp_path, monkeypatch, gauge):
    """Integrating the xy component of the Berry curvature obtained from
    the new sum-over-states quantum geometric tensor over the BZ must
    reproduce the Chern number of the (already tested) independent
    Fukui-Hatsugai-Suzuki Wilson-loop implementation, topology.chern --
    both in the trivial (C=0) and Haldane-gapped (C=+-1) cases, and in
    both gauges, since their Berry curvatures differ by the curl of a
    periodic function.

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
    c_qgt_triv = topology.chern_from_qgt(h_trivial, nk=8, occ_idxs=occ_idxs,
                                         gauge=gauge)
    assert abs(round(c_wilson_triv)) == 0
    assert abs(c_qgt_triv) < 1e-2

    h = g.get_hamiltonian()
    h.add_haldane(0.2)
    c_wilson = topology.chern(h, nk=8)
    c_qgt = topology.chern_from_qgt(h, nk=8, occ_idxs=occ_idxs, gauge=gauge)
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
    trace of the non-Abelian one, which is returned in the orbital basis,
    Q_ij = sum_{m,n in S} |u_m> Q_ij^{mn} <u_n|, so the trace runs over
    the orbitals."""
    h = _haldane_model(has_spin=True, t2=0.2) # 4 bands: 2 exactly spin-degenerate pairs
    for k in ([0.31, 0.17, 0.], [0.0, 0.0, 0.], [0.5, 0.2, 0.]):
        Q_ab = topology.quantum_geometric_tensor(h, k=k, occ_idxs=[0, 1])
        Q_na = topology.quantum_geometric_tensor(h, k=k, occ_idxs=[0, 1],
                                                   non_abelian=True)
        assert Q_na.shape == (2, 2, 4, 4)
        assert np.allclose(np.trace(Q_na, axis1=-2, axis2=-1), Q_ab)


def test_qgt_nonabelian_spin_degenerate_block_diagonal():
    """With no spin-orbit coupling or Zeeman splitting the Haldane
    Hamiltonian is block diagonal in spin and the two spin channels are
    identical copies of the same spinless problem. The orbital-basis
    non-Abelian tensor of the exactly spin-degenerate pair of occupied
    bands must therefore vanish between spin-up and spin-down orbitals,
    and each spin block must equal the spinless tensor. This is a
    statement about the projector, so it holds whatever basis the
    diagonalization picks inside the degenerate pair; in the band basis it
    held only because the solver happened to return spin-pure vectors."""
    h_spinful = _haldane_model(has_spin=True, t2=0.2)
    h_spinless = _haldane_model(has_spin=False, t2=0.2)
    k = [0.31, 0.17, 0.]
    Q_na = topology.quantum_geometric_tensor(h_spinful, k=k, occ_idxs=[0, 1],
                                              non_abelian=True)
    Q_ref = topology.quantum_geometric_tensor(h_spinless, k=k, occ_idxs=[0],
                                              non_abelian=True)
    up, dn = [0, 2], [1, 3] # spin-orbital order: site 0 up/down, site 1
    block = lambda a, b: Q_na[:, :, a][:, :, :, b]
    assert np.allclose(block(up, dn), 0.0, atol=1e-8) # no cross-spin part
    assert np.allclose(block(dn, up), 0.0, atol=1e-8)
    assert np.allclose(block(up, up), Q_ref)
    assert np.allclose(block(dn, dn), Q_ref)


@pytest.mark.parametrize("gauge", ["atomic", "lattice"])
@pytest.mark.parametrize("model", sorted(_MODELS))
def test_qgt_nonabelian_is_the_projector_derivative(model, gauge):
    """The orbital-basis tensor is P d_iP d_jP P, with P the projector on
    the chosen bands, in the Bloch basis of the chosen gauge, and d_i the
    derivative in reduced k. Check it against a finite difference of P
    built directly from the eigenvectors, on models whose subspace is an
    exact Kramers pair at every k (Kane-Mele, high-symmetry points
    included), an exactly threefold multiplet randomly mixed on every site,
    an exact band crossing inside the subspace, a spin-mixed pair, and a
    three-dimensional cell. The finite difference only ever sees P, so no
    choice of basis inside a multiplet enters it."""
    h, occ, ks = _MODELS[model]()
    qmax = 0.
    for k in ks:
        Q = topology.quantum_geometric_tensor(h, k=k, occ_idxs=occ,
                                              non_abelian=True, gauge=gauge)
        Q_fd = _projector_fd(h, k, occ, gauge)
        qmax = max(qmax, np.max(np.abs(Q_fd)))
        assert np.max(np.abs(Q - Q_fd)) < 1e-6*max(1., np.max(np.abs(Q_fd)))
    assert qmax > 1e-3 # the premise: a nonzero tensor somewhere


def test_qgt_threefold_multiplet_is_three_copies():
    """The threefold model is three copies of one two-band Hamiltonian
    rotated by a unitary that acts within each site, so it keeps every
    orbital at its site and the Abelian tensor of the multiplet must be
    exactly three times that of a single copy, in either gauge"""
    hm, occ, ks = _threefold()
    h1 = geometry.honeycomb_lattice().get_hamiltonian(has_spin=False)
    h1.add_haldane(0.2)
    h1.add_sublattice_imbalance(0.1)
    for gauge in ["atomic", "lattice"]:
        for k in ks:
            Q3 = topology.quantum_geometric_tensor(hm, k=k, occ_idxs=occ,
                                                   gauge=gauge)
            Q1 = topology.quantum_geometric_tensor(h1, k=k, occ_idxs=[0],
                                                   gauge=gauge)
            assert np.allclose(Q3, 3.*Q1, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("gauge", ["atomic", "lattice"])
def test_qgt_mesh_is_the_pointwise_tensor(gauge):
    """The mesh is evaluated in batches of k-points, the single-k entry
    point on its own; both must give the same numbers, in the non-Abelian
    orbital basis too, including across a chunk boundary"""
    from pyqula.topologytk import qgt
    h = _haldane_model(has_spin=True, t2=0.2)
    h.add_rashba(0.2)
    hm, orders, hkgen, scale, frac = qgt._multicell_and_orders(h, gauge=gauge)
    ks = [np.random.default_rng(3).random(3)*[1, 1, 0] for _ in range(7)]
    _, Qs = qgt._qgt_over_kpoints(hm, orders, hkgen, ks, [0, 1], True,
                                  1e-8, scale, frac, chunk=3)
    for k, Q in zip(ks, Qs):
        Qk = topology.quantum_geometric_tensor(h, k=k, occ_idxs=[0, 1],
                                               non_abelian=True, gauge=gauge)
        assert np.max(np.abs(Q - Qk)) < 1e-10


def test_qgt_is_c3_symmetric_in_the_atomic_gauge():
    """A crystal symmetry must show up in the quantum geometry: on the
    honeycomb lattice with a Haldane flux and a sublattice mass (symmetric
    under a C3 rotation about a hexagon center) the Cartesian tensor must
    obey Q(R K) = R Q(K) R^T, meaning that the Berry curvature is the same
    at three rotated k-points and the metric rotates with them. That holds
    only with the orbitals at their positions; the lattice gauge, which
    puts both sites at the cell origin, breaks it (the Berry curvature at
    the three points even changes sign), and the test checks that too, so
    that it cannot pass vacuously"""
    h = geometry.honeycomb_lattice().get_hamiltonian(has_spin=False)
    h.add_sublattice_imbalance(0.3)
    h.add_haldane(0.15)
    th = 2*np.pi/3
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    K0 = np.array([0.7, 0.4])
    def violation(gauge):
        Q0 = _cartesian_qgt(h, K0, [0], gauge)
        return max(np.max(np.abs(_cartesian_qgt(h, Rn@K0, [0], gauge)
                                 - Rn@Q0@Rn.T))
                   for Rn in (R, R@R))
    assert violation("atomic") < 1e-12
    assert violation("lattice") > 1e-2


@pytest.mark.parametrize("dim", [2, 3])
def test_qgt_does_not_depend_on_the_unit_cell(dim):
    """Redescribing the same crystal with a doubled unit cell cannot change
    its quantum geometry. The occupied states of the supercell at a
    Cartesian K are those of the primitive cell at K and at K+b1/2 (b1 the
    primitive reciprocal vector along the doubled direction), and with the
    orbitals at their positions the two descriptions differ by a
    k-independent unitary, so the Cartesian tensor of the supercell must be
    exactly the sum of the two primitive ones, pointwise; averaged over the
    BZ this is the statement that the Marzari-Vanderbilt gauge-invariant
    spread (the BZ average of Tr g) of a cell twice as large is twice as
    large. The lattice gauge fails it, which the test also checks."""
    if dim == 2:
        h = geometry.honeycomb_lattice().get_hamiltonian(has_spin=False)
        h.add_sublattice_imbalance(0.3)
        h.add_haldane(0.15)
        h2 = h.get_supercell([2, 1, 1])
        Ks = [np.array([0.3, 0.5]), np.array([-0.2, 1.1])]
    else:
        h, _, _ = _diamond()
        h2 = h.get_supercell([2, 1, 1])
        Ks = [np.array([0.91, 1.32, -0.44])]
    g = h.geometry
    A = np.array([g.a1, g.a2, g.a3][:dim])[:, :dim]
    b1 = 2*np.pi*np.linalg.inv(A).T[0]
    def violation(gauge):
        return max(np.max(np.abs(_cartesian_qgt(h2, K, [0, 1], gauge)
                   - _cartesian_qgt(h, K, [0], gauge)
                   - _cartesian_qgt(h, K + b1/2, [0], gauge))) for K in Ks)
    assert violation("atomic") < 1e-12
    assert violation("lattice") > 1e-2


@pytest.mark.parametrize("valley", [1, -1])
def test_qgt_massive_dirac_point(valley):
    """At the Dirac points of the honeycomb lattice with a sublattice mass m
    the Bloch Hamiltonian is exactly H = v (tau q_x s_x + q_y s_y) + m s_z
    to linear order in q, with v = 3 t a/2 (t=1 the hopping, a=1 the bond
    length), so the lower band has Berry curvature -tau v^2/(2 m^2) and an
    isotropic metric g_xx = g_yy = v^2/(4 m^2), g_xy = 0, in Cartesian
    units (the valley-contrasting Berry curvature of Xiao, Yao and Niu,
    PRL 99, 236809 (2007)); the sign of the curvature alternates between
    the two valleys. H(K) is diagonal here, so both gauges agree at this
    one point, and this test pins the Cartesian scale rather than the
    gauge"""
    m, v = 0.3, 1.5
    h = geometry.honeycomb_lattice().get_hamiltonian(has_spin=False)
    h.add_sublattice_imbalance(m)
    kr = [1/3., 1/3., 0.] if valley == 1 else [2/3., 2/3., 0.]
    g = h.geometry
    A = np.array([g.a1[:2], g.a2[:2]])
    K = 2*np.pi*np.linalg.solve(A, np.array(kr[:2])) # A K/(2 pi) = kr
    Q = _cartesian_qgt(h, K, [0])
    assert np.isclose(-2*Q[0, 1].imag, -valley*v**2/(2*m**2))
    assert np.allclose(Q.real, v**2/(4*m**2)*np.eye(2))


@pytest.mark.parametrize("vw", [(1.0, 0.5), (0.3, 1.0), (1.0, 0.0)])
def test_qgt_ssh_chain_closed_form(vw):
    """SSH chain, bonds of equal length d=1 alternating between v and w
    (lattice constant a=2). With r = min(v,w)/max(v,w) the BZ average of
    the metric of the lower band has a closed form in each gauge, from the
    winding angle phi(k) of the off-diagonal element, g = (1/4) (dphi/dk)^2.
    With both orbitals at the cell origin (lattice gauge) it is
    n^2/4 + r^2/(8 (1-r^2)) per unit of the dimensionless 2 pi-periodic k,
    n the winding number of phi (1 when the intercell bond w is the
    stronger), so it changes when v and w are exchanged, although that is
    the same chain with the cell shifted by one site. With the orbitals at
    +-d/2 (atomic gauge) the phase picks up k d, and the Marzari-Vanderbilt
    spread of the Wannier function becomes Omega_I = (d^2/4) (1+r^2)/(1-r^2)
    whichever bond is the stronger, which goes to (d/2)^2 in the dimerized
    limit r=0, the spread of an orbital shared by two sites a distance d
    apart, and diverges as the gap closes"""
    v, w = vw
    gc = geometry.chain().get_supercell(2) # sites at x=-0.5,0.5, a=2
    xc = np.mean(np.array(gc.r)[:, 0])
    def hopping(r1, r2):
        if abs(np.linalg.norm(r1 - r2) - 1.) > 1e-5: return 0.
        return v if abs((r1[0] + r2[0])/2. - xc) < 0.6 else w
    h = gc.get_hamiltonian(fun=hopping, has_spin=False)
    r = min(v, w)/max(v, w)
    a = 2.
    _, Qs = topology.quantum_geometric_tensor_mesh(h, nk=400, occ_idxs=[0],
                                                   gauge="lattice")
    g_red = np.mean(Qs[:, 0, 0].real) # reduced k, g_red = (2 pi)^2 g_k
    n = 1 if w > v else 0 # winding number of the off-diagonal element
    assert np.isclose(g_red/(2*np.pi)**2, n**2/4. + r**2/(8*(1 - r**2)),
                      atol=1e-10)
    _, Qs = topology.quantum_geometric_tensor_mesh(h, nk=400, occ_idxs=[0])
    spread = np.mean(Qs[:, 0, 0].real)*(a/(2*np.pi))**2 # Cartesian
    assert np.isclose(spread, (a**2/16.)*(1 + r**2)/(1 - r**2), atol=1e-10)


def test_qgt_kane_mele_spin_chern_numbers():
    """With inversion and time reversal every band of the Kane-Mele model
    is an exact Kramers pair, so the occupied pair has no preferred basis
    anywhere in the BZ, and its Abelian Berry curvature vanishes
    identically. The non-Abelian tensor still carries the spin-resolved
    information: S_z is conserved, the projector splits into spin blocks,
    and the trace of the non-Abelian Berry curvature over the spin-up
    orbitals integrates to C_up = +1 and over the spin-down ones to
    C_dn = -1, the quantum spin Hall state of Kane and Mele, PRL 95,
    226801 (2005)"""
    from pyqula.topologytk import qgt
    h = geometry.honeycomb_lattice().get_hamiltonian(has_spin=True)
    h.add_kane_mele(0.1)
    nk = 24
    ks, Qs = topology.quantum_geometric_tensor_mesh(h, nk=nk, occ_idxs=[0, 1],
                                                    non_abelian=True)
    F = qgt.berry_curvature_from_qgt(Qs, non_abelian=True)[:, 0, 1]
    up, dn = [0, 2], [1, 3]
    chern = lambda o: np.sum(np.trace(F[:, o][:, :, o], axis1=1,
                                      axis2=2)).real/(nk*nk*2*np.pi)
    assert np.isclose(chern(up), 1., atol=1e-3)
    assert np.isclose(chern(dn), -1., atol=1e-3)
    assert np.max(np.abs(np.trace(F, axis1=1, axis2=2))) < 1e-10


def test_qgt_matches_analytic_two_band_formula_in_both_gauges():
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
    read off the Bloch Hamiltonian of each gauge, built here from
    h.get_hk_gen() and the orbital positions, and differentiated with a
    plain central finite difference, entirely independent of qgt.py's
    exact analytic multicell derivative."""
    h = _haldane_model(t2=0.2)
    dk = 1e-5
    pauli = [np.array([[0, 1], [1, 0]], dtype=complex),
             np.array([[0, -1j], [1j, 0]], dtype=complex),
             np.array([[1, 0], [0, -1]], dtype=complex)]
    for gauge in ["atomic", "lattice"]:
        hk = _bloch_generator(h, gauge)
        def dhat(k):
            d = np.array([np.real(np.trace(hk(k)@s))/2. for s in pauli])
            return d/np.linalg.norm(d)
        for k in ([0.1, 0.2, 0.], [0.31, 0.17, 0.], [0.05, 0.4, 0.]):
            k = np.array(k, dtype=float)
            ex, ey = np.array([1., 0., 0.]), np.array([0., 1., 0.])
            dx = (dhat(k+dk*ex) - dhat(k-dk*ex))/(2*dk)
            dy = (dhat(k+dk*ey) - dhat(k-dk*ey))/(2*dk)
            g_analytic = 0.25*np.array([[np.dot(dx, dx), np.dot(dx, dy)],
                                         [np.dot(dy, dx), np.dot(dy, dy)]])
            omega_analytic = 0.5*np.dot(dhat(k), np.cross(dx, dy))

            Q = topology.quantum_geometric_tensor(h, k=k, occ_idxs=[0],
                                                  gauge=gauge)
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
    assert np.allclose(np.trace(omega_na, axis1=-2, axis2=-1), omega_ab)
    assert np.allclose(np.trace(g_na, axis1=-2, axis2=-1), g_ab)


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
    with pytest.raises(ValueError):
        topology.quantum_geometric_tensor(h, k=[0.31, 0.17, 0.], occ_idxs=[0])


def test_qgt_unknown_gauge_lists_the_accepted_ones():
    h = _haldane_model(t2=0.2)
    with pytest.raises(ValueError, match="atomic"):
        topology.quantum_geometric_tensor(h, k=[0.31, 0.17, 0.],
                                          gauge="cell")
