"""The quantum geometric tensor in Cartesian momentum components,
coordinates="cartesian" in topology.quantum_geometric_tensor and its path
and mesh versions.

The tensor is computed along the reduced momenta k_i (k = sum_i k_i b_i),
and since the reciprocal lattice vectors of the honeycomb lattice are not
orthogonal, the trace of the reduced metric is not g_xx+g_yy: it differs
between the three M points, which a C3 rotation maps into each other
(53.3 at (1/2,0) and (0,1/2), 36.5 at (1/2,1/2) for the spinful Haldane
model below). The Cartesian tensor is J Q J^T with J[a,i] = a_i[a]/(2 pi),
and it is checked here against a finite difference of the projector along
the Cartesian momentum, built without any of qgt.py's code, against the
point-group symmetry, and against the massive Dirac cone at K, where the
bound Tr g >= |Omega_xy| of Roy, PRB 90, 165139 (2014), Eq. (9), is
saturated.
"""
import numpy as np
import pytest

from pyqula import algebra
from pyqula import geometry
from pyqula import topology


def _haldane(has_spin=True):
    """Haldane model with t2=0.2, a gap [-0.9,0.9] at K, with the Fermi
    level moved to a safe 0.3 inside it"""
    h = geometry.honeycomb_lattice().get_hamiltonian(has_spin=has_spin)
    h.add_haldane(0.2)
    h.shift_fermi(0.3)
    return h


def _lattice(h):
    """Rows a_i of the periodic lattice vectors, (dim,3)"""
    g = h.geometry
    return np.array([g.a1, g.a2, g.a3][:h.dimensionality], dtype=float)


def _reduced(h, K):
    """Reduced momentum k_i = a_i.K/(2 pi) of a Cartesian momentum K"""
    kr = _lattice(h)@np.array(K, dtype=float)/(2*np.pi)
    return list(kr) + [0.]*(3 - len(kr))


def _atomic_projector(h, occ):
    """K -> projector on the bands occ at the Cartesian momentum K, with
    the Bloch Hamiltonian in the atomic gauge, D^dag H(k) D with
    D = diag(exp(2 pi i k.f_j)), f the fractional orbital positions"""
    g = h.geometry
    A = _lattice(h)
    f = np.linalg.lstsq(A.T, np.array(g.r).T, rcond=None)[0].T
    f = np.repeat(f, h.intra.shape[0]//len(g.r), axis=0)
    hk = h.get_hk_gen()
    def P(K):
        k = np.array(_reduced(h, K))
        ph = np.exp(2j*np.pi*(f@k[:h.dimensionality]))
        m = np.asarray(algebra.todense(hk(k)))
        w = np.linalg.eigh(np.conj(ph)[:, None]*m*ph[None, :])[1][:, occ]
        return w@w.conj().T
    return P


def _projector_fd_cartesian(h, K, occ, dK=1e-5):
    """P d_aP d_bP P with d_a the derivative along the Cartesian momentum
    component a=x,y,z, by a central finite difference of the projector"""
    P = _atomic_projector(h, occ)
    K = np.array(K, dtype=float)
    dP = [(P(K + dK*e) - P(K - dK*e))/(2*dK) for e in np.eye(3)]
    P0 = P(K)
    return np.array([[P0@dP[a]@dP[b]@P0 for b in range(3)]
                     for a in range(3)])


def _ssh():
    """A dimerized chain, one dimension"""
    gc = geometry.chain().get_supercell(2)
    xc = np.mean(np.array(gc.r)[:, 0])
    def hopping(r1, r2):
        if abs(np.linalg.norm(r1 - r2) - 1.) > 1e-5: return 0.
        return 1.0 if abs((r1[0] + r2[0])/2. - xc) < 0.6 else 0.5
    return gc.get_hamiltonian(fun=hopping, has_spin=False)


def _rashba():
    h = _haldane()
    h.add_rashba(0.3)
    h.add_exchange([0.1, 0.2, 0.1])
    return h


def _diamond():
    h = geometry.diamond_lattice_minimal().get_hamiltonian(has_spin=False)
    h.add_sublattice_imbalance(0.5)
    return h


_MODELS = { # model, occupied bands, Cartesian momenta
    "ssh": (_ssh, [0], [[0.37, 0., 0.], [1.3, 0., 0.]]),
    "haldane": (_haldane, [0, 1], [[0.7, 0.4, 0.], [-1.1, 0.3, 0.]]),
    "rashba": (_rashba, [0, 1], [[0.7, 0.4, 0.], [0.2, -0.9, 0.]]),
    "diamond": (_diamond, [0], [[0.91, 1.32, -0.44], [0.3, -0.2, 0.7]]),
}


@pytest.mark.parametrize("model", sorted(_MODELS))
def test_cartesian_tensor_is_the_projector_derivative(model):
    """With coordinates="cartesian" the non-Abelian tensor is P d_aP d_bP P
    with d_a the derivative along the Cartesian momentum, a 3x3 array of
    matrices whatever the dimensionality, whose components along a
    direction with no periodicity vanish (the finite difference along it
    does not change the reduced momentum at all)"""
    build, occ, Ks = _MODELS[model]
    h = build()
    qmax = 0.
    for K in Ks:
        Q = topology.quantum_geometric_tensor(h, k=_reduced(h, K),
                occ_idxs=occ, non_abelian=True, coordinates="cartesian")
        Q_fd = _projector_fd_cartesian(h, K, occ)
        assert Q.shape == Q_fd.shape
        qmax = max(qmax, np.max(np.abs(Q_fd)))
        assert np.max(np.abs(Q - Q_fd)) < 1e-6*max(1., np.max(np.abs(Q_fd)))
        Qab = topology.quantum_geometric_tensor(h, k=_reduced(h, K),
                occ_idxs=occ, coordinates="cartesian")
        assert np.allclose(Qab, np.trace(Q, axis1=-2, axis2=-1))
    assert qmax > 1e-3 # the premise: a nonzero tensor somewhere


def test_cartesian_trace_is_the_same_at_the_three_m_points():
    """The three M points of the honeycomb lattice are related by a C3
    rotation, so the Cartesian trace of the metric, and the Berry
    curvature, must be the same at the three of them; the trace of the
    reduced tensor is not, which is the premise of the test"""
    h = _haldane()
    Ms = [[0.5, 0., 0.], [0., 0.5, 0.], [0.5, 0.5, 0.]]
    red = [np.trace(topology.quantum_metric(h, k=k, occ_idxs=[0, 1]))
           for k in Ms]
    Qs = [topology.quantum_geometric_tensor(h, k=k, occ_idxs=[0, 1],
            coordinates="cartesian") for k in Ms]
    tr = [np.trace(topology.quantum_metric_from_qgt(Q)) for Q in Qs]
    om = [topology.berry_curvature_from_qgt(Q)[0, 1] for Q in Qs]
    assert max(red) - min(red) > 1. # the premise: reduced trace differs
    assert max(tr) - min(tr) < 1e-10
    assert max(om) - min(om) < 1e-10


def test_cartesian_tensor_rotates_with_the_crystal():
    """A C3 rotation R of the momentum rotates the Cartesian tensor,
    Q(RK) = R Q(K) R^T, on the in-plane block, with the out-of-plane
    components zero"""
    h = geometry.honeycomb_lattice().get_hamiltonian(has_spin=False)
    h.add_sublattice_imbalance(0.3)
    h.add_haldane(0.15)
    th = 2*np.pi/3
    R = np.array([[np.cos(th), -np.sin(th), 0.],
                  [np.sin(th), np.cos(th), 0.], [0., 0., 1.]])
    K0 = np.array([0.7, 0.4, 0.])
    Q = lambda K: topology.quantum_geometric_tensor(h, k=_reduced(h, K),
            occ_idxs=[0], coordinates="cartesian")
    Q0 = Q(K0)
    assert np.allclose(Q0[2, :], 0.) and np.allclose(Q0[:, 2], 0.)
    for Rn in (R, R@R):
        assert np.max(np.abs(Q(Rn@K0) - Rn@Q0@Rn.T)) < 1e-12


def test_trace_bound_is_saturated_at_the_dirac_point():
    """Tr g >= |Omega_xy| at every k (Roy, PRB 90, 165139 (2014),
    Eq. (9)), with the equality where the metric is isotropic: at K the
    Haldane model is a massive Dirac cone with velocity v=3/2 and mass
    0.9, where each spin gives Tr g = |Omega_xy| = v^2/(2 m^2), and so
    Tr g = |Omega_xy| = 2.7778 for the spin-degenerate pair. At M the
    bound holds strictly"""
    h = _haldane()
    ks, Qs = topology.quantum_geometric_tensor_mesh(h, nk=12,
            occ_idxs=[0, 1], coordinates="cartesian")
    tr = np.trace(topology.quantum_metric_from_qgt(Qs), axis1=1, axis2=2)
    om = topology.berry_curvature_from_qgt(Qs)[:, 0, 1]
    assert np.all(tr >= np.abs(om) - 1e-10)
    QK = topology.quantum_geometric_tensor(h, k=[1/3., 1/3., 0.],
            occ_idxs=[0, 1], coordinates="cartesian")
    exact = 2*1.5**2/(2*0.9**2)
    assert np.isclose(np.trace(QK.real), exact)
    assert np.isclose(topology.berry_curvature_from_qgt(QK)[0, 1], exact)
    QM = topology.quantum_geometric_tensor(h, k=[0.5, 0., 0.],
            occ_idxs=[0, 1], coordinates="cartesian")
    assert np.trace(QM.real) > abs(topology.berry_curvature_from_qgt(QM)[0, 1]) + 0.1


def test_cartesian_curvature_integrates_to_the_chern_number():
    """The Cartesian Berry curvature integrated over the Brillouin zone,
    of area (2 pi)^2/|a1 x a2|, divided by 2 pi, is the Chern number, 2
    for the spinful Haldane model; the mesh and path versions give the
    same tensor as the single k-point one"""
    h = _haldane()
    nk = 24
    ks, Qs = topology.quantum_geometric_tensor_mesh(h, nk=nk,
            occ_idxs=[0, 1], coordinates="cartesian")
    A = _lattice(h)
    area = (2*np.pi)**2/np.linalg.norm(np.cross(A[0], A[1]))
    om = topology.berry_curvature_from_qgt(Qs)[:, 0, 1]
    assert np.isclose(np.sum(om)*area/(nk*nk)/(2*np.pi), 2., atol=1e-6)
    for i in (0, 37, 311):
        Qk = topology.quantum_geometric_tensor(h, k=ks[i], occ_idxs=[0, 1],
                coordinates="cartesian")
        assert np.allclose(Qs[i], Qk)
    kpath = [[0., 0., 0.], [0.5, 0., 0.], [1/3., 1/3., 0.]]
    inds, gp, op = topology.quantum_geometric_tensor_path(h, kpath=kpath,
            nk=30, occ_idxs=[0, 1], coordinates="cartesian")
    assert gp.shape[1:] == (3, 3) and op.shape[1:] == (3, 3)


def test_unknown_coordinates_lists_the_accepted_ones():
    h = _haldane()
    with pytest.raises(ValueError, match="cartesian"):
        topology.quantum_geometric_tensor(h, k=[0.31, 0.17, 0.],
                                          coordinates="polar")
