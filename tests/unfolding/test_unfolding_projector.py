"""Phase-correctness of the unfolding projector itself.

The invariance tests in test_unfolding_defects.py check a sum rule
(sum of the weights over all bands at a fixed kpoint is the number of
orbitals), and that rule holds for *any* phase convention -- a wrong
k-rescaling, a wrong replica ordering and a wrong inverse supercell
matrix all satisfy it.  The tests here pin the phases instead, by
brute-force enumerating the primal kpoints that fold onto a given
supercell kpoint and checking that the operator puts the weight on the
right bands.
"""

import numpy as np
import pytest
from scipy.linalg import eigh

from pyqula import geometry
from pyqula.unfolding import bloch_projector


def folded_kpoints(Minv, k_super, dim=2, nj=4):
    """Every primal kpoint (mod the primal reciprocal lattice) that folds
    onto k_super, enumerated by brute force"""
    rng = range(-nj, nj + 1)
    js = [np.array([j0, j1, j2], dtype=float)
          for j0 in rng
          for j1 in (rng if dim > 1 else [0])
          for j2 in (rng if dim > 2 else [0])]
    uniq = []
    for j in js:
        c = (Minv @ (k_super + j)) % 1.0
        if not any(np.allclose(c, u, atol=1e-8) for u in uniq):
            uniq.append(c)
    return uniq


def check_against_direct_diagonalization(g0, M, k_super, has_spin=False,
                                         diagonal_builder=False):
    """The union of the primal spectra at the folded kpoints must be the
    supercell spectrum (pure Bloch folding), and the unfold operator must
    give the full weight |det M| to exactly those supercell states whose
    energy comes from the j=0 representative k0=Minv@k_super.

    diagonal_builder asks for the supercell to be built by passing a plain
    (n1,n2,n3) size rather than the matrix, which is a different code path
    in Geometry.get_supercell (supercell1d/2d/3d rather than the general
    non_orthogonal_supercell) and must give the same answer."""
    M = np.array(M)
    Minv = np.linalg.inv(M.astype(float))
    N = int(round(abs(np.linalg.det(M))))
    if diagonal_builder:
        nsuper = [M[i][i] for i in range(3)]
        assert np.allclose(M, np.diag(nsuper))
        g = g0.get_supercell(nsuper[:max(g0.dimensionality, 1)],
                             store_primal=True)
    else:
        g = g0.get_supercell(M, store_primal=True)
    h = g.get_hamiltonian(has_spin=has_spin)
    h0 = g0.get_hamiltonian(has_spin=has_spin)
    op = bloch_projector(h)
    Es, Vs = eigh(h.get_hk_gen()(k_super))
    hk0 = h0.get_hk_gen()
    uniq = folded_kpoints(Minv, k_super, dim=g0.dimensionality)
    assert len(uniq) == N
    primal = np.sort(np.concatenate([eigh(hk0(k0))[0] for k0 in uniq]))
    assert np.allclose(np.sort(Es), primal, atol=1e-8)
    ref = eigh(hk0(Minv @ k_super))[0]  # the j=0 representative
    ws = np.array([np.abs(op.m(Vs[:, i], k_super).dot(np.conjugate(Vs[:, i])))
                   for i in range(Vs.shape[1])])
    # compare per degenerate block rather than per eigenvector: a lattice
    # with a flat band (kagome) has the same energy at every folded kpoint,
    # so eigh returns an arbitrary mixture of the degenerate states and no
    # single one of them carries the full weight -- their sum does
    for e in np.unique(np.round(Es, 8)):
        block = np.isclose(Es, e, atol=1e-8)
        multiplicity = int(np.sum(np.isclose(ref, e, atol=1e-8)))
        assert np.isclose(np.sum(ws[block]), N * multiplicity, atol=1e-6)


@pytest.mark.parametrize("name,M", [
    ("chain", [[3, 0, 0], [0, 1, 0], [0, 0, 1]]),
    ("honeycomb_lattice", [[3, 0, 0], [0, 2, 0], [0, 0, 1]]),
    ("square_lattice", [[2, 0, 0], [0, 3, 0], [0, 0, 1]]),
    ("kagome_lattice", [[2, 0, 0], [0, 2, 0], [0, 0, 1]]),
])
@pytest.mark.parametrize("diagonal_builder", [True, False])
def test_diagonal_supercell_puts_the_weight_on_the_right_bands(name, M,
                                                               diagonal_builder):
    """The diagonal-supercell projector had no phase coverage at all: the
    two tests that exercised it only checked the sum rule.  Both ways of
    asking for the same supercell (a size or the matrix) are covered."""
    g0 = getattr(geometry, name)()
    k = np.array([0.37, 0.81, 0.]) if g0.dimensionality > 1 else np.array([0.37, 0., 0.])
    check_against_direct_diagonalization(g0, M, k,
                                         diagonal_builder=diagonal_builder)


@pytest.mark.parametrize("M", [
    [[2, 1, 0], [0, 1, 0], [0, 0, 1]],
    [[3, 1, 0], [0, 1, 0], [0, 0, 1]],
    [[2, 1, 0], [1, 2, 0], [0, 0, 1]],
])
def test_non_diagonal_supercell_puts_the_weight_on_the_right_bands(M):
    check_against_direct_diagonalization(geometry.honeycomb_lattice(), M,
                                         np.array([0.37, 0.81, 0.]))


def test_full_weight_is_the_replica_count_not_the_orbital_count():
    """A fully unfolded band carries weight |det M| (the number of primal
    replicas in the supercell), which is independent of how many orbitals
    the primal cell has.  A honeycomb lattice has n0=2 orbitals, so a
    det(M)=2 supercell cannot tell the two constants apart."""
    g0 = geometry.honeycomb_lattice()  # two orbitals per primal cell
    k = np.array([0.37, 0.81, 0.])
    for n in [2, 3, 4, 5]:
        M = np.array([[n, 1, 0], [0, 1, 0], [0, 0, 1]])
        g = g0.get_supercell(M, store_primal=True)
        h = g.get_hamiltonian(has_spin=False)
        op = bloch_projector(h)
        Es, Vs = eigh(h.get_hk_gen()(k))
        ws = [np.abs(op.m(Vs[:, i], k).dot(np.conjugate(Vs[:, i])))
              for i in range(len(Es))]
        assert np.isclose(np.max(ws), n, atol=1e-6)


def test_spinful_and_nambu_supercells_unfold():
    """norb_factor is read off the Hamiltonian, so the projector must work
    for one, two and four orbitals per site alike"""
    g0 = geometry.square_lattice()
    M = [[2, 0, 0], [0, 2, 0], [0, 0, 1]]
    k = np.array([0.23, 0.41, 0.])
    for setup in ["spinless", "spinful", "nambu"]:
        g = g0.get_supercell(M, store_primal=True)
        h = g.get_hamiltonian(has_spin=(setup != "spinless"))
        if setup == "nambu": h.setup_nambu_spinor()
        op = bloch_projector(h)
        Es, Vs = eigh(h.get_hk_gen()(k))
        ws = [np.abs(op.m(Vs[:, i], k).dot(np.conjugate(Vs[:, i])))
              for i in range(len(Es))]
        assert np.isclose(np.max(ws), 4, atol=1e-6)  # det(M)=4 replicas
        assert np.isclose(np.sum(ws), h.intra.shape[0], atol=1e-6)  # sum rule


def test_three_dimensional_supercell_unfolds():
    """The projector exp(2i.pi.n.Minv@k) is dimension agnostic, so a 3d
    supercell unfolds exactly like a 1d or 2d one"""
    g0 = geometry.cubic_lattice()
    check_against_direct_diagonalization(
            g0, [[2, 0, 0], [0, 2, 0], [0, 0, 2]], np.array([0.31, 0.47, 0.19]),
            diagonal_builder=True)


def test_three_dimensional_non_diagonal_supercell_unfolds():
    g0 = geometry.cubic_lattice()
    check_against_direct_diagonalization(
            g0, [[2, 1, 0], [0, 1, 0], [0, 0, 2]], np.array([0.31, 0.47, 0.19]))


def test_unfolding_operator_is_linear():
    """Unfolding is the linear operator O = P^dagger P, P being the matrix
    of Bloch phases of the primal cell. It used to be applied in the form
    v -> v*|Pv|^2, which gives the same expectation value on a normalized
    state but is cubic in v, so consumers that branch on Operator.linear
    (to decide whether they may apply it to a Green's function rather than
    to one eigenvector at a time) could not use it."""
    g0 = geometry.triangular_lattice()
    g = g0.get_supercell(2, store_primal=True)
    h = g.get_hamiltonian(has_spin=False)
    op = bloch_projector(h)
    assert op.linear is True
    k = np.array([0.1, 0.2, 0.])
    n = h.intra.shape[0]
    v = np.random.random(n) + 1j * np.random.random(n)
    w = np.random.random(n) + 1j * np.random.random(n)
    assert np.allclose(op.m(v + w, k), op.m(v, k) + op.m(w, k))
    assert np.allclose(op.m(3 * v, k), 3 * op.m(v, k))


def test_unfolding_operator_is_a_hermitian_projector_times_the_replica_count():
    """O = P^dagger P is N_replicas times the orthogonal projector onto the
    n0-dimensional space of primal-cell Bloch states, which is why a fully
    unfolded band carries weight N_replicas and the weights of all the
    bands at one kpoint add up to the number of orbitals"""
    from pyqula.unfolding import get_supercell_map, bloch_phase_matrix
    g0 = geometry.honeycomb_lattice()
    nrep = 3 * 3
    g = g0.get_supercell([3, 3], store_primal=True)
    h = g.get_hamiltonian(has_spin=False)
    n0 = len(g0.r)
    k = np.array([0.37, 0.81, 0.])
    M, replicas, primal = get_supercell_map(h.geometry, g0)
    P = bloch_phase_matrix(n0, replicas, primal, 1, M)(k).conjugate().toarray()
    O = np.conjugate(P).T @ P
    op = bloch_projector(h)
    v = np.random.random(h.intra.shape[0]) + 1j * np.random.random(h.intra.shape[0])
    assert np.allclose(op.m(v, k), O @ v)           # the operator really is O
    assert np.allclose(O, np.conjugate(O).T)        # hermitian
    assert np.allclose(O @ O, nrep * O)             # nrep times a projector
    assert np.isclose(np.trace(O).real, h.intra.shape[0])
    ev = np.linalg.eigvalsh(O)
    assert np.isclose(np.sum(ev > 1e-8), n0)        # rank n0
    assert np.allclose(ev[ev > 1e-8], nrep)


def test_green_function_consumers_accept_the_unfold_operator():
    """Both of these used to raise: the first because the operator was not
    linear and singlefs refused to hand a Green's function to it, the
    second because kdos never resolved the operator name"""
    g0 = geometry.triangular_lattice()
    g = g0.get_supercell(2, store_primal=True)
    h = g.get_hamiltonian(has_spin=False)
    out = h.get_fermi_surface(nk=6, mode="full", operator="unfold", write=False)
    assert np.all(np.isfinite(out[2]))
    h.get_kdos_bands(operator="unfold", mode="green", nk=3, delta=0.1,
                     energies=np.linspace(-1., 1., 3))


def test_unfolding_operator_requires_a_kpoint():
    """The projector is k-dependent, so there is no sensible default"""
    g0 = geometry.chain()
    g = g0.get_supercell(3, store_primal=True)
    h = g.get_hamiltonian(has_spin=False)
    op = bloch_projector(h)
    v = np.zeros(h.intra.shape[0], dtype=np.complex128) ; v[0] = 1.
    with pytest.raises(ValueError):
        op.m(v)


def test_supercell_of_a_supercell_unfolds():
    """The replica bookkeeping used to be set only by the general (matrix)
    builder and then ride along through Geometry.copy(), so a diagonal
    supercell of a matrix-built supercell carried a replica array of the
    wrong length and the projector died on a shape mismatch"""
    g0 = geometry.honeycomb_lattice()
    g1 = g0.get_supercell([[2, 1, 0], [0, 1, 0], [0, 0, 1]])
    g2 = g1.get_supercell([2, 2, 1], store_primal=True)
    assert len(g2.supercell_replica) == len(g2.r)
    h = g2.get_hamiltonian(has_spin=False)
    (ks, es, ds) = h.get_bands(operator="unfold", nk=5)
    assert np.isclose(np.max(ds), 4., atol=1e-6)  # 2x2 replicas of g1
    assert np.isclose(np.sum(ds), 5 * h.intra.shape[0], atol=1e-6)


def test_projector_is_built_once_per_kpoint():
    """get_bands applies the operator once per band per kpoint, so the
    k-dependent phase matrix must be cached rather than rebuilt for every
    eigenvector"""
    import pyqula.unfolding as U
    g0 = geometry.honeycomb_lattice()
    n = 4
    g = g0.get_supercell(n, store_primal=True)
    h = g.get_hamiltonian(has_spin=False)
    calls = [0]
    original = U.bloch_phase_matrix
    def counting(*args, **kwargs):
        f = original(*args, **kwargs)
        def wrapped(k):
            calls[0] += 1
            return f(k)
        return wrapped
    U.bloch_phase_matrix = counting
    try:
        kpath = np.array(g.get_kpath(nk=7)) * n
        h.get_bands(operator="unfold", kpath=kpath, write=False)
    finally:
        U.bloch_phase_matrix = original
    assert calls[0] == len(kpath)  # not len(kpath)*h.intra.shape[0]
