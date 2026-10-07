"""The block Chebyshev recursion behind the KPM mean field
(kpmtk/pairmomentsjax.py).

It replaced one numba recursion per density-matrix pair and per k-point
with one recursion on a block of every starting column, all k-points at
once, on the CPU or the GPU, in single or double precision. The oracles are
the per-pair numba kernel it replaced and numba's batched trace, which
share no code with it, and the precision of the recursion, which must not
move the density matrix by more than its roundoff.
"""
import warnings

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from pyqula import geometry, gpu, kpm
from pyqula.kpmtk import densitymatrix_kpm as D
from pyqula.kpmtk import pairmomentsjax
from pyqula.kpmtk.kpmnumba import kpm_moments_ij


def _noncollinear_chain():
    """Two sites, spinful, with Rashba coupling and a canted exchange
    field, so every spin block of H(k) is complex and occupied. With
    first-neighbor hopping only, the hopping block of H(k) vanishes at
    k=1/4, where the CSR form of H(k) drops it, which the ELL pattern has
    to survive"""
    h = geometry.chain().get_supercell(2).get_hamiltonian()
    h.add_rashba(0.3)
    h.add_exchange([[0.4, 0.1, 0.2], [-0.2, 0.3, 0.1]])
    h.shift_fermi(0.4)
    return h


def _scaled_hks(h, nk):
    hk = h.get_hk_gen()
    ks = [list(k) for k in h.geometry.get_kmesh(nk=nk)]
    scale = D._estimate_kpm_scale(hk, ks)
    return [csr_matrix(hk(k))/scale for k in ks]


def test_pair_values_match_the_per_pair_numba_recursion():
    """Every pair, every k: the block recursion contracted with arbitrary
    coefficients equals the same contraction of the moments the old
    per-pair kernel computes on its own"""
    ms = _scaled_hks(_noncollinear_chain(), nk=4)
    n = ms[0].shape[0]
    rng = np.random.default_rng(1)
    pairs = [(i, j) for i in range(n) for j in range(i, n)]
    coef = rng.normal(size=60)
    vals, mumax = pairmomentsjax.pair_values(ms, pairs, coef)
    for ik, m in enumerate(ms):
        for p, (i, j) in enumerate(pairs):
            # kpm_moments_ij(m,i=a,j=b) gives <e_b|T_n|e_a>
            mus = kpm_moments_ij(m, i=j, j=i, n=30)
            assert abs(vals[ik, p] - coef @ mus) < 1e-12
    assert mumax <= 1. + 1e-12


@pytest.mark.parametrize("nm", [80, 81])
def test_trace_moments_match_numba_full_trace(nm):
    """The device trace, doubled in double precision, for an even and an
    odd number of moments"""
    ms = _scaled_hks(_noncollinear_chain(), nk=4)
    mt, _ = pairmomentsjax.trace_moments(ms, nm)
    for ik, m in enumerate(ms):
        ref = kpm.full_trace(m, n=(nm + 1)//2)[:nm]
        assert np.max(np.abs(mt[ik] - ref)) < 1e-12


def test_chunking_does_not_change_the_result(monkeypatch):
    """Splitting the starting columns and the k-points into several calls
    (forced here with a tiny block budget) is bookkeeping only"""
    ms = _scaled_hks(_noncollinear_chain(), nk=6)
    n = ms[0].shape[0]
    pairs = [(i, j) for i in range(n) for j in range(i, n)]
    coef = np.linspace(1., -1., 50)
    whole, _ = pairmomentsjax.pair_values(ms, pairs, coef)
    # two starting columns per call, one k-point per call
    monkeypatch.setattr(pairmomentsjax, "_MAX_BLOCK", 3*n*2)
    pieces, _ = pairmomentsjax.pair_values(ms, pairs, coef)
    assert np.max(np.abs(whole - pieces)) < 1e-13


def _bdg_density_matrices(kpm_prec):
    h = geometry.triangular_lattice().get_hamiltonian()
    h.add_rashba(0.3)
    h.add_exchange([0.2, 0.1, 0.])
    h.setup_nambu_spinor()
    h.add_swave(0.15)
    h.shift_fermi(0.5)
    n = h.intra.shape[0]//2
    v = {(0, 0, 0): np.zeros((n, n), dtype=np.complex128)}
    v[(0, 0, 0)][0, 1] = -2.
    needed = D.required_elements_eh(v)
    dm = D.get_dm_kpm(h, v, nk=6, npol=300, kpm_prec=kpm_prec)[(0, 0, 0)]
    return dm, needed


def test_single_precision_moves_the_density_matrix_by_roundoff_only():
    """Pairing, spin-flip and occupation entries of a Nambu Hamiltonian:
    the complex64 recursion agrees with complex128 far below the KPM
    truncation error (about 1e-8 against 1e-3 at this npol)"""
    dd, needed = _bdg_density_matrices("double")
    ds, _ = _bdg_density_matrices("single")
    assert max(abs(dd[i, j]) for (_, i, j) in needed) > 0.05
    assert max(abs(dd[i, j] - ds[i, j]) for (_, i, j) in needed) < 1e-6


def test_the_device_fermi_search_agrees_with_the_cpu_one():
    """With the switch on, the trace of the Fermi search goes through the
    block kernel instead of numba's full_trace (on jax's CPU backend where
    there is no card), and must find the same Fermi level"""
    h = _noncollinear_chain()
    e_cpu = D.get_fermi4filling_kpm(h, 0.3, nk=10, npol=150)
    was = gpu.get_gpu()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore") # no GPU on this machine
            gpu.set_gpu(True)
        e_dev = D.get_fermi4filling_kpm(h, 0.3, nk=10, npol=150,
                kpm_prec="double")
        assert D.resolve_kpm_prec(None) == "single"
    finally:
        gpu.set_gpu(was)
    assert abs(e_dev - e_cpu) < 1e-10


def test_an_unknown_precision_lists_the_accepted_ones():
    with pytest.raises(ValueError, match="'double', 'single'"):
        D.resolve_kpm_prec("half")


def test_the_scale_guard_reads_the_precision_of_the_moments():
    """A complex64 recursion drifts by about n*eps, so a moment of a state
    at zero energy, where |T_2n(0)|=1, can exceed one by more than the
    double precision tolerance with a scale that is right"""
    mus = [1. + 5e-6]
    D._check_scale_covers_spectrum(mus, 1., False, kpm_prec="single")
    with pytest.raises(ValueError, match="diverge"):
        D._check_scale_covers_spectrum(mus, 1., False, kpm_prec="double")


def test_the_jax_solvers_refuse_kpm_prec():
    """use_jax=True diagonalizes, so a KPM precision would be ignored,
    and an ignored keyword is refused like the other KPM-only ones"""
    h = geometry.chain().get_hamiltonian()
    with pytest.raises(NotImplementedError, match="kpm_prec"):
        h.get_mean_field_hamiltonian(U=1., use_jax=True, kpm_prec="single",
                nk=4)


def _hubbard_island(rashba):
    """A spinful square island, complex with Rashba coupling and real
    without it, scaled into [-1,1], and the pairs of an onsite Hubbard
    interaction plus a first-neighbor one along x"""
    g = geometry.square_lattice().get_supercell(5)
    g.dimensionality = 0
    h = g.get_hamiltonian(has_spin=True, is_sparse=True)
    if rashba: h.add_rashba(0.3)
    h.add_exchange([0., 0., 0.3])
    ms = _scaled_hks(h, nk=1)
    s = np.arange(ms[0].shape[0]//2)
    pairs = np.concatenate([np.stack([2*s, 2*s], 1),
        np.stack([2*s+1, 2*s+1], 1), np.stack([2*s, 2*s+1], 1),
        np.stack([2*s, 2*s+2], 1)[:-1]])
    return ms, pairs


@pytest.mark.parametrize("kpm_prec", ["double", "single"])
@pytest.mark.parametrize("rashba", [0., 0.3])
@pytest.mark.parametrize("budget", [None, 2*50*8, 2*50*1])
def test_the_numba_recursion_agrees_with_the_jax_one(budget, rashba, kpm_prec,
        monkeypatch):
    """The CPU kernel (pairmomentsnumba) against the device one
    (pairmomentsjax), which share no code: a real and a complex H, blocks
    that are doubled, read on the rows, and a mixture of both when the
    budget splits the starting columns, in both precisions. The whole
    island is one block small enough to run its rows in one thread; the
    split blocks run them on all the threads"""
    from pyqula.kpmtk import pairmomentsnumba
    ms, pairs = _hubbard_island(rashba)
    coef = np.random.default_rng(2).normal(size=80)
    ref, _ = pairmomentsjax.pair_values(ms, pairs, coef, kpm_prec="double")
    if budget is not None:
        monkeypatch.setattr(pairmomentsnumba, "_MAX_BLOCK", budget)
        monkeypatch.setattr(pairmomentsnumba, "_MIN_COLUMNS", 1)
        monkeypatch.setattr(pairmomentsnumba, "_SERIAL_BLOCK", 0)
    vals, mumax = pairmomentsnumba.pair_values(ms, pairs, coef,
            kpm_prec=kpm_prec)
    tol = 1e-12 if kpm_prec == "double" else 1e-4
    assert np.max(np.abs(vals - ref)) < tol
    assert mumax <= 1. + 1e-6


def test_the_numba_recursion_reads_the_rows_of_a_dense_pair_set():
    """Every pair of a small cell, which the dense engine asks for on a
    k-mesh, is read on the rows, since doubling it would take an inner
    product per pair and step, N^3 per step; the pairs of an onsite
    interaction on an island are doubled. Both give the jax values, the
    cell with one k-point per thread"""
    from pyqula.kpmtk import pairmomentsnumba
    ms = _scaled_hks(_noncollinear_chain(), nk=4)
    n = ms[0].shape[0]
    every = np.array([(i, j) for i in range(n) for j in range(i, n)])
    assert not any(b[2] for b in pairmomentsnumba._plan(every, n, ms[0].nnz))
    ms2, onsite = _hubbard_island(0.3)
    onsite = onsite[:-(len(onsite)//4)] # the onsite pairs only
    n2 = ms2[0].shape[0]
    assert all(b[2] for b in pairmomentsnumba._plan(onsite, n2, ms2[0].nnz))
    coef = np.linspace(1., -1., 40)
    for m, p in [(ms, every), (ms2, onsite)]:
        a, _ = pairmomentsnumba.pair_values(m, p, coef)
        b, _ = pairmomentsjax.pair_values(m, p, coef)
        assert np.max(np.abs(a - b)) < 1e-12


@pytest.mark.parametrize("nm", [80, 81])
def test_the_numba_trace_matches_numba_full_trace(nm):
    """The doubled block trace against the per-vector batched kernel, for
    an even and an odd number of moments"""
    from pyqula.kpmtk import pairmomentsnumba
    ms = _scaled_hks(_noncollinear_chain(), nk=4)
    mt, _ = pairmomentsnumba.trace_moments(ms, nm)
    for ik, m in enumerate(ms):
        ref = kpm.full_trace(m, n=(nm + 1)//2)[:nm]
        assert np.max(np.abs(mt[ik] - ref)) < 1e-12


def test_the_numba_recursion_runs_serially_when_parallelism_is_disabled(
        monkeypatch):
    """parallel.set_enabled(False) clamps numba to one thread, which the
    kernel reads when it splits the rows, so the CPU engine of the KPM
    mean field becomes serial, which jax's CPU backend did not; the values
    do not change beyond roundoff"""
    import numba
    from pyqula import parallel
    from pyqula.kpmtk import pairmomentsnumba
    monkeypatch.setattr(pairmomentsnumba, "_SERIAL_BLOCK", 0) # split the rows
    ms, pairs = _hubbard_island(0.3)
    coef = np.linspace(1., -1., 60)
    ref, _ = pairmomentsnumba.pair_values(ms, pairs, coef)
    was, threads = parallel.enabled, numba.get_num_threads()
    try:
        parallel.set_enabled(False)
        assert numba.get_num_threads() == 1
        vals, _ = pairmomentsnumba.pair_values(ms, pairs, coef)
    finally: # set_enabled(True) lifts the clamp without restoring the count
        parallel.set_enabled(was)
        numba.set_num_threads(threads)
    assert np.max(np.abs(vals - ref)) < 1e-13


def test_every_device_call_of_a_kind_has_one_shape(monkeypatch):
    """The blocks of starting columns hold different numbers of pairs, and
    the last block and the last group of k-points are shorter, but every
    call of the device kernel of one kind (doubled, or read on the rows)
    gets the same shapes, padded, so that it compiles once per kind; the
    padding does not change the values"""
    ms = _scaled_hks(_noncollinear_chain(), nk=5)
    n = ms[0].shape[0]
    pairs = [(i, j) for i in range(n) for j in range(i, n)] # 1, 2, 3, 4 per column
    coef = np.linspace(1., -1., 30)
    whole, _ = pairmomentsjax.pair_values(ms, pairs, coef)
    kernel = pairmomentsjax._contracted
    # columns 0 and 1 (3 pairs) and 2 and 3 (7 pairs), one k-point per call; and
    # all four columns, two k-points per call, so the last call has one
    for budget in [3*n*3 + 4, 2*(3*n*n + 30*len(pairs)) + 4]:
        shapes = dict()
        def recorded(*args):
            kind = args[9] # doubled, the second static argument
            shapes.setdefault(kind, []).append(tuple(np.shape(a)
                for a in args if hasattr(a, "shape")))
            return kernel(*args)
        monkeypatch.setattr(pairmomentsjax, "_contracted", recorded)
        monkeypatch.setattr(pairmomentsjax, "_MAX_BLOCK", budget)
        pieces, _ = pairmomentsjax.pair_values(ms, pairs, coef)
        assert sum(len(x) for x in shapes.values()) > 2
        assert all(len(set(x)) == 1 for x in shapes.values()), shapes
        assert np.max(np.abs(whole - pieces)) < 1e-13


def test_the_ell_width_survives_an_entry_passing_through_zero():
    """In a mean-field loop an entry of H that passes through zero drops
    out of its sparse form; the ELL width, which sets the compiled kernel,
    stays the one already used for that dimension"""
    m = csr_matrix(np.array([[0., 1., 1.], [1., 0., 1.], [1., 1., 0.5]]))
    _, cols = pairmomentsjax._ell([m])
    m2 = m.copy()
    m2[0, 2] = 0.; m2[2, 0] = 0. # the widest row loses an entry
    m2.eliminate_zeros()
    _, cols2 = pairmomentsjax._ell([m2])
    assert cols2.shape == cols.shape


def test_the_device_doubles_in_double_precision_only():
    """The pairs of an onsite interaction are doubled on the device as on
    the CPU, in double precision; in single precision they are read on the
    rows, since the doubled moments are inner products summed over every
    orbital in the precision of the recursion"""
    ms, pairs = _hubbard_island(0.3)
    onsite = pairs[:-(len(pairs)//4)]
    n, K = ms[0].shape[0], 10
    assert all(b["doubled"] for b in pairmomentsjax._plan(onsite, n, K, 80,
        "double"))
    assert not any(b["doubled"] for b in pairmomentsjax._plan(onsite, n, K,
        80, "single"))
