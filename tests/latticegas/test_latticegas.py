import numpy as np
import pytest

from pyqula import geometry
from pyqula import latticegas


def _small_lattice_gas(seed=0, filling=1./3.):
    """Small 0d lattice-gas instance with a couple of interaction shells,
    small enough that the tests below run in well under a second"""
    np.random.seed(seed)
    g = geometry.triangular_lattice()
    g = g.get_supercell(3) # 9 sites
    g.dimensionality = 0
    lg = latticegas.LatticeGas(g, filling=filling)
    lg.mu = np.random.uniform(-1., 1., lg.nsites)
    lg.add_interaction(Jij=[1., 0.5, 0.2])
    return lg


def _naive_structure_factor(r, den, q):
    """Independent, unoptimized reimplementation of S(q), used as a
    ground truth for get_structure_factor"""
    den0 = den - np.mean(den)
    phase = r[:, 0] * q[0] + r[:, 1] * q[1] + r[:, 2] * q[2]
    s = np.sum(den0 * np.exp(-1j * phase))
    return np.abs(s) ** 2 / len(den)


def _naive_local_energy(mu, pairs, j, den, ii):
    """Independent, unoptimized reimplementation of the local-energy
    formula (loop over pairs in plain Python), used as a ground truth
    to check get_local_energy against"""
    e = mu[ii] * den[ii]
    for (p, q), jj in zip(pairs, j):
        if p == ii or q == ii:
            e += 0.5 * jj * den[p] * den[q]
    return e


def test_random_density_has_correct_count():
    Ntot, N = 37, 13
    den = latticegas.random_density(Ntot, N)
    assert len(den) == Ntot
    assert np.sum(den) == N
    assert set(np.unique(den)) <= {0., 1.}


def test_optimize_discrete_uniform_filling_raises_valueerror():
    lg = _small_lattice_gas()
    lg.set_filling(0.0) # every site empty -> only one distinct value
    with pytest.raises(ValueError):
        lg.optimize_energy(temp=0.1, ntries=10)


def test_local_energy_sum_matches_total_energy():
    lg = _small_lattice_gas()
    local = lg.get_local_energy()
    assert np.isclose(np.sum(local), lg.get_energy())


def test_local_energy_matches_naive_reference():
    lg = _small_lattice_gas()
    local = lg.get_local_energy()
    for ii in range(lg.nsites):
        expected = _naive_local_energy(lg.mu, lg.pairs, lg.j, lg.den, ii)
        assert np.isclose(local[ii], expected)


def test_local_mu_matches_naive_reference():
    lg = _small_lattice_gas()
    local_mu = lg.get_local_mu()
    for ii in range(lg.nsites):
        den0 = lg.den.copy()
        den0[ii] = 1.0
        expected = _naive_local_energy(lg.mu, lg.pairs, lg.j, den0, ii)
        assert np.isclose(local_mu[ii], expected)


def test_optimize_energy_does_not_increase_minimum_energy():
    lg = _small_lattice_gas(seed=1)
    fun = lambda x: latticegas.energy_numba(lg.mu, lg.pairs, lg.j, x)
    e0 = fun(lg.den)
    np.random.seed(1)
    es = lg.optimize_energy(temp=0.5, ntries=2000)
    assert len(es) == 2000
    assert np.min(es) <= e0


def test_swap_delta_energy_matches_brute_force_recompute():
    """swap_delta_energy (O(degree), used by the annealer's hot loop)
    must agree with recomputing the full energy before/after the swap
    from scratch (O(n_pairs)) -- this is the correctness guarantee for
    the incremental-energy optimization in optimize_discrete"""
    np.random.seed(2)
    lg = _small_lattice_gas(seed=2)
    n = lg.nsites
    ptr, idx, jarr = latticegas._build_adjacency(n, lg.pairs, lg.j)
    for _ in range(200):
        den = latticegas.random_density(n, np.random.randint(1, n))
        i1, i2 = np.random.choice(n, 2, replace=False)
        e_before = latticegas.energy_numba(lg.mu, lg.pairs, lg.j, den)
        den2 = den.copy()
        den2[i1], den2[i2] = den2[i2], den2[i1]
        e_after = latticegas.energy_numba(lg.mu, lg.pairs, lg.j, den2)
        delta_fast = latticegas.swap_delta_energy(lg.mu, ptr, idx, jarr, den, i1, i2)
        assert np.isclose(e_after - e_before, delta_fast, atol=1e-10)


def test_optimize_energy_running_total_matches_full_recompute():
    """The incrementally-tracked running energy returned in `es` must
    not drift (via floating-point accumulation or a resync bug) from
    the energy of the actual final density it reports"""
    lg = _small_lattice_gas(seed=3)
    np.random.seed(3)
    es = lg.optimize_energy(temp=1.0, ntries=5000, resync_every=17)
    recomputed = latticegas.energy_numba(lg.mu, lg.pairs, lg.j, lg.den)
    assert np.isclose(es[-1], recomputed, atol=1e-8)


def test_adjacency_cache_is_reused_and_invalidated():
    """The CSR adjacency only depends on pairs/j, which are invariant
    across repeated optimize_energy() calls -- it should be built once
    and reused (examples/latticegas/optimize/main.py calls
    optimize_energy in a loop), and rebuilt only when add_interaction
    changes pairs/j"""
    lg = _small_lattice_gas(seed=4)
    assert lg._adjacency is None # not built until first needed
    lg.optimize_energy(temp=0.5, ntries=50)
    adjacency_first = lg._adjacency
    assert adjacency_first is not None
    lg.optimize_energy(temp=0.5, ntries=50)
    assert lg._adjacency is adjacency_first # reused, not rebuilt
    lg.add_interaction(Jij=[0., 0., 0.3])
    assert lg._adjacency is None # invalidated by new interaction terms
    lg.optimize_energy(temp=0.5, ntries=50)
    assert lg._adjacency is not None
    assert lg._adjacency is not adjacency_first


def test_optimize_discrete_patience_stops_early():
    lg = _small_lattice_gas(seed=2)
    np.random.seed(2)
    es = lg.optimize_energy(temp=0.5, ntries=5000, patience=50)
    assert len(es) < 5000 # stopped before exhausting ntries
    assert len(es) > 0


def test_optimize_discrete_without_patience_runs_full_ntries():
    lg = _small_lattice_gas(seed=2)
    np.random.seed(2)
    es = lg.optimize_energy(temp=0.5, ntries=500) # patience defaults to None
    assert len(es) == 500


def test_anneal_does_not_increase_energy_and_concatenates_trajectory():
    lg = _small_lattice_gas(seed=3)
    e0 = lg.get_energy()
    np.random.seed(3)
    temps = [1.0, 0.5, 0.1]
    es = lg.anneal(temps=temps, ntries=300)
    assert len(es) == len(temps) * 300
    assert lg.get_energy() <= e0
    assert np.isclose(np.min(es), lg.get_energy()) # best-kept, not last-step


def test_write_read_roundtrip(tmp_path):
    lg = _small_lattice_gas(seed=4)
    np.random.seed(4)
    lg.optimize_energy(temp=0.5, ntries=200)
    den_before = lg.den.copy()
    path = str(tmp_path / "density.out")
    lg.write(name=path)
    lg2 = _small_lattice_gas(seed=0) # different initial density
    lg2.read(name=path)
    assert np.array_equal(lg2.den, den_before)


def test_read_rejects_mismatched_site_count(tmp_path):
    lg = _small_lattice_gas(seed=4)
    path = str(tmp_path / "density.out")
    lg.write(name=path)
    g_big = geometry.triangular_lattice().get_supercell(4)
    g_big.dimensionality = 0
    lg_big = latticegas.LatticeGas(g_big, filling=1. / 3.)
    with pytest.raises(ValueError):
        lg_big.read(name=path)


def test_flip_delta_energy_matches_brute_force_recompute():
    """flip_delta_energy (the grand-canonical move, O(degree)) must
    agree with recomputing the full energy before/after the flip from
    scratch (O(n_pairs)) -- same correctness guarantee as
    test_swap_delta_energy_matches_brute_force_recompute, for the
    single-site-flip move instead of the swap move"""
    np.random.seed(5)
    lg = _small_lattice_gas(seed=5)
    n = lg.nsites
    ptr, idx, jarr = latticegas._build_adjacency(n, lg.pairs, lg.j)
    for _ in range(200):
        den = latticegas.random_density(n, np.random.randint(0, n + 1))
        i = np.random.randint(0, n)
        e_before = latticegas.energy_numba(lg.mu, lg.pairs, lg.j, den)
        den2 = den.copy()
        den2[i] = 1. - den2[i]
        e_after = latticegas.energy_numba(lg.mu, lg.pairs, lg.j, den2)
        delta_fast = latticegas.flip_delta_energy(lg.mu, ptr, idx, jarr, den, i)
        assert np.isclose(e_after - e_before, delta_fast, atol=1e-10)


def test_optimize_grand_canonical_filling_fluctuates():
    lg = _small_lattice_gas(seed=9)
    lg.mu[:] = -0.3 # bias towards partial filling instead of collapsing to 0 or n
    np.random.seed(9)
    es, ns = lg.optimize_grand_canonical(temp=1.0, ntries=3000)
    assert len(np.unique(ns)) > 1 # filling is not conserved, unlike optimize_energy
    assert np.isclose(ns[-1], np.sum(lg.den))


def test_optimize_grand_canonical_running_total_matches_full_recompute():
    lg = _small_lattice_gas(seed=9)
    np.random.seed(9)
    es, ns = lg.optimize_grand_canonical(temp=1.0, ntries=3000, resync_every=17)
    recomputed = latticegas.energy_numba(lg.mu, lg.pairs, lg.j, lg.den)
    assert np.isclose(es[-1], recomputed, atol=1e-8)
    assert np.isclose(ns[-1], np.sum(lg.den))


def test_optimize_grand_canonical_allows_uniform_start():
    """Unlike optimize_energy (swap-based), the flip-based
    grand-canonical move does not require 2 distinct starting values"""
    lg = _small_lattice_gas(seed=9)
    lg.den[:] = 0. # every site empty
    es, ns = lg.optimize_grand_canonical(temp=1.0, ntries=50)
    assert len(es) == 50


def test_specific_heat_and_susceptibility_are_nonnegative():
    lg = _small_lattice_gas(seed=9)
    np.random.seed(9)
    es, ns = lg.optimize_grand_canonical(temp=1.0, ntries=3000)
    C = latticegas.get_specific_heat(es, 1.0)
    chi = latticegas.get_susceptibility(ns, 1.0)
    assert C >= 0.
    assert chi >= 0.


def test_specific_heat_zero_for_constant_trajectory():
    es = np.full(100, 3.5)
    assert latticegas.get_specific_heat(es, 1.0) == 0.


def test_optimize_energy_multistart_is_no_worse_than_initial_and_preserves_filling():
    lg = _small_lattice_gas(seed=6)
    e0 = lg.get_energy()
    n_before = np.sum(lg.den)
    np.random.seed(6)
    e_best = lg.optimize_energy_multistart(nstart=5, ntries=1000, temp=0.5)
    assert e_best <= e0
    assert np.isclose(e_best, lg.get_energy())
    assert np.sum(lg.den) == n_before # filling preserved (canonical move set)


def test_add_tensor_matches_naive_energy_reference():
    lg = _small_lattice_gas(seed=10)
    lg2 = lg.copy()

    def fun(r1, r2):
        d = np.linalg.norm(r1 - r2)
        return 1. / d ** 3

    lg2.add_tensor(fun)
    r = lg.geometry.r
    expected = lg.get_energy()
    for i1 in range(lg.nsites):
        for i2 in range(lg.nsites):
            if i1 == i2: continue
            expected += fun(r[i1], r[i2]) * lg.den[i1] * lg.den[i2]
    assert np.isclose(lg2.get_energy(), expected)


def test_add_tensor_skips_self_pairs():
    lg = _small_lattice_gas(seed=10)
    npairs_before = len(lg.pairs)
    lg.add_tensor(lambda r1, r2: 0.) # never triggers the >1e-7 threshold
    assert len(lg.pairs) == npairs_before


def test_regroup_merges_duplicates_and_preserves_energy():
    lg = _small_lattice_gas(seed=11)
    lg.add_interaction(Jij=[1., 0.5, 0.2]) # same shells again -> exact duplicates
    npairs_before = len(lg.pairs)
    e_before = lg.get_energy()
    lg.regroup()
    assert len(lg.pairs) < npairs_before
    assert np.isclose(lg.get_energy(), e_before)


def test_regroup_keeps_adjacency_correct_for_swap_delta_energy():
    """regroup() must not fold (i,j) into (j,i): _build_adjacency's
    per-site row sums need both directions present in their own row"""
    lg = _small_lattice_gas(seed=11)
    lg.add_interaction(Jij=[1., 0.5, 0.2])
    lg.regroup()
    n = lg.nsites
    ptr, idx, jarr = lg._get_adjacency()
    np.random.seed(12)
    for _ in range(100):
        den = latticegas.random_density(n, np.random.randint(1, n))
        i1, i2 = np.random.choice(n, 2, replace=False)
        e_before = latticegas.energy_numba(lg.mu, lg.pairs, lg.j, den)
        den2 = den.copy()
        den2[i1], den2[i2] = den2[i2], den2[i1]
        e_after = latticegas.energy_numba(lg.mu, lg.pairs, lg.j, den2)
        delta_fast = latticegas.swap_delta_energy(lg.mu, ptr, idx, jarr, den, i1, i2)
        assert np.isclose(e_after - e_before, delta_fast, atol=1e-10)


def test_structure_factor_q_zero_is_zero():
    lg = _small_lattice_gas(seed=13)
    qpath, sq = lg.get_structure_factor(qpath=[[0., 0., 0.]])
    assert np.isclose(sq[0], 0., atol=1e-20)


def test_structure_factor_matches_naive_reference():
    lg = _small_lattice_gas(seed=13)
    qs = [[0.3, 0., 0.], [0., 0.7, 0.], [0.5, 0.5, 0.]]
    qpath, sq = lg.get_structure_factor(qpath=qs)
    for i, q in enumerate(qs):
        expected = _naive_structure_factor(lg.geometry.r, lg.den, q)
        assert np.isclose(sq[i], expected)


def test_structure_factor_default_grid_shape():
    lg = _small_lattice_gas(seed=13)
    qpath, sq = lg.get_structure_factor(nq=8)
    assert qpath.shape == (64, 3)
    assert sq.shape == (64,)
    assert np.all(np.isfinite(sq))


def test_optimize_energy_zero_temperature_never_accepts_uphill_move():
    """temp=0 must run greedy (zero-temperature) dynamics: the running
    energy trace can never increase, and no NaN/inf should leak in
    from the (e-en)/temp division that the finite-temperature branch
    performs"""
    lg = _small_lattice_gas(seed=7)
    np.random.seed(7)
    es = lg.optimize_energy(temp=0.0, ntries=2000)
    assert np.all(np.isfinite(es))
    assert np.all(np.diff(es) <= 1e-10) # never increases


def test_anneal_zero_temperature_schedule_works():
    lg = _small_lattice_gas(seed=7)
    e0 = lg.get_energy()
    np.random.seed(7)
    es = lg.anneal(temps=[0.0, 0.0], ntries=300)
    assert np.all(np.isfinite(es))
    assert lg.get_energy() <= e0


def test_optimize_energy_checkpoint_at_matches_state_at_that_step():
    """checkpoint_at must capture the same density that a run stopped
    after exactly that many tries would end up with (checked by
    replaying the RNG from the same seed)"""
    lg = _small_lattice_gas(seed=8)
    np.random.seed(8)
    lg.optimize_energy(temp=0.5, ntries=100, checkpoint_at=[30, 70])
    den_at_30 = lg.checkpoints[30].copy()
    den_at_70 = lg.checkpoints[70].copy()
    assert set(lg.checkpoints.keys()) == {30, 70}

    lg2 = _small_lattice_gas(seed=8)
    np.random.seed(8)
    lg2.optimize_energy(temp=0.5, ntries=30)
    assert np.array_equal(lg2.den, den_at_30)

    lg3 = _small_lattice_gas(seed=8)
    np.random.seed(8)
    lg3.optimize_energy(temp=0.5, ntries=70)
    assert np.array_equal(lg3.den, den_at_70)


def test_optimize_energy_checkpoint_at_accepts_scalar():
    lg = _small_lattice_gas(seed=8)
    np.random.seed(8)
    lg.optimize_energy(temp=0.5, ntries=50, checkpoint_at=20)
    assert set(lg.checkpoints.keys()) == {20}


def test_optimize_energy_without_checkpoint_at_leaves_checkpoints_empty():
    lg = _small_lattice_gas(seed=8)
    np.random.seed(8)
    lg.optimize_energy(temp=0.5, ntries=50)
    assert lg.checkpoints == {}


def test_anneal_checkpoint_at_uses_global_step_numbering():
    """checkpoint_at on anneal() counts steps continuously across
    temperature stages, not restarting at each new temperature"""
    lg = _small_lattice_gas(seed=8)
    np.random.seed(8)
    ntries = 50
    temps = [1.0, 0.5, 0.1]
    lg.anneal(temps=temps, ntries=ntries, checkpoint_at=[10, 60, 120])
    assert set(lg.checkpoints.keys()) == {10, 60, 120}

    # step 60 = 10 steps into the 2nd temperature stage: replay it
    lg2 = _small_lattice_gas(seed=8)
    np.random.seed(8)
    lg2.optimize_energy(temp=temps[0], ntries=ntries)
    lg2.optimize_energy(temp=temps[1], ntries=10)
    assert np.array_equal(lg2.den, lg.checkpoints[60])
