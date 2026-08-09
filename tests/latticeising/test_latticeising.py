import numpy as np
import pytest

from pyqula import geometry
from pyqula import latticegas
from pyqula import latticeising


def _small_lattice_ising(seed=0, m=0.0):
    """Small 0d Ising instance with a couple of interaction shells,
    small enough that the tests below run in well under a second"""
    np.random.seed(seed)
    g = geometry.triangular_lattice()
    g = g.get_supercell(3) # 9 sites
    g.dimensionality = 0
    li = latticeising.LatticeIsing(g, m=m)
    li.b = np.random.uniform(-1., 1., li.nsites)
    li.add_interaction(Jij=[1., 0.5, 0.2])
    return li


def _naive_local_energy(b, pairs, j, s, ii):
    """Independent, unoptimized reimplementation of the local-energy
    formula (loop over pairs in plain Python), used as ground truth"""
    e = -b[ii] * s[ii]
    for (p, q), jj in zip(pairs, j):
        if p == ii or q == ii:
            e -= 0.5 * jj * s[p] * s[q]
    return e


def test_random_spins_has_correct_count():
    Ntot, N_up = 37, 13
    s = latticeising.random_spins(Ntot, N_up)
    assert len(s) == Ntot
    assert np.sum(s == 1.) == N_up
    assert set(np.unique(s)) <= {-1., 1.}


def test_ferromagnetic_coupling_orders_on_bipartite_lattice():
    """Sign-convention check: positive J must be ferromagnetic (favors
    alignment), not the LatticeGas convention (positive J repels).
    Uses a bipartite (square) lattice so ferromagnetic order is not
    frustrated"""
    np.random.seed(0)
    g = geometry.square_lattice().get_supercell(6)
    g.dimensionality = 0
    li = latticeising.LatticeIsing(g, m=0.0)
    li.add_interaction(Jij=[1.])
    li.anneal(temps=np.geomspace(3.0, 0.05, 10), ntries=2000)
    assert abs(li.get_magnetization()) > 0.9


def test_antiferromagnetic_coupling_orders_away_from_q_zero():
    """Negative J on a bipartite lattice must order into a checkerboard
    pattern, peaking the structure factor away from q=0"""
    np.random.seed(0)
    g = geometry.square_lattice().get_supercell(6)
    g.dimensionality = 0
    li = latticeising.LatticeIsing(g, m=0.0)
    li.add_interaction(Jij=[-1.])
    li.anneal(temps=np.geomspace(3.0, 0.05, 10), ntries=2000)
    _, sq0 = li.get_structure_factor(qpath=[[0., 0., 0.]])
    _, sq = li.get_structure_factor(nq=10)
    assert sq0[0] < 1e-10
    assert np.max(sq) > 1.0


def test_positive_field_polarizes_spins_up():
    """Sign-convention check for add_field(): positive b must favor
    s=+1, not s=-1 (no add_interaction here, so this isolates the
    field term from the coupling term)"""
    np.random.seed(0)
    g = geometry.square_lattice().get_supercell(4)
    g.dimensionality = 0
    li = latticeising.LatticeIsing(g, m=0.0)
    li.add_field(0.5)
    li.optimize_energy(temp=0.05, ntries=2000)
    assert li.get_magnetization() > 0.99


def test_negative_field_polarizes_spins_down():
    np.random.seed(0)
    g = geometry.square_lattice().get_supercell(4)
    g.dimensionality = 0
    li = latticeising.LatticeIsing(g, m=0.0)
    li.add_field(-0.5)
    li.optimize_energy(temp=0.05, ntries=2000)
    assert li.get_magnetization() < -0.99


def test_optimize_conserved_uniform_magnetization_raises_valueerror():
    li = _small_lattice_ising()
    li.set_magnetization(1.0) # every spin up -> only one distinct value
    with pytest.raises(ValueError):
        li.optimize_conserved(temp=0.1, ntries=10)


def test_local_energy_sum_matches_total_energy():
    li = _small_lattice_ising()
    local = li.get_local_energy()
    assert np.isclose(np.sum(local), li.get_energy())


def test_local_energy_matches_naive_reference():
    li = _small_lattice_ising()
    local = li.get_local_energy()
    for ii in range(li.nsites):
        expected = _naive_local_energy(li.b, li.pairs, li.j, li.s, ii)
        assert np.isclose(local[ii], expected)


def test_local_field_matches_flip_delta_identity():
    """get_local_field()[i] must satisfy flip_delta_energy_ising(...,i)
    == 2*s[i]*local_field[i] exactly, by construction"""
    li = _small_lattice_ising(seed=1)
    ptr, idx, jarr = li._get_adjacency()
    field = li.get_local_field()
    for i in range(li.nsites):
        delta = latticeising.flip_delta_energy_ising(li.b, ptr, idx, jarr, li.s, i)
        assert np.isclose(delta, 2. * li.s[i] * field[i])


def test_flip_delta_energy_matches_brute_force_recompute():
    np.random.seed(5)
    li = _small_lattice_ising(seed=5)
    n = li.nsites
    ptr, idx, jarr = latticeising._build_adjacency(n, li.pairs, li.j)
    for _ in range(200):
        s = latticeising.random_spins(n, np.random.randint(0, n + 1))
        i = np.random.randint(0, n)
        e_before = latticeising.ising_energy_numba(li.b, li.pairs, li.j, s)
        s2 = s.copy()
        s2[i] = -s2[i]
        e_after = latticeising.ising_energy_numba(li.b, li.pairs, li.j, s2)
        delta_fast = latticeising.flip_delta_energy_ising(li.b, ptr, idx, jarr, s, i)
        assert np.isclose(e_after - e_before, delta_fast, atol=1e-10)


def test_swap_delta_energy_matches_brute_force_recompute():
    np.random.seed(2)
    li = _small_lattice_ising(seed=2)
    n = li.nsites
    ptr, idx, jarr = latticeising._build_adjacency(n, li.pairs, li.j)
    for _ in range(200):
        s = latticeising.random_spins(n, np.random.randint(1, n))
        i1, i2 = np.random.choice(n, 2, replace=False)
        e_before = latticeising.ising_energy_numba(li.b, li.pairs, li.j, s)
        s2 = s.copy()
        s2[i1], s2[i2] = s2[i2], s2[i1]
        e_after = latticeising.ising_energy_numba(li.b, li.pairs, li.j, s2)
        delta_fast = latticeising.swap_delta_energy_ising(li.b, ptr, idx, jarr, s, i1, i2)
        assert np.isclose(e_after - e_before, delta_fast, atol=1e-10)


def test_optimize_energy_returns_energy_and_magnetization_trajectories():
    li = _small_lattice_ising(seed=3)
    np.random.seed(3)
    es, ms = li.optimize_energy(temp=1.0, ntries=2000)
    assert len(es) == 2000
    assert len(ms) == 2000
    assert np.isclose(ms[-1], np.sum(li.s))


def test_optimize_energy_running_total_matches_full_recompute():
    li = _small_lattice_ising(seed=3)
    np.random.seed(3)
    es, ms = li.optimize_energy(temp=1.0, ntries=5000, resync_every=17)
    recomputed = latticeising.ising_energy_numba(li.b, li.pairs, li.j, li.s)
    assert np.isclose(es[-1], recomputed, atol=1e-8)
    assert np.isclose(ms[-1], np.sum(li.s))


def test_optimize_energy_magnetization_fluctuates():
    li = _small_lattice_ising(seed=9)
    li.b[:] = 0.1 # small bias, not enough to fully polarize
    np.random.seed(9)
    es, ms = li.optimize_energy(temp=2.0, ntries=3000)
    assert len(np.unique(ms)) > 1


def test_optimize_conserved_preserves_magnetization():
    li = _small_lattice_ising(seed=6)
    m0 = np.sum(li.s)
    np.random.seed(6)
    li.optimize_conserved(temp=0.5, ntries=1000)
    assert np.sum(li.s) == m0


def test_optimize_conserved_patience_stops_early():
    li = _small_lattice_ising(seed=2)
    np.random.seed(2)
    es = li.optimize_conserved(temp=0.5, ntries=5000, patience=50)
    assert len(es) < 5000
    assert len(es) > 0


def test_anneal_does_not_increase_energy_and_concatenates_trajectory():
    li = _small_lattice_ising(seed=3)
    e0 = li.get_energy()
    np.random.seed(3)
    temps = [1.0, 0.5, 0.1]
    es, ms = li.anneal(temps=temps, ntries=300)
    assert len(es) == len(temps) * 300
    assert len(ms) == len(temps) * 300
    assert li.get_energy() <= e0
    assert np.isclose(np.min(es), li.get_energy())


def test_adjacency_cache_is_reused_and_invalidated():
    li = _small_lattice_ising(seed=4)
    assert li._adjacency is None
    li.optimize_energy(temp=0.5, ntries=50)
    adjacency_first = li._adjacency
    assert adjacency_first is not None
    li.optimize_energy(temp=0.5, ntries=50)
    assert li._adjacency is adjacency_first
    li.add_interaction(Jij=[0., 0., 0.3])
    assert li._adjacency is None
    li.optimize_energy(temp=0.5, ntries=50)
    assert li._adjacency is not adjacency_first


def test_write_read_roundtrip(tmp_path):
    li = _small_lattice_ising(seed=4)
    np.random.seed(4)
    li.optimize_energy(temp=0.5, ntries=200)
    s_before = li.s.copy()
    path = str(tmp_path / "spin.out")
    li.write(name=path)
    li2 = _small_lattice_ising(seed=0)
    li2.read(name=path)
    assert np.array_equal(li2.s, s_before)


def test_read_rejects_mismatched_site_count(tmp_path):
    li = _small_lattice_ising(seed=4)
    path = str(tmp_path / "spin.out")
    li.write(name=path)
    g_big = geometry.triangular_lattice().get_supercell(4)
    g_big.dimensionality = 0
    li_big = latticeising.LatticeIsing(g_big, m=0.0)
    with pytest.raises(ValueError):
        li_big.read(name=path)


def test_optimize_energy_multistart_is_no_worse_than_initial():
    li = _small_lattice_ising(seed=6)
    e0 = li.get_energy()
    np.random.seed(6)
    e_best = li.optimize_energy_multistart(nstart=5, ntries=1000, temp=0.5)
    assert e_best <= e0
    assert np.isclose(e_best, li.get_energy())


def test_add_tensor_matches_naive_energy_reference():
    li = _small_lattice_ising(seed=10)
    li2 = li.copy()

    def fun(r1, r2):
        d = np.linalg.norm(r1 - r2)
        return 1. / d ** 3

    li2.add_tensor(fun)
    r = li.geometry.r
    expected = li.get_energy()
    for i1 in range(li.nsites):
        for i2 in range(li.nsites):
            if i1 == i2: continue
            expected -= fun(r[i1], r[i2]) * li.s[i1] * li.s[i2]
    assert np.isclose(li2.get_energy(), expected)


def test_regroup_merges_duplicates_and_preserves_energy():
    li = _small_lattice_ising(seed=11)
    li.add_interaction(Jij=[1., 0.5, 0.2]) # same shells again -> exact duplicates
    npairs_before = len(li.pairs)
    e_before = li.get_energy()
    li.regroup()
    assert len(li.pairs) < npairs_before
    assert np.isclose(li.get_energy(), e_before)


def test_structure_factor_q_zero_is_zero():
    li = _small_lattice_ising(seed=13)
    qpath, sq = li.get_structure_factor(qpath=[[0., 0., 0.]])
    assert np.isclose(sq[0], 0., atol=1e-20)


def test_specific_heat_and_susceptibility_reused_from_latticegas():
    """latticegas.get_specific_heat/get_susceptibility are generic
    over any (es, ns) trajectory, so LatticeIsing reuses them directly
    for (es, ms) instead of redefining equivalent functions"""
    li = _small_lattice_ising(seed=9)
    np.random.seed(9)
    es, ms = li.optimize_energy(temp=2.0, ntries=3000)
    C = latticegas.get_specific_heat(es, 2.0)
    chi = latticegas.get_susceptibility(ms, 2.0)
    assert C >= 0.
    assert chi >= 0.


def test_optimize_energy_zero_temperature_never_accepts_uphill_move():
    li = _small_lattice_ising(seed=7)
    np.random.seed(7)
    es, ms = li.optimize_energy(temp=0.0, ntries=2000)
    assert np.all(np.isfinite(es))
    assert np.all(np.diff(es) <= 1e-10)


def test_optimize_energy_checkpoint_at_matches_state_at_that_step():
    li = _small_lattice_ising(seed=8)
    np.random.seed(8)
    li.optimize_energy(temp=0.5, ntries=100, checkpoint_at=[30, 70])
    s_at_30 = li.checkpoints[30].copy()
    assert set(li.checkpoints.keys()) == {30, 70}

    li2 = _small_lattice_ising(seed=8)
    np.random.seed(8)
    li2.optimize_energy(temp=0.5, ntries=30)
    assert np.array_equal(li2.s, s_at_30)
