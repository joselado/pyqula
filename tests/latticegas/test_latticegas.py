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
