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


def test_trace_moments_match_numba_full_trace():
    ms = _scaled_hks(_noncollinear_chain(), nk=4)
    mt, _ = pairmomentsjax.trace_moments(ms, 80)
    for ik, m in enumerate(ms):
        assert np.max(np.abs(mt[ik] - kpm.full_trace(m, n=40))) < 1e-12


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
