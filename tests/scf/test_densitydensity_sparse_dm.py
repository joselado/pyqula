import numpy as np

from pyqula import geometry
from pyqula.scftk.densitydensity import (Vinteraction, dm_sparse_pairs,
        get_dm, get_mf)

NK, T = 4, 1e-3


def _system(nsuper=4):
    """A honeycomb supercell large enough that a first-neighbour bond
    direction is genuinely sparse (below full_dm_accumulate_sparse's own
    dense_fraction), which is where the sparse kernel is actually used.

    One SCF iteration (maxite=0) purely to get Vinteraction's own
    interaction dictionary and a Hamiltonian carrying a generic (random)
    mean field, rather than rebuilding either here."""
    g = geometry.honeycomb_lattice().supercell(nsuper)
    h = g.get_hamiltonian()
    scf = Vinteraction(h, V1=1.0, U=1.0, filling=0.5, nk=2, maxite=0,
            load_mf=False, verbose=0)
    return scf.hamiltonian, scf.v


def test_sparse_density_matrix_gives_the_same_mean_field(tmp_path,
                                                        monkeypatch):
    """The mean-field kernels (normal_term_ii/jj/ij) and the
    double-counting energy multiply EVERY density-matrix entry by v[i,j],
    so an entry where v is zero cannot contribute to any of them. Computing
    only the entries v's nonzero pattern reads therefore has to give the
    same mean field as the full (n,n) density matrix -- that exact
    invariance is what makes the sparse kernel a legitimate replacement
    for the dense one, and it is what this asserts."""
    monkeypatch.chdir(tmp_path)
    h, v = _system()
    ds = [(0, 0, 0)] + list(v.keys())
    dense = h.get_density_matrix(ds=ds, nk=NK, T=T)
    sparse = get_dm(h, v, nk=NK, T=T)
    # the premise: some direction really is computed sparsely, i.e. the
    # two density matrices are NOT the same object-for-object
    assert any(np.max(np.abs(dense[d]-sparse[d])) > 1e-12 for d in ds)
    mfd, mfs = get_mf(v, dense), get_mf(v, sparse)
    for d in mfd:
        assert np.max(np.abs(mfd[d]-mfs[d])) < 1e-12, d


def test_sparse_pairs_cover_every_entry_the_mean_field_reads(tmp_path,
                                                             monkeypatch):
    """Independent check of the mask itself, without any density matrix:
    perturbing the dense density matrix anywhere OUTSIDE the mask must
    leave the mean field unchanged, and the mask must not be the trivial
    everything mask."""
    monkeypatch.chdir(tmp_path)
    h, v = _system(nsuper=3)
    n = h.intra.shape[0]
    ds = [(0, 0, 0)] + list(v.keys())
    pairs = dm_sparse_pairs(v, ds, n, has_spin=h.has_spin)
    dense = h.get_density_matrix(ds=ds, nk=NK, T=T)
    ref = get_mf(v, dense)
    rng = np.random.RandomState(0)
    poked = dict()
    nout = 0
    for d in ds:
        mask = np.zeros((n, n), dtype=bool)
        rows, cols = pairs[d]
        mask[rows, cols] = True
        noise = rng.random_sample((n, n)) + 1j*rng.random_sample((n, n))
        noise[mask] = 0. # only outside the mask
        nout += np.sum(~mask)
        poked[d] = dense[d] + noise
    assert nout > 0, "the mask is everything, the test would be vacuous"
    out = get_mf(v, poked)
    for d in ref:
        assert np.max(np.abs(ref[d]-out[d])) < 1e-12, d
