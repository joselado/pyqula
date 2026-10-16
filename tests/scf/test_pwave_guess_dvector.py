import numpy as np

from pyqula import geometry, meanfield


def _pairing(h, d):
    h0 = h.copy().get_multicell()
    for t in h0.hopping: t.m *= 0.
    h0.intra *= 0.
    h0.add_pairing(delta=0.1, mode="pwave", d=d)
    return h0.get_hopping_dict()


def test_the_pwave_guess_seeds_equal_and_opposite_spin_triplets():
    """mf="pwave" used to seed d along z only, the opposite-spin triplet,
    a poor start for an equal-spin state. It now seeds d along (1,1,1):
    exactly the sum of the three d components over sqrt(3), where d along x
    or y and d along z occupy different entries of the Nambu blocks."""
    h = geometry.triangular_lattice().get_hamiltonian()
    h.setup_nambu_spinor()
    mf = meanfield.guess(h, mode="pwave")
    gs = [_pairing(h, d) for d in ([1, 0, 0], [0, 1, 0], [0, 0, 1])]
    for k in mf:
        total = sum(np.array(g[k]) for g in gs)/np.sqrt(3.)
        assert np.max(np.abs(np.array(mf[k]) - total)) < 1e-14
    k = (1, 0, 0)
    equal = np.abs(np.array(gs[0][k])) > 1e-12
    opposite = np.abs(np.array(gs[2][k])) > 1e-12
    assert equal.any() and opposite.any() and not (equal & opposite).any()
    m = np.abs(np.array(mf[k])) > 1e-12
    assert m[equal].all() and m[opposite].all()
