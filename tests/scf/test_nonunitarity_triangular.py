import numpy as np
import pytest

from pyqula import geometry


def test_up_up_pairing_non_unitarity_points_along_z():
    """Sign/convention check for the spin-triplet non-unitarity vector.

    The convention is q = i*(d x d^*), fixed by Delta Delta^dag =
    |d|^2 + q.sigma with Delta = i*(d.sigma)*sigma_y, so q is the spin
    moment of the Cooper pairs. A d-vector proportional to (1,i,0) is a
    pure up-up pairing, whose pairs carry spin +z; the opposite ordering
    of the cross product, i*(d^* x d), would return -q and report the
    pairs as spin-down."""
    g = geometry.chain()
    h = g.get_hamiltonian()
    h.setup_nambu_spinor()
    h.add_pairing(delta=0.3, mode="pwave", d=[1., 1j, 0.])
    q = h.get_dvector_non_unitarity(nk=20)[0]
    assert q[2] > 0.  # pairs are up-up, so q is along +z
    assert np.allclose(q[0:2], 0., atol=1e-8)  # and has no in-plane part


@pytest.mark.slow
def test_triplet_scf_dvector_non_unitarity_matches_reference(tmp_path, monkeypatch):
    """Regression check for a non-collinear superconducting mean-field
    calculation (Nambu spinor, random init) on a ferromagnetic triangular
    lattice: the spin-triplet d-vector non-unitarity must match the value
    recorded from a known-good run, and must be parallel to the
    magnetization -- the pairs form in the majority band, so the pair spin
    follows the magnetic moment. Marked slow: the SCF convergence itself
    (not the k-mesh) drives the runtime here -- an explicit nk=4 (vs. the
    default nk=8) barely changed it."""
    monkeypatch.chdir(tmp_path)
    g = geometry.triangular_lattice()
    h = g.get_hamiltonian()
    h.add_exchange([3., 3., 3.])
    h.setup_nambu_spinor()
    h = h.get_mean_field_hamiltonian(V1=-1.0, filling=0.3, mf="random", nk=4)
    d = h.get_dvector_non_unitarity()
    assert np.allclose(d, -0.02182528, atol=1e-4)
    # q is parallel (not antiparallel) to the magnetization
    m = np.array(h.get_magnetization())
    assert np.dot(d[0], m[0])/np.linalg.norm(d[0])/np.linalg.norm(m[0]) > 0.99
