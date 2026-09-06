import numpy as np
import scipy.linalg as lg

from pyqula import algebra, klist, specialhamiltonian


def test_twisted_bilayer_graphene_bands_match_dense_diagonalization(tmp_path,
                                                                    monkeypatch):
    """specialhamiltonian.twisted_bilayer_graphene at the smallest
    commensurate moire index (n=1): the sparse (ARPACK) band energies along
    G-K-M-K'-G must be the eight eigenvalues closest to zero of the dense
    Bloch matrix at the same k-points.

    This used to pin a hardcoded sum instead. That is not portable -- the
    recorded value was updated once for an environment that produced it, and
    then no longer matched anywhere, leaving the test red while the physics
    was fine. Diagonalizing densely on the spot checks the same thing without
    depending on which machine recorded the number.
    """
    monkeypatch.chdir(tmp_path)  # get_bands writes BANDS.OUT to cwd
    kpath = ["G", "K", "M", "K'", "G"]
    h = specialhamiltonian.twisted_bilayer_graphene(n=1, ti=0.4, has_spin=False)
    h.set_filling(0.5, nk=1)
    (k, e) = h.get_bands(num_bands=8, kpath=kpath, nk=8)
    assert e.shape == (104,)
    ks = klist.get_kpath(h.geometry, kpath=kpath, nk=8)
    assert len(ks)*8 == len(e)
    hk = h.get_hk_gen()
    ref = []
    for ki in ks:
        m = np.array(algebra.todense(hk(ki)), dtype=np.complex128)
        ev = lg.eigvalsh(m)
        ref += sorted(ev[np.argsort(np.abs(ev))[:8]])  # the eight nearest zero
    assert np.allclose(e, ref, atol=1e-10), np.max(np.abs(e - np.array(ref)))
    # and the flat bands are where a twisted bilayer puts them
    assert np.max(np.abs(e)) < 1.5
