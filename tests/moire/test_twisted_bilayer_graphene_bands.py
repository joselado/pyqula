import numpy as np
import scipy.linalg as lg

from pyqula import algebra, klist, specialhamiltonian


def test_twisted_bilayer_graphene_bands_match_dense_diagonalization(tmp_path,
                                                                    monkeypatch):
    """specialhamiltonian.twisted_bilayer_graphene at the smallest
    commensurate moire index (n=1): the sparse (ARPACK) band energies along
    G-K-M-K'-G must be the eight eigenvalues closest to zero of the dense
    Bloch matrix at the same k-points.

    The band energies themselves are still pinned, below, since the dense
    comparison only checks the eigensolver and would pass for a wrong
    Hamiltonian too. The pinned value is the one this test was written with;
    `6c8cfbc` replaced it with -13.738703648103538, which no commit of this
    repository reproduces -- not the 0.0.94 release, not HEAD, and not
    `6c8cfbc`'s own source -- so the test had been red ever since.
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
    # and the Hamiltonian itself is the one this was recorded against
    assert np.isclose(np.sum(e), -13.735331753001446, atol=1e-6), np.sum(e)
