import numpy as np

from pyqula import geometry
from pyqula import ldos


def _written_map(e):
    return np.atleast_2d(np.genfromtxt("MULTILDOS/LDOS_"+str(e)+"_.OUT"))[:, 2]


def test_multi_ldos_tb_dense_branch_accepts_kdependent_operator(tmp_path, monkeypatch):
    """Regression check for ldos.multi_ldos_tb's dense-Hamiltonian branch:
    it used to call op(iw, k=k[0]) -- passing a bare scalar instead of the
    full k-vector -- which crashed any k-dependent operator (e.g. valley)
    with an IndexError, and even for k-independent operators the
    accumulator array was real-only while operator expectation values
    come back complex128, crashing with a casting error. Both the k-vector
    and the missing .real were fixed."""
    monkeypatch.chdir(tmp_path)  # multi_ldos_tb writes into ./MULTILDOS
    g = geometry.honeycomb_zigzag_ribbon(4)
    h = g.get_hamiltonian()
    h.add_peierls(0.05)
    op = h.get_operator("valley")
    out = ldos.multi_ldos(h, op=op, energies=np.linspace(-1.0, 1.0, 10), nk=4)
    d = np.genfromtxt("DOSMAP.OUT")
    assert d.shape[0] > 0
    assert np.all(np.isfinite(d[:, 2]))  # the LDOS column stayed real and finite


def test_the_operator_weighted_map_is_the_operator_weighted_ldos(tmp_path,
                                                                 monkeypatch):
    """The weight of an eigenstate used to be the vector A|psi>, which was
    then multiplied elementwise by |psi(i)|^2 -- a quantity that changes
    when |psi> is multiplied by a global phase, so not an observable at
    all. It is now the expectation value <psi|A|psi>, the same convention
    get_ldos uses in mode='arpack', which makes the map written at a given
    energy the array get_ldos returns at that energy -- for a k-dependent
    operator too."""
    monkeypatch.chdir(tmp_path)
    g = geometry.honeycomb_zigzag_ribbon(4)
    h = g.get_hamiltonian()
    h.add_peierls(0.05)
    e, delta, nk = 0.2, 0.05, 4
    op = "valley"  # k-dependent, the case this file exists for
    ref = np.array(h.get_ldos(e=e, delta=delta, mode="arpack", nk=nk,
                              nrep=1, write=False, operator=op)[2])
    assert np.max(np.abs(ref)) > 1e-3  # the reference is not trivially zero
    ldos.multi_ldos(h, operator=op, energies=np.array([e]), delta=delta,
                    nrep=1, nk=nk)
    got = _written_map(e)
    assert np.max(np.abs(got-ref)) < 1e-10*np.max(np.abs(ref))
