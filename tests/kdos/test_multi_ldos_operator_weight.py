import numpy as np

from pyqula import geometry
from pyqula import ldos


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
