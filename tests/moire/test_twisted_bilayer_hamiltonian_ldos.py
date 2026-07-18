import numpy as np

from pyqula import specialgeometry
from pyqula.specialhopping import twisted_matrix
from pyqula import ldos


def test_twisted_bilayer_hamiltonian_ldos_matches_reference(tmp_path, monkeypatch):
    """Regression check for a multicell twisted-bilayer Hamiltonian (custom
    mgenerator wrapping twisted_matrix) at the smallest commensurate moire
    index (n=1): the total LDOS must match the value recorded from a
    known-good run."""
    monkeypatch.chdir(tmp_path)  # ldos.ldos writes LDOS.OUT to cwd
    n = 1
    g = specialgeometry.twisted_bilayer(n)
    ti = 0.4
    fm = twisted_matrix(ti=ti)

    def fm2(rs1, rs2):
        m = fm(rs1, rs2)
        rm = rs1 - rs2
        diag = np.zeros((len(rs1), len(rs1)))
        for i in range(len(rs1)):
            diag[i, i] = 1.0
        return diag @ m

    h = g.get_hamiltonian(is_sparse=True, has_spin=False, is_multicell=True,
                           mgenerator=fm2)
    (x, y, d) = ldos.ldos(h)
    assert np.isclose(np.sum(d), 0.4232734102261245, atol=1e-6)
