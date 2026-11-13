"""The 1D velocity and current are the Cartesian band speed along a1, so a
state gives the same number whatever cell describes it. They used to be the
derivative per cell, dE/dtheta, which scales as 1/|a1|: the same state of a
chain gave 1.1756 in the primitive cell and 0.5878 in a cell of two sites
(bug_audit_6.md #11)"""
import numpy as np
import pytest
from pyqula import geometry, algebra


@pytest.mark.parametrize("name", ["velocity", "current"])
@pytest.mark.parametrize("n", [1, 2, 3])
def test_chain_speed_does_not_depend_on_the_cell(name, n):
    """first-neighbor chain of lattice constant 1: |dE/dk| = sqrt(4-E^2)
    for every state, in the primitive cell and in supercells"""
    h = geometry.chain().get_hamiltonian(has_spin=False).get_supercell(n)
    op = h.get_operator(name)
    k = [0.137, 0., 0.]
    (es, ws) = np.linalg.eigh(np.array(algebra.todense(h.get_hk_gen()(k)),
            dtype=complex))
    for (e, w) in zip(es, ws.T):
        v = op.braket(w, k=k)
        assert abs(abs(v) - np.sqrt(4. - e**2)) < 1e-10, (n, e, v)
