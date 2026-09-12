import numpy as np

from pyqula import geometry
from pyqula.scftk.attractive_hubbard_spinless import attractive_hubbard


def _chain():
    h = geometry.chain().get_hamiltonian()
    h.remove_spin()
    return h


def test_converged_pairing_does_not_depend_on_the_mixing(tmp_path,
                                                         monkeypatch):
    """A converged self-consistent solution is a fixed point of the gap
    equation, so it cannot depend on the linear-mixing parameter used to
    reach it -- the standard invariance check for an SCF loop.

    The convergence test used to be taken against the ALREADY-MIXED
    vector, so the residual it compared was (1-mix) times the true one:
    the effective tolerance was maxerror/(1-mix), and at mix=1.0 it was
    identically zero, so the loop stopped after a single iteration and
    returned the random initial guess mapped once."""
    monkeypatch.chdir(tmp_path)
    out = []
    for mix in (1.0, 0.9, 0.5):
        np.random.seed(0)
        scf = attractive_hubbard(_chain(), g=-2.0, nk=6, mix=mix,
                maxerror=1e-6, maxite=5000)
        assert scf.converged
        out.append(np.array(scf.hamiltonian.intra)[0, 1].real)
    assert max(out)-min(out) < 1e-5, out


def test_maxite_stops_a_run_that_cannot_reach_the_tolerance(tmp_path,
                                                            monkeypatch):
    """The loop had no iteration cap at all, so a tolerance it cannot
    reach is an infinite loop -- the three sibling SCF loops in the
    package all take maxite and report converged=False instead."""
    monkeypatch.chdir(tmp_path)
    np.random.seed(0)
    scf = attractive_hubbard(_chain(), g=-2.0, nk=6, mix=0.9,
            maxerror=1e-14, maxite=3)
    assert scf.converged is False
