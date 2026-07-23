import numpy as np
import pytest

from pyqula import geometry
from pyqula import heterostructures


def _normal_sc_junction(delta, transparency, rashba=0., zeeman=None):
    """One plain normal lead + one (possibly weakly) superconducting lead,
    exactly the setup used throughout pyqula's existing Andreev-reflection
    examples (e.g. examples/transport/andreev_reflection/main.py), optionally
    dressed with spin-orbit coupling and a Zeeman field on the superconducting
    side to make the Hamiltonian non-trivial (closer to the nanowire model
    the Floquet-Keldysh paper actually studies)."""
    g = geometry.chain()
    h1 = g.get_hamiltonian()  # normal lead
    h2 = g.get_hamiltonian()
    if rashba:
        h2.add_rashba(rashba)
    if zeeman is not None:
        h2.add_zeeman(zeeman)
    h2.add_swave(delta)
    HT = heterostructures.build(h1, h2)
    HT.set_coupling(transparency)
    HT.delta = min(1e-4, 0.02*delta)
    return HT


@pytest.mark.parametrize("transparency", [0.3, 0.7, 1.0])
@pytest.mark.parametrize("delta,rashba,zeeman", [
    (0.3, 0., None),               # plain Andreev reflection (moderate gap)
    (0.05, 0., None),               # small gap
    (0.02, 0.4, [0.3, 0., 0.]),     # small gap + spin-orbit + Zeeman
])
def test_floquet_keldysh_linear_response_matches_equilibrium_andreev(
        transparency, delta, rashba, zeeman):
    """For a normal-superconductor junction (one lead non-superconducting,
    the other with a pairing gap that can be made arbitrarily small), the
    zero-bias slope dI/dV of the new Floquet-Keldysh DC current
    (Heterostructure.get_dc_current) must match the existing equilibrium
    Andreev conductance (Heterostructure.didv, via transporttk.didv.didv_BdG)
    -- the two are the same physical quantity (linear-response subgap
    conductance) computed by two independent code paths."""
    HT = _normal_sc_junction(delta, transparency, rashba=rashba, zeeman=zeeman)

    # equilibrium Andreev conductance; avoid evaluating exactly at E=0, a
    # coincidental numerical singularity of this particular chain geometry
    Gref = HT.didv(energy=1e-4)

    dv = 0.02*delta
    Icalc = HT.get_dc_current(dv, nmax=10, nmax_max=50, tol=1e-4)
    slope = Icalc/dv

    assert abs(slope-Gref) < 0.05*max(Gref, 1e-8)
