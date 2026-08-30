import numpy as np
import pytest

from pyqula import islands
from pyqula import interactions
from pyqula import scftypes


@pytest.mark.slow
def test_scf_graphene_island_antiferro_magnetization_matches_reference(tmp_path, monkeypatch):
    """Regression check for an antiferromagnetic Hubbard SCF calculation on
    a small honeycomb island (n=2 instead of 4) with a Zeeman field: the
    summed magnetization must match the value recorded from a known-good
    run. Marked slow: SCF convergence itself (not the island size) drives
    the runtime.

    The reference was re-recorded once the interaction actually reached the
    SCF: this test used to call the old scftypes.selfconsistency interface
    as `selfconsistency(h, g=1.0, mode="U")`, and neither `g` nor `mode` is
    part of the signature that name is now aliased to, so both were dropped
    and the loop ran with U=0. The old value, 13.2, is exactly the bare
    Zeeman field it started from (33 sites x 0.4) -- the test was measuring
    its own input. `fun=None` in get_hamiltonian was dead in the same way.

    Recorded a second time when get_magnetization's default changed from
    the mean-field exchange field to the moment it induces; the value here
    is now the summed moment, which points against the applied field.
    """
    monkeypatch.chdir(tmp_path)
    g = islands.get_geometry(name="honeycomb", n=2, nedges=3, rot=0.0)
    h = g.get_hamiltonian(has_spin=True)
    h.add_zeeman([0., .4, 0.])
    mf = scftypes.guess(h, mode="antiferro")
    scf = scftypes.selfconsistency(h, filling=0.5, U=1.0, mix=0.9, mf=mf)
    m = scf.hamiltonian.get_magnetization()
    # -3.0 along y, i.e. opposing the +0.4 y Zeeman field, as a moment
    # must. The field here is along y, the one component whose sign
    # spectrum.ev used to flip, so this doubles as a check on that
    assert np.isclose(np.sum(m), -2.9974891228532368, atol=1e-4)
