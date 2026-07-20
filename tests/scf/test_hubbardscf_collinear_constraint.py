import numpy as np
import pytest

from pyqula import geometry
from pyqula import meanfield


@pytest.mark.slow
def test_hubbardscf_no_inplane_magnetism_constrains_to_z(tmp_path, monkeypatch):
    """Regression check for the collinear-Hubbard examples migrated off
    the legacy scftypes.hubbardscf (whose get_udxc step has an explicit
    `raise # nor working anymore`, so it always crashed): the modern
    meanfield.hubbardscf plus constrains=["no_inplane_magnetism"] is the
    replacement for the old collinear=True kwarg. Check the constrained
    SCF actually keeps the mean-field magnetization along z (in-plane
    Mx, My close to zero) while the unconstrained run does not have to."""
    monkeypatch.chdir(tmp_path)
    g = geometry.honeycomb_lattice()
    h0 = g.get_hamiltonian()
    h0.add_kane_mele(0.05)
    mf = meanfield.guess(h0, mode="antiferro", fun=lambda x: 1.0)
    scf = meanfield.hubbardscf(h0, nk=4, filling=0.5, U=3.0, mix=0.5,
            mf=mf, constrains=["no_inplane_magnetism"])
    h = scf.hamiltonian
    mx = h.get_vev("mx", nk=4)
    my = h.get_vev("my", nk=4)
    assert np.max(np.abs(mx)) < 1e-6
    assert np.max(np.abs(my)) < 1e-6
