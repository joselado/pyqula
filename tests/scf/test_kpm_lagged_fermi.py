"""The Fermi level of the KPM mean-field loop, taken from the recursion
of the previous iteration (kpmtk/densitymatrix_kpm.py, LaggedFermi).

The iterates are not at the requested filling until the mean field stops
moving, so what has to hold is that the loop does not stop before they
are, and that the state it stops at is the one an exact search every
iteration converges to, up to the resolution of the Fermi level that
npol gives.
"""
import numpy as np
import pytest

from pyqula import geometry
from pyqula.kpmtk import densitymatrix_kpm as D
from pyqula.scftk.densitydensity_kpm import Vinteraction_kpm


def _island(sparse):
    """A spinful square island, whose ferromagnet at filling 0.4 is a
    metal, so that the Fermi level moves with the mean field"""
    g = geometry.square_lattice().get_supercell(4)
    g.dimensionality = 0
    return g.get_hamiltonian(has_spin=True, is_sparse=sparse)


_KW = dict(U=2., filling=0.4, mf="ferro", npol=150, mix=0.5, T=1e-2,
        maxerror=1e-6, maxite=200, verbose=0, write=False)


def _searched_every_iteration(monkeypatch):
    """The loop as it was before the lag: an exact search for every
    Hamiltonian, whose filling error is not part of the convergence check"""
    shift, update = D.LaggedFermi.shift, D.LaggedFermi.update
    def search(self, h):
        self.mu = None
        return shift(self, h)
    def no_error(self, h, trace):
        update(self, h, trace)
        self.error = 0.
    monkeypatch.setattr(D.LaggedFermi, "shift", search)
    monkeypatch.setattr(D.LaggedFermi, "update", no_error)


@pytest.mark.parametrize("sparse", [False, True])
def test_the_returned_hamiltonian_is_at_the_requested_filling(sparse):
    """The occupied fraction at the Fermi level of the returned Hamiltonian,
    read from the trace of its own expansion, is the requested filling to
    the tolerance of the loop"""
    np.random.seed(1)
    scf = Vinteraction_kpm(_island(sparse), **_KW)
    assert scf.converged
    scale, mus = D._kpm_trace(scf.hamiltonian, 1, None, _KW["npol"], None)
    xs, ys = D._dos_profile(mus, 4*_KW["npol"])
    count, _ = D._filling_count(scale, xs, ys, T=_KW["T"])
    assert abs(count(0.) - _KW["filling"]) < 1e-5


def test_the_lagged_fermi_level_reaches_the_state_of_a_search(monkeypatch):
    """Against an exact search every iteration the converged mean field
    moves by what the two estimates of the Fermi level differ by, which is
    the KPM resolution of the Fermi level (2e-3 here, moving the mean field
    by 1.5e-2) and not an error of either; a lag that settled in another
    state would move it by the exchange splitting, U times the moment,
    above 0.2 here"""
    np.random.seed(1)
    lagged = Vinteraction_kpm(_island(True), **_KW)
    _searched_every_iteration(monkeypatch)
    np.random.seed(1)
    searched = Vinteraction_kpm(_island(True), **_KW)
    assert lagged.converged and searched.converged
    assert abs(lagged.hamiltonian.fermi - searched.hamiltonian.fermi) < 5e-3
    diff = max(np.max(np.abs((lagged.mf[d] - searched.mf[d]).toarray()))
            for d in lagged.mf)
    assert diff < 3e-2
    m = np.real(lagged.dm[(0, 0, 0)].diagonal())
    assert np.max(np.abs(m[0::2] - m[1::2])) > 0.1 # the moment
