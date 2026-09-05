"""Four code paths that could not run at all: an import that never
resolved, an import that was never made, and a keyword collision. Each is
covered here by a call that used to raise, plus a check that the value it
now returns is the right one."""

import numpy as np
import pytest

from pyqula import geometry, dos, filling, densitymatrix


def _island(n=4):
    g = geometry.chain().supercell(n)
    g.dimensionality = 0
    h = g.get_hamiltonian(has_spin=True)
    h.add_exchange([0.3, 0.5, 0.2])
    h.add_onsite(0.2)
    return h


def test_get_dm_vev_resolves_and_keeps_the_sign():
    """h.get_dm_vev delegated with `from . import get_dm_vev`, a package
    attribute src/pyqula/__init__.py deliberately never populates, so it
    raised ImportError for every argument -- and behind it vev.py used a
    Python-2 absolute `from operators import Operator`, so the module
    could not be imported either. fbee7c9 filed half of its
    density-matrix transpose fix in that unreachable module, so check the
    sign of the imaginary operator it was about."""
    h = _island()
    (es, ws) = h.get_eigenvectors()
    for name in ["sx", "sy", "sz"]:
        m = h.get_operator(name).get_matrix()
        ref = sum([np.conjugate(w).dot(m @ w) for (e, w) in zip(es, ws)
                   if e < 0.]).real
        assert abs(h.get_dm_vev(m).real - ref) < 1e-8, name


def test_dos_ewindow_runs_in_two_dimensions(tmp_path, monkeypatch):
    """e481ffb routed dos2d_ewindow through hk_matrix_batch but left the
    local import naming only peigvalsh, so it raised NameError before
    doing any work. Its 1D sibling imports both."""
    monkeypatch.chdir(tmp_path)
    h = geometry.honeycomb_lattice().get_hamiltonian()
    dos.dos_ewindow(h, use_green=False, nk=6,
                    energies=np.linspace(-1., 1., 20))
    assert (tmp_path / "DOS.OUT").exists()


def test_full_dm_accepts_delta_as_an_alias_for_T():
    """full_dm's smearing is named T but is forwarded as `delta=T`, so a
    caller who spelled it `delta` -- the name the rest of the library uses
    -- collided with it."""
    h = _island(3)
    by_t = densitymatrix.full_dm(h, T=1e-2)
    by_delta = densitymatrix.full_dm(h, delta=1e-2)
    assert np.max(np.abs(np.array(by_t) - np.array(by_delta))) < 1e-12
    with pytest.raises(TypeError):
        densitymatrix.full_dm(h, T=1e-2, delta=1e-3)  # the ambiguity


def test_individual_filling_targets_the_right_occupancy():
    """set_filling(average=False) died in the delta collision above. Once
    it runs, `filling` is a fraction of all the states (check_filling's
    convention, 0..1) while get_vev returns an occupancy per site, which
    runs to 2 for a spinful Hamiltonian -- the solver used to compare the
    two directly and aim at half the occupancy it should."""
    h = _island(3)
    filling.set_filling(h, filling=0.5, average=False)
    occ = h.get_vev()
    assert np.max(np.abs(occ - 1.0)) < 5e-2  # half filling, spinful: 1/site
