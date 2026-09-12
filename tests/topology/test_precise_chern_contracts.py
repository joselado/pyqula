import numpy as np
import pytest

from pyqula import geometry, topology


def _haldane(t2=0.05):
    g = geometry.honeycomb_lattice()
    h = g.get_hamiltonian(has_spin=False)
    h.add_haldane(t2)
    return h


def test_precise_chern_refuses_an_operator_it_would_have_ignored(tmp_path,
                                                                 monkeypatch):
    """precise_chern's default mode='Wilson' calls berry_curvature without
    the operator, so an operator-projected Chern number asked for in that
    mode silently came back as the unprojected one -- the same class as the
    topology.get_operator dispatch gap, but on the entry point that does
    not auto-switch. It names the requirement now.

    mode='Green' is the mode that does honour it, and is left reachable;
    it is not exercised here because precise_chern has no k-mesh knob and
    a Green-mode adaptive integration of this model runs for minutes."""
    monkeypatch.chdir(tmp_path)
    # spinful, so that "sz" is a legitimate operator here and the refusal
    # under test is the mode one rather than the Hilbert-space guard
    g = geometry.honeycomb_lattice()
    hs = g.get_hamiltonian(has_spin=True)
    hs.add_haldane(0.05)
    with pytest.raises(ValueError, match="mode='Green'"):
        topology.precise_chern(hs, operator="sz")
    # without an operator the Wilson mode is untouched, and still right
    c = topology.precise_chern(_haldane())
    assert abs(abs(c) - 1.0) < 0.1


def test_precise_chern_says_why_it_has_no_nk(tmp_path, monkeypatch):
    """Every sibling Chern path takes nk; these two integrate adaptively
    and cannot. They used to answer with a bare
    `TypeError: unexpected keyword argument 'nk'`, which names the keyword
    but not the reason or the alternative."""
    monkeypatch.chdir(tmp_path)
    h = _haldane()
    with pytest.raises(ValueError, match="adaptively"):
        topology.precise_chern(h, nk=10)
    with pytest.raises(ValueError, match="adaptively"):
        topology.precise_spin_chern(h, nk=10)
