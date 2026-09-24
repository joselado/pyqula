import inspect

import numpy as np
import pytest

from pyqula import geometry


class _TooManyIterations(Exception):
    pass


def _count_iterations(monkeypatch, cap=1100):
    """Count the iterations of both numpy SCF loops through the mixing step
    each of them takes once per iteration, and raise past `cap`, so that a
    loop with no limit fails here instead of never returning."""
    from pyqula.scftk import densitydensity as dd
    calls = [0]
    orig = dd.mix_mf
    def counting(*args, **kwargs):
        calls[0] += 1
        if calls[0] > cap: raise _TooManyIterations
        return orig(*args, **kwargs)
    monkeypatch.setattr(dd, "mix_mf", counting)
    return calls


def _chain():
    # the user guide's example of a mean field that never converges: a
    # ferromagnetic chain at filling 0.2 on an nk=8 mesh
    return geometry.chain().get_hamiltonian(has_spin=True)


@pytest.mark.parametrize("route", ["get_szsz_mean_field_hamiltonian",
    "get_mean_field_hamiltonian"])
def test_a_mean_field_that_never_converges_returns_by_default(monkeypatch,
        route):
    """maxite used to default to None, no limit, on every numpy SCF loop, so
    a calculation that never converges never returned. It now stops at 1000
    iterations and returns None, both through the density-density loop
    (SzSz) and through VJinteraction's own loop."""
    calls = _count_iterations(monkeypatch)
    h = _chain()
    out = getattr(h, route)(J1=-2.0, filling=0.2, mf="ferroZ", nk=8,
            mix=0.3)
    assert out is None
    assert 1000 <= calls[0] <= 1001


def test_a_spinless_mean_field_that_never_converges_returns_by_default(
        monkeypatch):
    """The spinless density-density route: at fixed mu=0 the Hartree shift
    holds this chain at filling ~0.32, a metal whose level at mu flips its
    occupation on the nk=6 mesh at T~0."""
    calls = _count_iterations(monkeypatch)
    h = geometry.bichain().get_hamiltonian(has_spin=False)
    out = h.get_mean_field_hamiltonian(V1=3.0, mu=0.0, nk=6, mf="CDW")
    assert out is None
    assert 1000 <= calls[0] <= 1001


def test_maxite_none_still_means_no_limit(monkeypatch):
    """maxite=None, given explicitly, keeps meaning no limit."""
    _count_iterations(monkeypatch)
    with pytest.raises(_TooManyIterations):
        _chain().get_mean_field_hamiltonian(J1=-2.0, filling=0.2,
                mf="ferroZ", nk=8, mix=0.3, maxite=None)


def test_the_other_numpy_loops_default_to_the_same_limit():
    """The kpm loop is too slow to drive past 1000 iterations here; its
    default is the same one."""
    from pyqula.scftk.densitydensity_kpm import generic_densitydensity_kpm
    from pyqula.scftk.densitydensity import generic_densitydensity
    from pyqula.scftk.spinspin import Jinteraction
    for fun in (generic_densitydensity_kpm,
            generic_densitydensity, Jinteraction):
        assert inspect.signature(fun).parameters["maxite"].default == 1000
