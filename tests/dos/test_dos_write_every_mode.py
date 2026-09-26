"""write= decides whether DOS.OUT is written, in every mode of get_dos.

It used to be left in **kwargs and reach routines that do not take it:
mode="adaptive" raised TypeError inside get_bands, which it got twice
there, and mode="KPM" dropped it and wrote DOS.OUT whatever was asked."""

import os

import numpy as np
import pytest

from pyqula import geometry


@pytest.mark.parametrize("kw", [dict(mode="ED"), dict(mode="KPM"),
                                dict(use_kpm=True), dict(mode="adaptive"),
                                dict(mode="Green")])
def test_write_is_honoured(kw, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    h = geometry.chain().get_hamiltonian(has_spin=False)
    run = dict(energies=np.linspace(-1., 1., 5), nk=10, delta=0.2, **kw)
    (e, d) = h.get_dos(write=False, **run)
    assert not os.path.exists("DOS.OUT")
    assert np.all(np.isfinite(d))
    h.get_dos(write=True, **run)
    assert os.path.exists("DOS.OUT")


def test_write_is_honoured_with_an_operator_in_the_adaptive_mode(tmp_path,
                                                                 monkeypatch):
    """the operator branch is the one that went through get_bands"""
    monkeypatch.chdir(tmp_path)
    h = geometry.chain().get_hamiltonian(has_spin=True)
    (e, d) = h.get_dos(mode="adaptive", operator="sz", write=False, nk=10,
                       energies=np.linspace(-1., 1., 5), delta=0.2)
    assert not os.path.exists("DOS.OUT")
    assert np.all(np.isfinite(d))
