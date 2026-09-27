import os

import numpy as np
import pytest

from pyqula import geometry
from pyqula import kdos


def _out_files(path):
    return sorted(f for f in os.listdir(path) if f.endswith(".OUT"))


def test_kdos_bands_takes_write_false_and_writes_nothing(tmp_path,monkeypatch):
    """kdos_bands forwarded **kwargs to get_bands(..., write=False), so
    passing write=False raised "got multiple values for keyword argument
    'write'"; and the k-path, built before that, wrote BANDLINES.OUT and
    KPOINTS_BANDS.OUT anyway. write=False has to write nothing and return
    the same three columns write=True reads back from KDOS_BANDS.OUT"""
    monkeypatch.chdir(tmp_path)
    h = geometry.honeycomb_lattice().get_hamiltonian()
    es = np.linspace(-3.,3.,40)
    silent = kdos.kdos_bands(h,nk=6,energies=es,write=False)
    assert _out_files(tmp_path) == []
    written = kdos.kdos_bands(h,nk=6,energies=es)
    assert "KDOS_BANDS.OUT" in _out_files(tmp_path)
    assert silent.shape == written.shape
    assert np.max(np.abs(silent-written)) < 1e-12


@pytest.mark.parametrize("build",[geometry.honeycomb_lattice,
                                  geometry.cubic_lattice])
def test_get_bands_write_false_writes_no_kpath_files(build,tmp_path,
        monkeypatch):
    """get_bands(write=False) skipped BANDS.OUT but still wrote the k-path
    files of the default 2d and 3d paths; the bands themselves must not
    depend on whether they are written"""
    monkeypatch.chdir(tmp_path)
    h = build().get_hamiltonian()
    (k0,e0) = h.get_bands(nk=30,write=False)
    assert _out_files(tmp_path) == []
    (k1,e1) = h.get_bands(nk=30)
    assert "BANDLINES.OUT" in _out_files(tmp_path)
    assert np.max(np.abs(k0-k1)) < 1e-12
    assert np.max(np.abs(e0-e1)) < 1e-12
