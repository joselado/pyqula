import os

import numpy as np
import pytest

from pyqula import geometry
from pyqula import filewrite
from pyqula import topology
from pyqula import islands
from pyqula import dos
from pyqula import ldos
from pyqula import spectrum
from pyqula import bandstructure


@pytest.fixture(autouse=True)
def unset_switch(monkeypatch):
    """every test starts with the switch unset, and cannot leave it set for
    the rest of the suite, whatever it does to it"""
    monkeypatch.setattr(filewrite,"_write",None)


def _files(path):
    return sorted(os.listdir(path))


def _flat(out):
    """every number a routine returned, real and imaginary parts, as one
    real array"""
    if np.isscalar(out): out = [out]
    if isinstance(out,np.ndarray): out = [out]
    z = np.concatenate([np.ravel(np.array(o,dtype=complex)) for o in out])
    return np.concatenate([z.real,z.imag])


def _haldane():
    h = geometry.honeycomb_lattice().get_hamiltonian(has_spin=False)
    h.add_haldane(0.1)
    return h


def _island(t2=0.1):
    """a small Haldane island, a finite (0d) system"""
    g = islands.get_geometry(name="honeycomb",n=3,nedges=6,rot=0.0,
                             clean=False)
    h = g.get_hamiltonian(has_spin=False)
    h.add_haldane(t2)
    return h


def _mean_field(w):
    """a spinless chain with a first-neighbor repulsion, from a seeded
    random guess, which is the loop that saves MF.pkl"""
    np.random.seed(0)
    h = geometry.chain().get_hamiltonian(has_spin=False)
    hmf = h.get_mean_field_hamiltonian(V1=1.0,filling=0.5,nk=4,mf="random",
            load_mf=False,write=w)
    return hmf.get_hk_gen()([0.,0.,0.])


def _kane_mele():
    h = geometry.honeycomb_lattice().get_hamiltonian()
    h.add_kane_mele(0.1)
    return h


# name -> call(write). Each is called once with the switch set to False and
# no write=, and once with write=True, and must return the same numbers
_routines = {
    "bands": lambda w: _haldane().get_bands(nk=20,write=w),
    "dos": lambda w: _haldane().get_dos(nk=10,
            energies=np.linspace(-1.,1.,20),write=w),
    "kdos_bands": lambda w: _haldane().get_kdos_bands(nk=5,
            energies=np.linspace(-1.,1.,20),write=w),
    "chern": lambda w: _haldane().get_chern(nk=8,write=w),
    "spin_chern": lambda w: _kane_mele().get_spin_chern(nk=10,write=w),
    "z2": lambda w: topology.z2_invariant(_kane_mele(),nk=10,nt=10,write=w),
    "berry_path": lambda w: topology.get_berry_curvature_path(_haldane(),
            nk=10,write=w),
    "qpi": lambda w: _haldane().get_qpi(nk=6,
            energies=np.linspace(-1.,1.,3),write=w),
    "mean_field": _mean_field,
    "density": lambda w: _haldane().get_density(nk=4,write=w),
    "multildos": lambda w: _haldane().get_multildos(nk=4,
            energies=np.linspace(-1.,1.,3),write=w),
    "dos0d": lambda w: dos.dos0d(_island(),energies=np.linspace(-1.,1.,20),
            write=w),
    "ldos1d": lambda w: ldos.ldos1d(
            geometry.chain().get_hamiltonian(has_spin=False),write=w),
    "real_space_chern": lambda w: topology.real_space_chern(_island(),
            write=w),
    "real_space_vev": lambda w: spectrum.real_space_vev(_island(),write=w),
    "lowest_bands": lambda w: bandstructure.lowest_bands(_island(),nbands=4,
            write=w),
    }


@pytest.mark.parametrize("name",list(_routines))
def test_switched_off_writes_nothing_and_returns_the_same(name,tmp_path,
        monkeypatch):
    (off,on) = (tmp_path/"off",tmp_path/"on")
    off.mkdir() ; on.mkdir()
    filewrite.set_write(False)
    monkeypatch.chdir(off)
    silent = _routines[name](None) # no write= in the call
    assert _files(off) == [], name
    monkeypatch.chdir(on)
    written = _routines[name](True) # the call wins over the switch
    assert _files(on) != [], name
    (a,b) = (_flat(silent),_flat(written))
    assert a.shape == b.shape
    assert np.max(np.abs(a-b)) < 1e-10, name


def test_explicit_false_wins_over_the_switch_set_to_true(tmp_path,
        monkeypatch):
    monkeypatch.chdir(tmp_path)
    filewrite.set_write(True)
    _haldane().get_bands(nk=10,write=False)
    assert _files(tmp_path) == []


def test_switched_on_writes_even_where_the_default_is_not_to(tmp_path,
        monkeypatch):
    """chern_density writes nothing by default; set_write(True) makes it"""
    monkeypatch.chdir(tmp_path)
    topology.chern_density(_haldane(),nk=2,es=np.linspace(-1.,1.,4))
    assert _files(tmp_path) == []
    filewrite.set_write(True)
    topology.chern_density(_haldane(),nk=2,es=np.linspace(-1.,1.,4))
    assert "CHERN_DENSITY.OUT" in _files(tmp_path)


def test_unset_keeps_each_routine_default(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)
    filewrite.set_write(False)
    filewrite.set_write(None)
    _haldane().get_bands(nk=10)
    assert "BANDS.OUT" in _files(tmp_path)


def test_resolution_order():
    assert filewrite.resolve(None,True) is True
    assert filewrite.resolve(None,False) is False
    filewrite.set_write(False)
    assert filewrite.get_write() is False
    assert filewrite.resolve(None,True) is False
    assert filewrite.resolve(True,False) is True
    filewrite.set_write(True)
    assert filewrite.resolve(None,False) is True
    assert filewrite.resolve(False,True) is False


@pytest.mark.parametrize("value",["yes",1,0,"False"])
def test_set_write_takes_only_true_false_or_none(value):
    with pytest.raises(ValueError, match="True, False or None"):
        filewrite.set_write(value)
    assert filewrite.get_write() is None
