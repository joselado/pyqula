"""The device path of the general nonlinear conductivity must reproduce the
CPU one.

Under pyqula.gpu.set_gpu(True) src/pyqula/conductivitytk/nonlinearkernel.py
runs the density-matrix recursion compiled with jax, vmapped over the
frequency rows, with the vertices built on the device and the eigenpairs
from htk/eigenvectorsjax.peigh_gpu. The recursion itself is the same
function on both backends, so a wrong vertex hopping, a wrong padding of
the last k-batch or frequency chunk, or a switch that never reached the
device would still give a plausible tensor; these tests diff against the
numpy path instead, and a spy checks that the device eigensolver was
called. On a machine with no GPU they run through jax's CPU backend, which
checks correctness and says nothing about speed.

In single precision the 1/(w1...wn) prefactor cancels Fermi-sea terms of
order one, so the float32 error grows as (t/|w + i delta|)^n and the
tolerance is scaled with it (future_development/nonlinear_response.md,
"Single precision").
"""
import itertools
import warnings

import numpy as np
import pytest

from pyqula import geometry
from pyqula import gpu
from pyqula.conductivitytk import nonlinearkernel as nlk
from pyqula.htk import eigenvectorsjax
from testutils import gpu_backend


_eps32 = np.finfo(np.float32).eps


def _gapped_honeycomb(m=0.5):
    """Honeycomb lattice with a staggered onsite mass, gap 2m. No
    inversion, so the second order is finite"""
    h = geometry.honeycomb_lattice().get_hamiltonian(has_spin=False)
    h.add_sublattice_imbalance(m)
    return h


def _in_plane(order):
    """Every in-plane component of the order-n tensor"""
    return ["".join(c) for c in itertools.product("xy",repeat=order+1)]


def _spy(monkeypatch):
    """Record the precision and the number of k-points of every call of
    the device eigensolver"""
    calls = []
    f = eigenvectorsjax.peigh_gpu
    def spy(hks,**kwargs):
        calls.append((kwargs.get("prec"),len(hks)))
        return f(hks,**kwargs)
    monkeypatch.setattr(eigenvectorsjax,"peigh_gpu",spy)
    return calls


def _device(f):
    """f() under set_gpu(True), with the no-GPU warning of a machine
    without a card silenced"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with gpu_backend(True):
            return f()


@pytest.mark.parametrize("order,frequencies",[(2,None),(2,(1,-2)),
        (3,None),(3,(1,2,-1))])
def test_gpu_path_matches_numpy(order,frequencies,monkeypatch):
    """Double precision on the device equals numpy to 1e-12, for harmonic
    generation and for a general frequency tuple, whose orderings of the
    fields are all distinct. nk=7 with 49 k-points and 5 frequencies
    with k-batches of 15 points and frequency chunks of 3 rows makes the
    device pad both the last k-batch and the last frequency chunk. The
    last chunk holds two different frequencies and one pad row, so taking
    the wrong rows of the padded chunk gives a wrong number; with a
    single real row and its copy as the pad, either slice would agree"""
    h = _gapped_honeycomb()
    kw = dict(order=order,energies=[0.2,0.6,1.0,1.5,2.0],
            frequencies=frequencies,nk=7,delta=0.1,
            components=_in_plane(order))
    (w,s0) = nlk.nonlinear_conductivity(h,**kw)
    s0 = np.array(list(s0.values()))
    assert np.max(np.abs(s0))>1e-2 # not a tensor forced to zero
    calls = _spy(monkeypatch)
    monkeypatch.setattr(nlk,"_device_chunks",lambda *args: (15,3))
    (w,s1) = _device(lambda: nlk.nonlinear_conductivity(h,**kw))
    s1 = np.array(list(s1.values()))
    assert calls==[("double",15)]*3 + [("double",4)] # 49 k-points
    assert np.max(np.abs(s1-s0))<1e-12*np.max(np.abs(s0))


@pytest.mark.parametrize("order",[2,3])
def test_single_precision_matches_double(order,monkeypatch):
    """Single precision on the device against numpy in double, at the
    frequencies where single precision means something.

    Two errors add up. The 1/(w1...wn) prefactor cancels terms of order
    (t/|w + i delta|)^n in the natural units e^(n+1) a^(n+1-d)/t^(n-1),
    each carrying an eps32 rounding error, and the single-precision
    eigenvectors put a relative error of a few eps32 on sigma itself
    (with the eigenpairs in double it drops from 1e-6 to 3e-7 of the
    largest component at w=1 and w=2). So the tolerance at each frequency
    is 20 eps32 ((t/|w + i delta|)^n + max|sigma|), with t=1. TF32
    products, a 1e-3 relative error, would miss it by a factor of 50 or
    more.

    The error of the single-precision eigenpairs alone already makes the
    result differ from numpy, so a second run takes the eigenpairs in
    double and keeps prec="single" for the rest. Its error is then that
    of the recursion, the vertices and the traces, which in float32 stays
    far above the 1e-12 of a recursion run in double"""
    h = _gapped_honeycomb()
    delta = 0.1
    E = np.array([0.2,1.0,2.0])
    kw = dict(order=order,energies=E,nk=8,delta=delta,
            components=_in_plane(order))
    (w,s0) = nlk.nonlinear_conductivity(h,**kw)
    s0 = np.array(list(s0.values())).T # (nw,ncomp)
    calls = _spy(monkeypatch)
    (w,s1) = _device(lambda: nlk.nonlinear_conductivity(h,prec="single",
            **kw))
    s1 = np.array(list(s1.values())).T
    assert [c[0] for c in calls]==["single"]
    scale = np.max(np.abs(s0),axis=1)
    assert np.all(scale>1e-2)
    tol = 20*_eps32*((1./np.abs(E + 1j*delta))**order + scale)
    err = np.max(np.abs(s1-s0),axis=1)
    assert np.all(err<tol),(err,tol)
    assert np.all(err>0.) # it did run in single precision
    f = eigenvectorsjax.peigh_gpu # the spy, which calls the real solver
    def double_eigenpairs(hks,**kwargs):
        kwargs["prec"] = "double"
        return f(hks,**kwargs)
    monkeypatch.setattr(eigenvectorsjax,"peigh_gpu",double_eigenpairs)
    (w,s2) = _device(lambda: nlk.nonlinear_conductivity(h,prec="single",
            **kw))
    s2 = np.array(list(s2.values())).T
    assert calls[-1][0]=="double" # the eigenpairs were in double
    err2 = np.max(np.abs(s2-s0),axis=1)
    assert np.all(err2<tol),(err2,tol)
    assert np.all(err2>1e-9*scale),err2 # the recursion ran in complex64


def test_single_precision_warns_at_low_frequency():
    """At a frequency below the broadening the float32 error grows as
    (t/delta)^n, and the kernel says so"""
    h = _gapped_honeycomb()
    def f():
        with pytest.warns(UserWarning,match="single"):
            return nlk.nonlinear_conductivity(h,order=2,energies=[0.],nk=4,
                    components="xxx",prec="single")
    (w,s) = _device(f)
    assert np.all(np.isfinite(s))


def test_precision_option_checked():
    """An unknown precision lists the accepted ones, and single precision
    on the CPU, whose recursion is double only, is refused"""
    h = _gapped_honeycomb()
    kw = dict(order=2,energies=[1.],nk=4,components="xxx")
    with pytest.raises(ValueError,match="single.*double"):
        nlk.nonlinear_conductivity(h,prec="half",**kw)
    with pytest.raises(ValueError,match="single.*double"):
        _device(lambda: nlk.nonlinear_conductivity(h,prec="half",**kw))
    assert not gpu.get_gpu()
    with pytest.raises(NotImplementedError,match="single"):
        nlk.nonlinear_conductivity(h,prec="single",**kw)


def test_gpu_switch_restores(monkeypatch):
    """After set_gpu(False) the numpy path runs again and gives the same
    numbers, bit for bit, as before the device was used"""
    h = _gapped_honeycomb()
    kw = dict(order=2,energies=[0.5,1.5],nk=6,components="xxx")
    (w,s0) = nlk.nonlinear_conductivity(h,**kw)
    (w,s1) = _device(lambda: nlk.nonlinear_conductivity(h,**kw))
    assert not gpu.get_gpu()
    calls = _spy(monkeypatch)
    (w,s2) = nlk.nonlinear_conductivity(h,**kw)
    assert calls==[] # back on the CPU
    assert np.array_equal(s0,s2)
    assert np.allclose(s1,s0,rtol=1e-12,atol=0.)


def test_empty_energy_list_on_the_device():
    """An empty list of energies gives empty arrays on the device, as on
    the CPU; the frequency chunk of the device sweep is never zero rows"""
    h = _gapped_honeycomb()
    kw = dict(order=2,energies=[],nk=4,delta=0.1,components=["xxx","yxy"])
    (w0,s0) = nlk.nonlinear_conductivity(h,**kw)
    (w1,s1) = _device(lambda: nlk.nonlinear_conductivity(h,**kw))
    assert len(w0)==len(w1)==0
    for c in kw["components"]: assert s0[c].shape==s1[c].shape==(0,)


@pytest.mark.parametrize("order,nvertices,nsuffixes",[(2,9,6),(4,40,30)])
def test_device_chunks_fit_the_budget(order,nvertices,nsuffixes,monkeypatch):
    """The device chunks count the vertex kernel (three arrays of the size
    of hv) and the sweep (hv, dE, df and 3.5 density matrices per suffix
    and row), so both footprints stay below the budget, and a chunk is
    never empty"""
    monkeypatch.setattr(nlk,"_device_budget",lambda: 1e9)
    n,itemsize = 72,16
    unit = n*n*itemsize
    kb,wc = nlk._device_chunks(1600,n,nvertices,nsuffixes,100,itemsize)
    assert kb>=1 and wc>=1
    assert 3*nvertices*kb*unit<=1e9
    assert ((nvertices+2)*kb + 3.5*nsuffixes*kb*wc)*unit<=1e9
    assert nlk._device_chunks(1600,n,nvertices,nsuffixes,0,itemsize)[1]==1
