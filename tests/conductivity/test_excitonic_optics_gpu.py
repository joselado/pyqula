"""The device path of the Chebyshev excitonic optics must reproduce the
CPU one.

bsetk/chebyshevjax.py writes the convolution matvec of bsetk/convolution.py
and the two resolvent routines of bsetk/chebyshev.py out again in jax, so a
wrong index in the gather, the channel bookkeeping or the halved recursion
of the brakets would still give a plausible spectrum; these tests diff
against the CPU route instead. A spy checks that pyqula.gpu.set_gpu(True)
reaches the device kernels, since a switch that never took effect would
give the right answer on the CPU.

They run through jax's CPU fallback on a machine with no GPU, which checks
correctness and says nothing about speed; the device measurements are in
future_development/kpm_excitonic_optics.md.
"""
import warnings

import numpy as np
import pytest

from pyqula import geometry
from pyqula.bsetk import chebyshevjax
from pyqula.bsetk.interaction import density_interaction
from testutils import gpu_backend


def _model(spinful=False):
    h = geometry.honeycomb_lattice().get_hamiltonian(has_spin=spinful)
    h.add_sublattice_imbalance(1.0)
    if spinful: # an onsite U couples only part of the orbital channels
        return h,density_interaction(h,U=1.5)
    V = density_interaction(h,Vr=lambda r1,r2:
            0.8/np.sqrt((r1-r2).dot(r1-r2)+0.25))
    return h,V


@pytest.mark.parametrize("spinful",[False,True])
@pytest.mark.parametrize("kpm_prec,tol",[("double",1e-8),("single",1e-4)])
def test_device_route_equals_the_cpu_one(kpm_prec,tol,spinful,monkeypatch):
    h,V = _model(spinful)
    kw = dict(energies=[0.,0.9225,1.845,2.5],nk=9 if spinful else 12,
            delta=0.05,V=V,method="kpm")
    (w,l0) = h.get_excitonic_optical_conductivity(**kw)
    (w,s0) = h.get_excitonic_shg_conductivity(components="xxx",**kw)
    calls = []
    for name in ["resolvent_apply","resolvent_braket"]:
        f = getattr(chebyshevjax,name)
        def spy(*args,_f=f,_name=name,**kwargs):
            calls.append(_name)
            return _f(*args,**kwargs)
        monkeypatch.setattr(chebyshevjax,name,spy)
    with warnings.catch_warnings(): # no GPU on this machine
        warnings.simplefilter("ignore")
        with gpu_backend(True):
            (w,l1) = h.get_excitonic_optical_conductivity(kpm_prec=kpm_prec,
                    **kw)
            (w,s1) = h.get_excitonic_shg_conductivity(components="xxx",
                    kpm_prec=kpm_prec,**kw)
    assert calls==["resolvent_braket","resolvent_apply"]
    assert np.allclose(l1,l0,atol=tol*np.max(np.abs(l0)))
    assert np.allclose(s1,s0,atol=tol*np.max(np.abs(s0)))


def test_single_precision_needs_the_device():
    h,V = _model()
    with pytest.raises(ValueError,match="set_gpu"):
        h.get_excitonic_optical_conductivity(energies=[1.],nk=6,V=V,
                method="kpm",kpm_prec="single")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with gpu_backend(True):
            with pytest.raises(ValueError,match="'double', 'single'"):
                h.get_excitonic_optical_conductivity(energies=[1.],nk=6,
                        V=V,method="kpm",kpm_prec="half")
