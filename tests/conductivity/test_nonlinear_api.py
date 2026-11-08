"""The public entry points of the nonlinear response forward every keyword.

conductivity.nonlinear_conductivity, nonlinear_susceptibility,
nonlinear_dc_conductivity, berry_curvature_dipole, shift_current and
injection_current (src/pyqula/conductivity.py), and the h.get_* delegators
of hamiltonians.py that call them, are thin wrappers of the routines in
conductivitytk/nonlinearkernel.py, nonlineardc.py and photocurrent.py. Every
physics test calls those routines directly, so a wrapper that drops a
keyword (falls back to the default T, or turns a rectified call into a
harmonic by losing frequencies=) would go unnoticed.

Each test calls the public routine with every keyword away from its default
and asserts that the result equals the conductivitytk routine with the same
keywords. It then asserts, keyword by keyword, that the conductivitytk
routine changes when that keyword is left at its default, so that a
forgotten keyword makes the equality fail rather than pass by accident.
The last test checks the arguments that the public routines refuse before
forwarding.
"""
import inspect

import numpy as np
import pytest

from pyqula import geometry
from pyqula import strain
from pyqula import conductivity
from pyqula.conductivitytk import nonlinearkernel
from pyqula.conductivitytk import nonlineardc
from pyqula.conductivitytk import photocurrent


def _hamiltonian(has_spin=False):
    """Doped honeycomb lattice with a mass, uniaxial strain and a Haldane
    term, so that inversion, time reversal, C3 and the mirrors are broken
    and every component used below is nonzero. In the spinful case an
    exchange field along z splits the two spin channels."""
    h = geometry.honeycomb_lattice().get_hamiltonian(has_spin=has_spin)
    h.add_sublattice_imbalance(0.4)
    strain.uniaxial_strain(h,d=np.array([1.,0.4,0.]),s=0.1)
    h.add_haldane(0.07)
    if has_spin: h.add_exchange([0.,0.,0.3])
    h.shift_fermi(-0.8)
    return h


def _flatten(out):
    """Every number of a result, whatever its layout (a value, an array,
    a (energies, array) pair or nested dicts), in a fixed order"""
    if isinstance(out,dict):
        return np.concatenate([_flatten(out[key]) for key in sorted(out)])
    if isinstance(out,tuple):
        return np.concatenate([_flatten(o) for o in out])
    return np.atleast_1d(np.asarray(out,dtype=np.complex128)).ravel()


def _check_forwarding(public,reference,h,kwargs,skip=()):
    """public(h,**kwargs) equals reference(h,**kwargs), and reference
    changes when any one keyword away from its default (except those in
    skip) is dropped from kwargs"""
    defaults = {k:p.default for (k,p) in
            inspect.signature(reference).parameters.items()}
    keys = [k for k in kwargs if k not in skip
            and (k not in defaults or kwargs[k]!=defaults[k])]
    assert len(keys)>0
    ref = _flatten(reference(h,**kwargs))
    scale = np.max(np.abs(ref))
    assert scale>0.
    for fun in public:
        out = _flatten(fun(h,**kwargs))
        assert out.shape==ref.shape
        assert np.max(np.abs(out-ref))<=1e-12*scale
    for key in keys:
        dropped = {k:v for (k,v) in kwargs.items() if k!=key}
        try: alt = _flatten(reference(h,**dropped))
        except (ValueError,TypeError): continue # a dropped key fails loudly
        assert alt.shape!=ref.shape or np.max(np.abs(alt-ref))>1e-6*scale, \
            "the keyword "+key+" does not change the result, so its "+\
            "forwarding is not tested"


def test_nonlinear_conductivity_forwards_every_keyword():
    h = _hamiltonian()
    kwargs = dict(order=3,energies=[0.4,1.1],frequencies=(1,-1,2),nk=9,
            T=0.09,delta=0.13,components=["xyyx","yxxx"],degeneracy_tol=0.5)
    public = [conductivity.nonlinear_conductivity,
            lambda h,**kw: h.get_nonlinear_conductivity(**kw)]
    # the recursion never divides by an energy difference, so it does not
    # use degeneracy_tol and its forwarding cannot be seen in the result
    _check_forwarding(public,nonlinearkernel.nonlinear_conductivity,h,
            kwargs,skip=["degeneracy_tol"])


def test_nonlinear_susceptibility_forwards_every_keyword():
    h = _hamiltonian()
    kwargs = dict(order=3,energies=[0.4,1.1],frequencies=(1,-1,2),nk=9,
            T=0.09,delta=0.13,components=["xyyx","yxxx"],
            length_unit=1.42,energy_unit=2.7)
    public = [conductivity.nonlinear_susceptibility,
            lambda h,**kw: h.get_nonlinear_susceptibility(**kw)]
    _check_forwarding(public,nonlinearkernel.nonlinear_susceptibility,h,
            kwargs)


@pytest.mark.parametrize("kwargs",[
    dict(order=2,term="all",tau=1.3,qmd="kaplan",nk=21,T=0.11,
        components=["xyy","yxx","xxx"],degeneracy_tol=0.5),
    dict(order=3,term="drude",tau=0.7,nk=21,T=0.11,components=["xxyy"]),
    ])
def test_nonlinear_dc_conductivity_forwards_every_keyword(kwargs):
    h = _hamiltonian()
    public = [conductivity.nonlinear_dc_conductivity,
            lambda h,**kw: h.get_nonlinear_dc_conductivity(**kw)]
    _check_forwarding(public,nonlineardc.nonlinear_dc_conductivity,h,
            kwargs)


def test_nonlinear_dc_conductivity_forwards_the_channel():
    h = _hamiltonian(has_spin=True)
    kwargs = dict(term="drude",tau=1.3,channel="up",nk=21,T=0.11,
            components=["xyy","yxx"])
    public = [conductivity.nonlinear_dc_conductivity,
            lambda h,**kw: h.get_nonlinear_dc_conductivity(**kw)]
    _check_forwarding(public,nonlineardc.nonlinear_dc_conductivity,h,
            kwargs)


def test_berry_curvature_dipole_forwards_every_keyword():
    h = _hamiltonian(has_spin=True)
    kwargs = dict(nk=21,T=0.13,channel="up",degeneracy_tol=0.5)
    public = [conductivity.berry_curvature_dipole,
            lambda h,**kw: h.get_berry_curvature_dipole(**kw)]
    _check_forwarding(public,nonlineardc.berry_curvature_dipole,h,
            kwargs)


@pytest.mark.parametrize("name",["shift_current","injection_current"])
def test_photocurrents_forward_every_keyword(name):
    h = _hamiltonian()
    # a degeneracy_tol this large merges bands near the gap edge into one
    # multiplet, which is not physical but makes the keyword visible
    kwargs = dict(energies=[1.2,1.9],nk=9,T=0.03,delta=0.09,
            components=["yyy","xxy"],degeneracy_tol=1.5)
    public = [getattr(conductivity,name),
            lambda h,**kw: getattr(h,"get_"+name)(**kw)]
    _check_forwarding(public,getattr(photocurrent,name),h,kwargs)


@pytest.mark.parametrize("name",["nonlinear_conductivity","shift_current"])
def test_single_precision_request_reaches_the_kernel(name):
    """prec= is forwarded: on the CPU backend prec="single" is refused by
    the kernel, so a wrapper that dropped it would return a result"""
    from pyqula import gpu
    if gpu.get_gpu(): pytest.skip("the device accepts prec='single'")
    h = _hamiltonian()
    with pytest.raises(NotImplementedError):
        getattr(photocurrent if name=="shift_current" else nonlinearkernel,
                name)(h,energies=[1.2],nk=4,prec="single")
    for fun in [getattr(conductivity,name),
            lambda h,**kw: getattr(h,"get_"+name)(**kw)]:
        with pytest.raises(NotImplementedError):
            fun(h,energies=[1.2],nk=4,prec="single")


@pytest.mark.parametrize("name,module",[
    ("nonlinear_conductivity",nonlinearkernel),
    ("nonlinear_susceptibility",nonlinearkernel),
    ("nonlinear_dc_conductivity",nonlineardc),
    ("berry_curvature_dipole",nonlineardc),
    ("shift_current",photocurrent),
    ("injection_current",photocurrent),
    ])
def test_every_keyword_reaches_the_conductivitytk_routine(name,module,
        monkeypatch):
    """A spy in place of the conductivitytk routine receives every keyword
    of the public signature with the value passed, also the ones the
    result does not depend on (degeneracy_tol of the recursion). order is
    checked by the public routine before it forwards, so it gets a valid
    value away from its default instead of a sentinel"""
    received = {}
    def spy(h,**kwargs):
        received.update(kwargs)
        return "spy"
    monkeypatch.setattr(module,name,spy)
    signature = inspect.signature(getattr(conductivity,name)).parameters
    kwargs = {k:("sentinel",k) for (k,p) in signature.items()
            if k!="h" and p.kind is p.POSITIONAL_OR_KEYWORD}
    if "order" in kwargs: kwargs["order"] = 3
    h = _hamiltonian()
    assert getattr(conductivity,name)(h,**kwargs)=="spy"
    assert received==kwargs
    received.clear()
    assert getattr(h,"get_"+name)(**kwargs)=="spy"
    assert received==kwargs


def test_public_arguments_are_checked():
    """Inputs that the signatures accept but the formulas cannot take are
    refused by the public routines with a message naming what they need:
    an order that is not a positive integer (a float is not rounded), an
    empty list of components, a scalar in place of the array of
    frequencies, and T=None in the dc routines, which are Fermi-surface
    sums with no broadening to default to. A numpy integer is an order"""
    h = _hamiltonian()
    c = conductivity
    kw = dict(energies=[1.],nk=4)
    for order in [2.,True,"2"]:
        for f in [c.nonlinear_conductivity,c.nonlinear_susceptibility,
                c.nonlinear_dc_conductivity]:
            with pytest.raises(TypeError,match="positive integer"):
                f(h,order=order,nk=4)
    for order in [0,-1]:
        with pytest.raises(ValueError,match="positive integer"):
            c.nonlinear_conductivity(h,order=order,**kw)
    (w,s) = c.nonlinear_conductivity(h,order=np.int64(2),components="xxx",
            **kw)
    assert np.all(np.isfinite(s))
    for f in [c.nonlinear_conductivity,c.nonlinear_susceptibility,
            c.shift_current,c.injection_current]:
        with pytest.raises(ValueError,match="at least one component"):
            f(h,components=[],**kw)
        with pytest.raises(ValueError,match="1D array"):
            f(h,energies=1.,nk=4,components="xxx")
    with pytest.raises(ValueError,match="at least one component"):
        c.nonlinear_dc_conductivity(h,components=[],nk=4)
    with pytest.raises(ValueError,match="at least one component"):
        h.get_nonlinear_conductivity(components=[],**kw)
    for f in [c.nonlinear_dc_conductivity,c.berry_curvature_dipole,
            lambda h,**kw: h.get_nonlinear_dc_conductivity(**kw),
            lambda h,**kw: h.get_berry_curvature_dipole(**kw)]:
        with pytest.raises(ValueError,match="T>0, got T=None"):
            f(h,T=None,nk=4)
