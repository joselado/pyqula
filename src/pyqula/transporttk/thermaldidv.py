import numpy as np
from scipy.integrate import quad

thermalmode = "adaptive" # thermal mode

THERMAL_WINDOW = 20 # max |energy-energy0|/temp the thermal quad below integrates
                     # over; exposed as a module constant (rather than a
                     # value local to finite_T_didv) so other callers that
                     # need to know this function's effective energy range
                     # ahead of time (e.g. kappa.py's finite-temperature
                     # path, which builds a self-energy interpolant sized
                     # to cover it) don't have to duplicate the magic number.

def finite_T_didv(self,temp,energy=0.0,**kwargs):
    """Finite temperature dIdV.

    Both thermal-quadrature modes below evaluate zero_T_didv at many
    energies within +-THERMAL_WINDOW*temp of `energy` (147 evaluations
    for one "adaptive"-mode call was measured at temp=0.02) -- all of them
    on the SAME junction, so if it is Floquet-Keldysh-eligible (see
    transporttk.didv._both_leads_superconducting) AND the caller
    explicitly opted into selfenergy_method="aaa" (see dc_current's own
    docstring for why that's not the default -- an unresolved accuracy
    gap), one shared AAA self-energy interpolant (keldyshtk.current.
    build_shared_selfenergy), sized to cover the whole window up front, is
    built once here and reused by every one of those evaluations instead
    of each independently building (and discarding) its own -- the same
    sharing kappa.py's _shared_selfenergy_for_branch already does for its
    own outer coupling/energy sweep, applied here to the window a single
    finite_T_didv call visits internally, with or without any outer sweep
    at all. Skipped if the caller already passed selfenergy_qtci
    explicitly, or didn't ask for selfenergy_method="aaa"."""
    from .didv import zero_T_didv, _both_leads_superconducting
    if ("selfenergy_qtci" not in kwargs and kwargs.get("selfenergy_method") == "aaa"
            and _both_leads_superconducting(self)):
        from ..keldyshtk.current import build_shared_selfenergy
        nmax_max = kwargs.get("nmax_max", 40)
        shared = build_shared_selfenergy(self, abs(energy)+THERMAL_WINDOW*temp,
                nmax_max=nmax_max, delta=kwargs.get("delta"), dv=kwargs.get("dv"))
        if shared is not None:
            kwargs = dict(kwargs, selfenergy_qtci=shared)
    if thermalmode=="adaptive":
        from .fermidirac import fermidirac as FD
        dt = THERMAL_WINDOW # max T range
        de = temp # energy difference to compute the derivative
        ### Use simpson integration
        def f(e):
            out = zero_T_didv(self,energy=energy+e,**kwargs)
            out *= FD(e-de,temp=temp) - FD(e+de,temp=temp)
            return out/de
        from ..integration import peak_integrate
        return quad(f,-dt*temp,dt*temp,epsrel=1e-4,limit=60)[0]/2.
    elif thermalmode=="pm": # poor's man mode
        from .fermidirac import dFDdT
        nT = 45
        Ts = np.linspace(-4*temp,4*temp,nT)
        def f(e):
            return zero_T_didv(self,energy=energy+e,**kwargs)
        out = [f(e)*dFDdT(e,temp=temp)*temp for e in Ts]
        norm = np.sum(dFDdT(Ts,temp=temp)*temp)
        out = np.trapezoid(out)/norm
        return out
    else: raise


