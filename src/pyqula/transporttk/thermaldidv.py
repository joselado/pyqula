import numpy as np
from scipy.integrate import quad

thermalmode = "adaptive" # thermal mode

def finite_T_didv(self,temp,energy=0.0,**kwargs):
    """Finite temperature dIdV"""
    from .didv import zero_T_didv
    if thermalmode=="adaptive":
        from .fermidirac import fermidirac as FD
        dt = 20 # max T range
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
        out = np.trapz(out)/norm
        return out
    else: raise


