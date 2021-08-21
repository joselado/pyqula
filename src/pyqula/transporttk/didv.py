from ..parallel import pcall
from ..heterostructures import didv
import numpy as np
from ..integration import simpson


# wrapper to compute the dIdV curve

def generic_didv(self,temp=0.,**kwargs):
    """Wrapper to compute the dIdV at finite temperature"""
    if temp==0.: return zero_T_didv(self,**kwargs) # zero temperature
    else: # finite temperature
        return finite_T_didv(self,temp=temp,**kwargs)


def finite_T_didv(self,temp,energy=0.0,**kwargs):
    """Finite temperature dIdV"""
#    return zero_T_didv(self,energy=energy,**kwargs)
    from .fermidirac import fermidirac as FD
    dt = 8 # max T range
    de = temp # energy difference to compute the derivative
    ### Use simpson integration
    def f(e):
        out = zero_T_didv(self,energy=energy+e,**kwargs) 
        out *= FD(e-de,temp=temp) - FD(e+de,temp=temp) 
        return out/de
    from scipy.integrate import quad
    return quad(f,-dt*temp,dt*temp,epsrel=1e-2,limit=20)[0]
#    ne = 79 # number of energies
#    from ..integration import simpson
##    return simpson(f,xlim=[-dt*temp,dt*temp],eps=1e-3)
#    es = np.linspace(-dt*temp,dt*temp,ne,endpoint=True) # energies
#    gs = np.array([zero_T_didv(self,energy=energy+e,**kwargs) for e in es]) # conductance
#    # discard extreme points for numerical stability
#    from ..interpolatetk.outlier import discard_outliers
#    es,gs = discard_outliers(es,gs)
#    # weight for the integral
#    weight = lambda x: FD(x-de,temp=temp) - FD(x+de,temp=temp) 
#    from ..interpolatetk.constrainedinterpolation import positive_interpolator
#    fg = positive_interpolator(es,gs) # interpolated gs
#    # interpolated energies
#    ewin = dt*temp
#    ne2 = 100*ne
#    es_int = np.linspace(-ewin,ewin,ne2,endpoint=True) # energies
#    gs_int = fg(es_int) # interpolated gs
#    weight = FD(es_int-de,temp=temp) - FD(es_int+de,temp=temp) 
#    norm = np.trapz(weight,dx=2*ewin/ne2)/(2*de) # normalization
#    integ = np.trapz(gs_int*weight,dx=2*ewin/ne2)/(2*de) # normalization
#    print(norm)
#    return integ/norm


def zero_T_didv(self,energy=0.0,delta=None,error=1e-4,nk=10,kwant=False,
          opl=None,opr=None,**kwargs):
    if delta is None: delta = self.delta # set the own delta
    if self.dimensionality==1: # one dimensional
        try:
            return didv(self,energy=energy,delta=delta,kwant=kwant,
              opl=opl,opr=opr) # return value
        except: 
            print("Something wrong in didv, returning 0")
            return 1e-10
    elif self.dimensionality==2: # two dimensional
      # function to integrate
      print("Computing",energy)
      f = lambda k: self.generate(k,self.scale_lc,self.scale_rc).didv(energy=energy,delta=delta)
      out = pcall(f,np.linspace(0.,1.,nk,endpoint=False))
#      return simpson(f,eps=np.mean(out))
      return np.trapz(out,dx=1./nk)
    else: raise


