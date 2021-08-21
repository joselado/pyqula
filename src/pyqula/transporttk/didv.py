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
    from .fermidirac import fermidirac as FD
    ne = 9 # number of energies
    dt = 4 # max T range
    de = temp/4. # energy difference to compute the derivative
    es = np.linspace(energy-dt*temp,energy+dt*temp,ne,endpoint=True) # energies
    gs = np.array([zero_T_didv(self,energy=e,**kwargs) for e in es]) # conductance
    weight = FD(es-de) - FD(es+de) # weight for the integral
    return np.trapz(weight*gs,dx=dt*temp*2/ne) # return result


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


