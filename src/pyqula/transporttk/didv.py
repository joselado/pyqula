from ..parallel import pcall
from ..heterostructures import didv
import numpy as np
from ..integration import simpson


# wrapper to compute the dIdV curve

def generic_didv(self,energy=0.0,delta=None,error=1e-4,nk=10,kwant=False,
          opl=None,opr=None):
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


