import numpy as np
from . import fullgreen

def get_dos(self,energies=None,write=True,**kwargs):
    """Compute density of states"""
    if self.dimensionality==1:
        if energies is None: energies = np.linspace(-1.0,1.0,100)
        ds = [self.get_coupled_central_dos(energy=e,**kwargs) for e in energies]
        if write: np.savetxt("DOS.OUT",np.array([energies,ds]).T)
        return energies,np.array(ds)
    else: raise





def device_dos(HT,energy=0.0,mode="central",
        operator=None,ic=0,**kwargs):
   """ Calculates the density of states 
       of a HTstructure by a  
       green function approach, input is a HTstructure class
   """
   g = fullgreen.get_full_green(HT,energy,mode=mode,ic=ic,**kwargs)
   if ic is not None: # specific cell
     if operator is not None:
       g = HT.Hr.get_operator(operator)*g
   elif ic is None: # all the cells
       g0 = 0. # initialize
       for i in range(len(g)): # loop
         gi = g[i]
         if operator is not None:
           gi = HT.Hr.get_operator(operator)*gi
         g0 = g0 + gi
       g = g0 # overwrite
   d = -np.trace(g.imag)
   return d

