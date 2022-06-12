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





def device_dos(HT,energy=0.0,mode="central",operator=None,**kwargs):
   """ Calculates the density of states 
       of a HTstructure by a  
       green function approach, input is a HTstructure class
   """
   g = fullgreen.get_full_green(HT,energy,mode=mode,**kwargs)
   if operator is not None:
       g = HT.Hr.get_operator(operator)*g
   d = -np.trace(g.imag)
   return d

