# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")





from pyqula import geometry
from pyqula import topology
import numpy as np
from pyqula import phasediagram
g = geometry.bichain() # create geometry of a chain
def getz2(x1,x2): 
  # calculate the Z2 invariant for certain Zeeman and Rashba
  h = g.get_hamiltonian(has_spin=True) # get the Hamiltonian, spinfull
  h.add_rashba(0.3) # add SOC
  h.shift_fermi(x2) # add SOC
  h.add_zeeman([0.,0.,0.3]) # add mass
  h.add_swave(x1) # add mass
  z2 = abs(topology.berry_phase(h,nk=40)/np.pi) # get the Z2
  print(x1,x2,z2)
  return z2
# now write the Phase diagram in a file
phasediagram.diagram2d(getz2,x=np.linspace(-.0,0.3,30,endpoint=True),y=np.linspace(0.,4.,30,endpoint=True),nite=1)







