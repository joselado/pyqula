# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")





# zigzag ribbon
from pyqula import geometry
from pyqula import topology
import numpy as np
from pyqula import phasediagram
g = geometry.honeycomb_lattice() # create geometry of a chain
def getz2(x1,x2): 
  # calculate the Z2 invariant for certain Zeeman and Rashba
  h = g.get_hamiltonian(has_spin=True) # get the Hamiltonian, spinfull
  h.add_kane_mele(x1) # add SOC
  h.add_sublattice_imbalance(x2) # add mass
  z2 = topology.z2_invariant(h,nk=10,nt=10) # get the Z2
  print(x1,x2,z2)
  return z2
# now write the Phase diagram in a file
phasediagram.diagram2d(getz2,x=np.linspace(-.05,0.05,10,endpoint=True),y=np.linspace(-.1,.1,10,endpoint=True),nite=3)







