# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")





from pyqula import islands
from pyqula import interactions
from pyqula import scftypes
import numpy as np
g = islands.get_geometry(name="honeycomb",n=4,nedges=3,rot=0.0) # get an island
# maximum distance to the origin
rmax = np.sqrt(np.max([ri.dot(ri) for ri in g.r]))
def fhop(r1,r2):
  """Function to calculate the hopping, it will create different hoppings
  for different atoms. The hopping becomes smaller the further the atom
  is from the origin"""
  tmax = 1.0 # minimum hopping
  tmin = 0.7 # maximum hopping
  dr = r1-r2 # vector between the two sites
  drmod = np.sqrt(dr.dot(dr)) # distance
  rm = (r1+r2)/2. # average position
  rmmod = np.sqrt(rm.dot(rm)) # value of the average position
  if 0.9<drmod<1.1: # if first neighbor
    lamb = rmmod/rmax # ratio between 0 and 1
    return tmax*(1.-lamb) + tmin*lamb
  else: return 0.0
fhop = None
h = g.get_hamiltonian(fun=fhop,has_spin=True) # get the Hamiltonian
h.add_zeeman([0.,.4,0.])
g.write()
mf = scftypes.guess(h,mode="antiferro")
scf = scftypes.selfconsistency(h,filling=0.5,g=1.0,
                mix=0.9,mf=mf,mode="U")
#scf.hamiltonian.get_bands()







