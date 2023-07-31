# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")





# zigzag ribbon
import numpy as np
from pyqula import geometry
from pyqula import meanfield

g = geometry.honeycomb_lattice()
g.write()
Us = np.linspace(0.,4.,10) # different Us
f = open("EVOLUTION.OUT","w") # file with the results
for U in Us: # loop over Us
  
  h = g.get_hamiltonian() # create hamiltonian of the system
  h.add_swave(0.)
  mf = meanfield.guess(h,mode="antiferro") # antiferro initialization
  # perform SCF with specialized routine for Hubbard
  h = h.get_mean_field_hamiltonian(nk=13,filling=0.5,U=U,V=0.1,verbose=1,
                mix=0.9,mf=mf)
  gap = h.get_gap() # compute the gap
  f.write(str(U)+"   "+str(gap)+"\n") # save in a file
  
f.close()







