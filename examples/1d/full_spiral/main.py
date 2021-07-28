# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")





# zigzag ribbon
from pyqula import geometry
from pyqula import scftypes
import numpy as np
# this calculates
g = geometry.chain() # chain geometry
h0 = g.get_hamiltonian() # create hamiltonian of the system


fo = open("STIFFNESS.OUT","w") # open file
for a in np.linspace(0.,1.0,100): # loop over angles, in units of 2pi
  h = h0.copy()
  h.generate_spin_spiral(vector=[0.,0.,1.],qspiral=[a,0.,0.])
  h.add_zeeman([1.0,0.0,0.0])
  e = h.get_total_energy(nk=10)
  fo.write(str(a)+"    "+str(e)+"\n") # write
  print(a,e)
fo.close()







