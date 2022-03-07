# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")





# zigzag ribbon
from pyqula import geometry
from pyqula import scftypes
import numpy as np
# this spin calculates the spin stiffness of an interacting 1d chain
g = geometry.chain() # chain geometry
h0 = g.get_hamiltonian() # create hamiltonian of the system
fo = open("STIFFNESS.OUT","w") # open file
for a in np.linspace(0.,.2,20): # loop over angles, in units of pi
  h = h0.copy()
  h.generate_spin_spiral(qspiral=[a,0.,0.],vector=[0.,1.,0.])
  scf = scftypes.hubbardscf(h,filling=0.1,nkp=100,g=2.5,silent=True,
           mag=[[0.,0.,.1]],mix=0.9,maxerror=1e-4)
  fo.write(str(a)+"    "+str(scf.total_energy)+"\n") # write
  print(a,scf.total_energy)
fo.close()







