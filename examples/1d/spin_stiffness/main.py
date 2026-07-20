# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")





# zigzag ribbon
from pyqula import geometry
from pyqula import meanfield
import numpy as np
# this spin calculates the spin stiffness of an interacting 1d chain
g = geometry.chain() # chain geometry
h0 = g.get_hamiltonian() # create hamiltonian of the system
fo = open("STIFFNESS.OUT","w") # open file
angles,energies = [],[] # store the swept results
for a in np.linspace(0.,.2,20): # loop over angles, in units of pi
  h = h0.copy()
  h.generate_spin_spiral(qspiral=[a,0.,0.],vector=[0.,1.,0.])
  mf = meanfield.guess(h,mode="ferroZ",fun=0.1) # initial guess
  scf = meanfield.hubbardscf(h,filling=0.1,nk=100,U=2.5,verbose=0,
           mf=mf,mix=0.9,maxerror=1e-4)
  fo.write(str(a)+"    "+str(scf.total_energy)+"\n") # write
  print(a,scf.total_energy)
  angles.append(a)
  energies.append(scf.total_energy)
fo.close()

import matplotlib.pyplot as plt

plt.plot(angles,energies,marker="o")
plt.xlabel("Spiral angle [$\\pi$]") ; plt.ylabel("Total energy")
plt.show()







