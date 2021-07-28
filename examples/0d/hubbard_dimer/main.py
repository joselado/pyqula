# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")





import numpy as np
from pyqula import geometry
from pyqula import meanfield

# this example performs a selconsistent calculation in a dimer

g = geometry.dimer() # generate the geometry
Us = np.linspace(0.,4.,10) # different Hubbard U
mz1,mz2 = [],[] # lists to store magnetizations
for U in Us: # loop over Hubbard U
    h = g.get_hamiltonian() # create hamiltonian of the system
    mf = meanfield.guess(h,mode="antiferro") # antiferro initialization
    scf = meanfield.hubbardscf(h,filling=0.5,U=U,mf=mf) # perform SCF
    mz = scf.hamiltonian.get_magnetization()[:,2] # extract the magnetization
    mz1.append(mz[0]) # magnetization of the first site
    mz2.append(mz[1]) # magnetization of the second site

# now plot all the results

import matplotlib.pyplot as plt

plt.plot(Us,mz1,marker="o",label="Site #1",c="red") 
plt.plot(Us,mz2,marker="o",label="Site #2",c="blue") 
plt.legend()
plt.xlabel("Hubbard U")
plt.ylabel("Magnetic moment")
plt.show()










