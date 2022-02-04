# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")

from pyqula import geometry
g = geometry.chain() # linear chain
ns = 1 # number of supercells
g = g.get_supercell(ns)
h = g.get_hamiltonian() # get the Hamiltonian
from pyqula import rkky
m = rkky.rkky_map(h,n=10//ns,nk=100,mode="LR")
import numpy as np
np.savetxt("RKKY.OUT",m)

import matplotlib.pyplot as plt
plt.plot(m[:,0],m[:,3],marker="o")

plt.show()



