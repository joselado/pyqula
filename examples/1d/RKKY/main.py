# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")

from pyqula import geometry
g = geometry.single_square_lattice() # linear chain
g = geometry.triangular_lattice() # linear chain
g = geometry.chain()
ns = 3
g = g.get_supercell(ns)
h = g.get_hamiltonian() # get the Hamiltonian
#h.shift_fermi(1.8)
r0  = g.r[0]
#ds = [h.get_rkky(ri=r0,rj=r0+i*a0) for i in range(ns)]
from pyqula import rkky
m = rkky.rkky_map(h,n=10//ns,nk=200,mode="LR")
ds = m[:,3]
import numpy as np
np.savetxt("RKKY.OUT",m)

import matplotlib.pyplot as plt
#plt.scatter(m[:,0],m[:,1],c=m[:,3],marker="o")
plt.scatter(m[:,0],m[:,3])

plt.show()



