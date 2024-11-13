# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")


import matplotlib.pyplot as plt


from pyqula import geometry
from pyqula import heterostructures
import numpy as np

g = geometry.square_ribbon(10) # create a square lattice
h = g.get_hamiltonian() # create the Hamiltonian
h.add_onsite(-3.0) # shift chemical potential

Delta_SC = 1e-1 # superconducting gap
h1 = h.copy() # first lead
h2 = h.copy() # second lead

h2.add_swave(Delta_SC)

L = 40 # length
central = [h1.copy() for i in range(L)]
central += [h2.copy() for i in range(L)]
HT = heterostructures.build(h1,h2,central=central) # create the junction
#HT.get_dos(ic=10,delta=1e-1)
from pyqula import parallel
#parallel.cores = 4
#HT.get_kdos(delta=3e-1,ic=10)
(x,y,d) = HT.get_ldos(energy=0.,delta=Delta_SC/100)


import matplotlib.pyplot as plt


plt.scatter(x,y,c="black")
plt.scatter(x,y,c=d/np.max(d),cmap="rainbow")
plt.colorbar(label="LDOS")
plt.xlabel("x position")
plt.ylabel("y position")
plt.axis("equal")

plt.show()
