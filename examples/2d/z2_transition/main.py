# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")





# zigzag ribbon
from pyqula import geometry
from pyqula import topology
import numpy as np
g = geometry.honeycomb_lattice() # create geometry of a chain
ms = np.linspace(0.,0.4,40)
z2s,gaps = [],[]


nn = 60

for m in ms: # loop over masses
    # calculate the Z2 invariant for certain Zeeman and Rashba
    h = g.get_hamiltonian(has_spin=True) # get the Hamiltonian, spinfull
    h.add_kane_mele(0.05) # add SOC
    h.add_sublattice_imbalance(m) # add mass
    z2 = topology.z2_invariant(h,nk=nn,nt=nn) # get the Z2
    gap = h.get_gap()
    z2s.append(z2)
    gaps.append(gap)


import matplotlib.pyplot as plt

plt.subplot(2,1,1)
plt.scatter(ms,gaps)
plt.xlabel("Sublattice imbalance") ; plt.ylabel("Gap")
plt.subplot(2,1,2)
plt.scatter(ms,z2s)
plt.xlabel("Sublattice imbalance") ; plt.ylabel("Z2 invariant")
plt.show()
