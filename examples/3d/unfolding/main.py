# Add the root path of the pyqula library
import os ; import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")

from pyqula import geometry
import numpy as np
g0 = geometry.cubic_lattice() # primitive geometry
n = 2
g = g0.get_supercell([n,n,n],store_primal=True) # create a supercell
h = g.get_hamiltonian() # get the Hamiltonian
fons = lambda r: (np.sum((r - g.r[0])**2)<1e-2)*100 # onsite in the impurity
h.add_onsite(fons) # add onsite energy
kpath = np.array(g.get_kpath(nk=100))*n # enlarged k-path
(k,e,d) = h.get_bands(operator="unfold",kpath=kpath) # unfolded bands

import matplotlib.pyplot as plt

plt.scatter(k,e,c=d,cmap="inferno")
plt.colorbar(label="Unfolded weight")
plt.xlabel("k") ; plt.ylabel("Energy")
plt.show()
