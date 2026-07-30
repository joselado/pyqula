# Add the root path of the pyqula library
import os ; import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")

from pyqula import geometry
import numpy as np
g0 = geometry.honeycomb_lattice()
M = [[2,1,0],[0,1,0],[0,0,1]] # non-diagonal supercell matrix, det(M)=2
g = g0.get_supercell(M,store_primal=True) # non-orthogonal supercell
h = g.get_hamiltonian() # get the Hamiltonian
fons = lambda r: (np.sum((r - g.r[0])**2)<1e-2)*100 # onsite in the impurity
h.add_onsite(fons) # add onsite energy
kpath = np.array(g.get_kpath(nk=200)) # k-path in the supercell's own BZ
(x,y,z) = h.get_kdos_bands(operator="unfold",delta=1e-1,kpath=kpath) # unfolded bands

import matplotlib.pyplot as plt

plt.scatter(x,y,c=z,cmap="inferno")
plt.show()
