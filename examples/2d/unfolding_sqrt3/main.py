# Add the root path of the pyqula library
import os ; import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")

from pyqula import geometry
import numpy as np
g0 = geometry.triangular_lattice() # primitive triangular lattice
g = g0.get_supercell(np.sqrt(3),store_primal=True) # sqrt(3) x sqrt(3) supercell
# the same cell can be asked for explicitly, g0.get_supercell([[2,1,0],[-1,1,0],[0,0,1]])
h = g.get_hamiltonian() # get the Hamiltonian
fons = lambda r: (np.sum((r - g.r[0])**2)<1e-2)*0.6 # onsite in one of the three sites
h.add_onsite(fons) # three-sublattice charge order
kpath = g.get_unfolded_kpath(nk=200) # primitive k-path, in supercell coordinates
(x,y,z) = h.get_kdos_bands(operator="unfold",delta=1e-1,kpath=kpath) # unfolded bands

import matplotlib.pyplot as plt

plt.scatter(x,y,c=z,cmap="inferno")
plt.colorbar(label="Spectral function")
plt.xlabel("k-path") ; plt.ylabel("Energy")
plt.show()
