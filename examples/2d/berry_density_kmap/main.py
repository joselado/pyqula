# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")


# example to compute the dOmega/dE in reciprocal space.
# Integrating this quantity in energy provides the Berry curvature


import numpy as np
from pyqula import geometry
from pyqula import topology
g = geometry.honeycomb_lattice()
h = g.get_hamiltonian(has_spin=False)
h.shift_fermi(0.5)
h.add_haldane(0.1)
topology.dOmega_dE_kmap(h,nk=200)
(k,e) = h.get_bands()

import matplotlib.pyplot as plt

d = np.genfromtxt("BERRY_DENSITY_KMAP.OUT").T
kx,ky,b = d[0],d[1],d[2]

plt.subplot(1,2,1)
plt.scatter(kx,ky,c=b,cmap="inferno")
plt.colorbar(label="dOmega/dE")
plt.xlabel("kx") ; plt.ylabel("ky")

plt.subplot(1,2,2)
plt.scatter(k,e)
plt.xlabel("k-path") ; plt.ylabel("Energy")

plt.show()







