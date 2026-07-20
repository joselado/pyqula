# Add the root path of the pyqula library
import os ; import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")

from pyqula import geometry
g = geometry.honeycomb_lattice()
h = g.get_hamiltonian()
h.add_rashba(0.2) # Rashba spin-orbit coupling
h.add_zeeman([0.,0.,0.6]) # Zeeman field
from pyqula import topology
(kx,ky,omega) = h.get_berry_curvature() # compute Berry curvature
c = h.get_chern() # compute the Chern number

import matplotlib.pyplot as plt

plt.scatter(kx,ky,c=omega,cmap="bwr")
plt.colorbar(label="Berry curvature")
plt.xlabel("kx") ; plt.ylabel("ky")
plt.title("Chern number = "+str(c))
plt.show()

