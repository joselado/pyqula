# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")

from pyqula import geometry
g = geometry.honeycomb_lattice()
h = g.get_hamiltonian()
(k,e) = h.get_bands_map()

import matplotlib.pyplot as plt

plt.scatter(k[:,0],k[:,1],c=e)
plt.colorbar(label="Energy")
plt.xlabel("kx") ; plt.ylabel("ky")
plt.show()







