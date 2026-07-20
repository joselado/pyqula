# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")



from pyqula import geometry
g = geometry.honeycomb_lattice()
h = g.get_hamiltonian()
h.add_anti_kane_mele(0.05)
(k,e,c) = h.get_bands(operator="sz")

import matplotlib.pyplot as plt

plt.scatter(k,e,c=c,cmap="bwr")
plt.colorbar(label="$S_z$")
plt.xlabel("k-path")
plt.ylabel("Energy")
plt.show()







