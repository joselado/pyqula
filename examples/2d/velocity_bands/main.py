# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")





from pyqula import geometry
from pyqula import topology
from pyqula import dos
g = geometry.honeycomb_lattice()
h = g.get_hamiltonian(has_spin=True)
(k,e,c) = h.get_bands(operator="velocity")

import matplotlib.pyplot as plt

plt.scatter(k,e,c=c,cmap="bwr")
plt.colorbar(label="velocity")
plt.xlabel("k-path")
plt.ylabel("Energy")
plt.show()







