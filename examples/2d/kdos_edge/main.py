# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")




from pyqula import geometry
from pyqula import kdos
g = geometry.honeycomb_lattice()
h = g.get_hamiltonian()
h.add_haldane(0.05)
(k,e,ds,db) = kdos.surface(h)

import matplotlib.pyplot as plt

plt.scatter(k,e,c=ds,cmap="inferno")
plt.colorbar(label="Surface DOS")
plt.xlabel("k-path")
plt.ylabel("Energy")
plt.show()








