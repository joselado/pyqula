# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")





from pyqula import geometry
g = geometry.diamond_lattice_minimal()
h = g.get_hamiltonian(has_spin=True)
h1 = h.copy()
h2 = h.copy()
h1.add_antiferromagnetism(0.5)
h2.add_antiferromagnetism(-0.5)
from pyqula import kdos
data = kdos.interface(h1,h2) # columns: k, E, Bulk1, Surf1, Bulk2, Surf2, interface

import matplotlib.pyplot as plt

plt.scatter(data[:,0],data[:,1],c=data[:,-1],cmap="inferno")
plt.colorbar(label="Interface DOS")
plt.xlabel("k-path")
plt.ylabel("Energy")
plt.show()







