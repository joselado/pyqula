# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")



from pyqula import geometry
g = geometry.kagome_lattice()
h = g.get_hamiltonian()
(k,e) = h.get_bands(kpath=["G","K"])

import matplotlib.pyplot as plt

plt.scatter(k,e)
plt.xlabel("k-path")
plt.ylabel("Energy")
plt.show()







