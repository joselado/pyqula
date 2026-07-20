# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")




from pyqula import geometry
g = geometry.honeycomb_lattice()
h = g.get_hamiltonian(has_spin=True) # create hamiltonian of the system
h = h + h
(k,e) = h.get_bands()

import matplotlib.pyplot as plt

plt.scatter(k,e)
plt.xlabel("k-path")
plt.ylabel("Energy")
plt.show()







