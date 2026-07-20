# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")





# zigzag ribbon
from pyqula import geometry
from pyqula import ribbon
g = geometry.honeycomb_lattice()
g = ribbon.bulk2ribbon(g,n=5,boundary=[6,1])
h = g.get_hamiltonian(has_spin=False)
(k,e) = h.get_bands()
g = g.supercell(4)
g.write()

import matplotlib.pyplot as plt

plt.scatter(k,e)
plt.xlabel("k-path") ; plt.ylabel("Energy")
plt.show()







