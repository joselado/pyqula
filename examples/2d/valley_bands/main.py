# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")





from pyqula import geometry
g = geometry.honeycomb_lattice()
g = g.get_supercell(4)
h = g.get_hamiltonian(has_spin=False)
from pyqula import operators

op = h.get_operator("valley")
(k,e,c) = h.get_bands(operator=op)

import matplotlib.pyplot as plt
plt.scatter(k,e,c=c,cmap="rainbow")
plt.xlabel("Momentum") ; plt.ylabel("Energy")
plt.xticks([])
plt.show()





