# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")





# zigzag ribbon
from pyqula import geometry
g = geometry.honeycomb_zigzag_ribbon(10) # create geometry of a zigzag ribbon
h = g.get_hamiltonian(has_spin=False) # create hamiltonian of the system
#exit()
(k,e,c) = h.get_bands(operator="valley")

import matplotlib.pyplot as plt

plt.scatter(k,e,c=c,cmap="rainbow")
plt.colorbar(label="valley")
plt.xlabel("k-path") ; plt.ylabel("Energy")
plt.show()







