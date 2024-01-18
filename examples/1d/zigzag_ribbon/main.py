# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")





# zigzag ribbon
from pyqula import geometry
g = geometry.honeycomb_zigzag_ribbon(20) # create geometry of a zigzag ribbon
h = g.get_hamiltonian(has_spin=False) # create hamiltonian of the system
#exit()
(ks,es) = h.get_bands()
(esd,ds) = h.get_dos()

import matplotlib.pyplot as plt
plt.subplot(1,2,1)
plt.scatter(ks,es)
plt.subplot(1,2,2)
plt.plot(esd,ds)
plt.show()


