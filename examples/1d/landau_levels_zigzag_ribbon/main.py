# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")





# zigzag ribbon
from pyqula import geometry
g = geometry.honeycomb_zigzag_ribbon(20) # create geometry of a zigzag ribbon
h = g.get_hamiltonian() # create hamiltonian of the system
h.remove_spin() # remove spin degree of freedom
h.add_peierls(.03) # add the magnetic field
(k,e) = h.get_bands() # calculate band structure

import matplotlib.pyplot as plt

plt.scatter(k,e)
plt.xlabel("k-path") ; plt.ylabel("Energy")
plt.show()







