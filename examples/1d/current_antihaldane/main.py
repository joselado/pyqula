# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")





# zigzag ribbon
from pyqula import geometry
g = geometry.honeycomb_zigzag_ribbon(10) # create geometry of a zigzag ribbon
h = g.get_hamiltonian(has_spin=False) # create hamiltonian of the system
h.add_antihaldane(0.1)
from pyqula import ldos
(es,ds) = ldos.spatial_energy_profile(h,operator=h.get_operator("current"),nk=100)
pos = h.geometry.r[:,1] # transverse position of each site

import matplotlib.pyplot as plt

plt.contourf(pos,es,ds.T,levels=100,cmap="inferno")
plt.colorbar(label="Current LDOS")
plt.xlabel("y") ; plt.ylabel("Energy")
plt.show()








