# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")





# zigzag ribbon
from pyqula import geometry
from pyqula import operators
g = geometry.honeycomb_zigzag_ribbon(10) # create geometry of a zigzag ribbon
h = g.get_hamiltonian() # create hamiltonian of the system
h.add_peierls(0.05)
from pyqula import ldos
ldos.multi_ldos(h,op=h.get_operator("valley"))

import numpy as np
import matplotlib.pyplot as plt

d = np.genfromtxt("DOSMAP.OUT").T
plt.scatter(d[0],d[1],c=d[2],cmap="inferno")
plt.colorbar(label="valley LDOS")
plt.xlabel("Site index") ; plt.ylabel("Energy")
plt.show()







