# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")





# zigzag ribbon
import numpy as np
from pyqula import geometry
g = geometry.honeycomb_lattice()
h = g.get_hamiltonian() # create hamiltonian of the system
(k,e) = h.get_bands() # get the bandstructure (also written to BANDS.OUT)

import matplotlib.pyplot as plt

plt.scatter(k,e)
plt.ylabel("Energy")
plt.xlabel("k-path")
plt.xticks([])
plt.show()






