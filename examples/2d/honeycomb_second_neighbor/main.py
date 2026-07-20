# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")





# zigzag ribbon
import numpy as np
from pyqula import geometry
g = geometry.honeycomb_lattice()
h = g.get_hamiltonian(tij=[1.0,0.2]) # create hamiltonian of the system
(k,e) = h.get_bands()

import matplotlib.pyplot as plt

plt.scatter(k,e)
plt.xlabel("k-path")
plt.ylabel("Energy")
plt.show()






