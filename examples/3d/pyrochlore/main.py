# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")





import numpy as np
from pyqula import geometry
g = geometry.pyrochlore_lattice()
h = g.get_hamiltonian()
h.turn_dense()
g.center() # center the geometry
g.write()
ms = [-ri for ri in g.r] # magnetizations
(k,e) = h.get_bands()

import matplotlib.pyplot as plt

plt.scatter(k,e)
plt.xlabel("k-path")
plt.ylabel("Energy")
plt.show()








