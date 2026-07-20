# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")





from pyqula import geometry
from pyqula import specialhamiltonian

g = geometry.honeycomb_lattice()
h = specialhamiltonian.flux2d(g,n=80)
h.geometry.write()
(k,e) = h.get_bands()

import matplotlib.pyplot as plt

plt.scatter(k,e)
plt.xlabel("k-path") ; plt.ylabel("Energy")
plt.show()







