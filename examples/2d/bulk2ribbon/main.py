# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")


from pyqula import geometry
from pyqula import ribbon
import numpy as np

g = geometry.honeycomb_lattice() # create the geometry
h = g.get_hamiltonian()
hr = ribbon.bulk2ribbon(h,n=20) # create a ribbon from this 2D Hamiltonian
(k,e) = hr.get_bands() # compute band structure

import matplotlib.pyplot as plt

plt.scatter(k,e)
plt.xlabel("k-path") ; plt.ylabel("Energy")
plt.show()







