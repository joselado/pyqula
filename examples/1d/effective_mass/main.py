# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")



import numpy as np
from pyqula import geometry
g = geometry.chain()
h = g.get_hamiltonian(has_spin=False) # create hamiltonian of the system
#h.add_rashba(1.)
#h.add_exchange([0.,0.,1.])
from pyqula import mass
(k,e,c) = h.get_bands(operator="mass")

import matplotlib.pyplot as plt

plt.scatter(k,e,c=c,cmap="rainbow")
plt.colorbar(label="Effective mass")
plt.xlabel("k-path") ; plt.ylabel("Energy")
plt.show()



