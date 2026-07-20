# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")





from pyqula import geometry
from pyqula import topology
from pyqula import klist
from pyqula import kdos
g = geometry.diamond_lattice_minimal()
h = g.get_hamiltonian(has_spin=True)
h.intra *= 1.3
h.add_kane_mele(0.05)
(k,e,dsurf,dbulk) = kdos.surface(h,operator=None)

import matplotlib.pyplot as plt

plt.scatter(k,e,c=dsurf,cmap="inferno")
plt.colorbar(label="Surface DOS")
plt.xlabel("k-path")
plt.ylabel("Energy")
plt.show()







