# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")





from pyqula import geometry  # library to create crystal geometries
from pyqula import hamiltonians  # library to work with hamiltonians
from pyqula import sculpt  # to modify the geometry
from pyqula import correlator
from pyqula import kpm
from pyqula import densitymatrix
import numpy as np
import matplotlib.pyplot as plt
import time
g = geometry.bichain()
g = g.supercell(100)
g.dimensionality = 0
h = g.get_hamiltonian(has_spin=False)
h.add_sublattice_imbalance(.03)
#h.add_rashba(.5)
#h.add_zeeman([0.,1.,0.])
#h.intra += np.diag(np.random.random(h.intra.shape[0]))
pairs = [(0,i) for i in range(20)]
y1 = densitymatrix.restricted_dm(h,mode="KPM",pairs=pairs)
y2 = densitymatrix.restricted_dm(h,mode="full",pairs=pairs)
plt.scatter(range(20),y1,label="KPM",c="red",s=90,marker="o")
plt.scatter(range(20),y2,label="Exact",c="blue",s=20,marker="o")
plt.legend()
plt.show()







