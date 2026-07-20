# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")





from pyqula import geometry
from pyqula import hamiltonians
import numpy as np
from pyqula import specialgeometry
g = specialgeometry.twisted_bilayer(20)
#g = geometry.honeycomb_lattice()
g.write()
#g = geometry.read()
from pyqula.specialhopping import twisted,twisted_matrix
h = g.get_hamiltonian(is_sparse=True,has_spin=False,is_multicell=False,
     mgenerator=twisted_matrix(ti=0.4,lambi=7.0))
from pyqula import kdos
(x,y,z) = kdos.kdos_bands(h,ntries=1)
#h.get_bands()

import matplotlib.pyplot as plt

plt.scatter(x,y,c=z,cmap="inferno")
plt.xlabel("k-path") ; plt.ylabel("Energy")
plt.show()







