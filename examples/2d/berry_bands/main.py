# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")

from pyqula import geometry
from pyqula import topology
from pyqula import dos
g = geometry.honeycomb_lattice()
h = g.get_hamiltonian(has_spin=False)
#h.add_sublattice_imbalance(0.5)
h.add_haldane(0.1)
op = h.get_operator("berry")
(k,e,b) = h.get_bands(operator=op)

#(kb,bb) = topology.write_berry(h,mode="Green",operator=None)
(kb,bb) = topology.write_berry(h,mode="Wilson",operator=None)


import matplotlib.pyplot as plt
import numpy as np
bmax = np.max(np.abs(b))/2
plt.subplot(2,1,1)
plt.scatter(k,e,c=b,vmin=-bmax,vmax=bmax,cmap="rainbow")
plt.colorbar()
plt.subplot(2,1,2)
plt.scatter(kb,bb)



plt.show()
# this seems broken




