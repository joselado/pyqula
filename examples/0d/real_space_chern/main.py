# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")





from pyqula import islands
import numpy as np
g = islands.get_geometry(name="honeycomb",n=14,nedges=4,rot=0.0,clean=False) 
h = g.get_hamiltonian(has_spin=False)
h.add_haldane(.1)
from pyqula import topology
#op = h.get_operator("valley",delta=1e-2)
(r,c) = topology.real_space_chern(h)

import matplotlib.pyplot as plt

plt.scatter(r[:,0],r[:,1],c=c,cmap="bwr",vmin=-1.0,vmax=1.0)
plt.colorbar()
plt.axis("equal")
plt.show()





