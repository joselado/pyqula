# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")





from pyqula import islands
from pyqula import spectrum
from pyqula import operators

import numpy as np
g = islands.get_geometry(name="honeycomb",n=11,nedges=6,rot=0.0) # get an island
h = g.get_hamiltonian(has_spin = False)
h.add_haldane(0.05) # topological gap

ibulk = h.geometry.closest_index([0.,0.,0.]) # get the central site
iedge = h.geometry.closest_index([-20.,0.,0.]) # get the edge site

opbulk = h.get_operator("site",index=ibulk) # bulk operator
opedge = h.get_operator("site",index=iedge) # edge operator

(e_bulk,d_bulk) = h.get_dos(operator=opbulk,delta=0.02)
(e_edge,d_edge) = h.get_dos(operator=opedge,delta=0.02)

import matplotlib.pyplot as plt

plt.plot(e_bulk,d_bulk,label="bulk")
plt.plot(e_edge,d_edge,label="edge")
plt.legend() ; plt.xlabel("Energy") ; plt.ylabel("DOS")
plt.show()







