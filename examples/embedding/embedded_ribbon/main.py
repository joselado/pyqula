# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")



from pyqula import geometry
import numpy as np
from pyqula import embedding
from pyqula import islands
g = geometry.honeycomb_zigzag_ribbon(10) # ribbon
h = g.get_hamiltonian() # get the Hamiltonian
# perform a calculation for the isotaled system

from pyqula import potentials
edge = potentials.edge_potential(g) # edge potential

#edge = lambda r: r[1]>0.*edge0(r)

# create a selfenergy in a single edge (the left one) to kill magnetism 
hs = h.copy()*0.
hs.add_onsite(edge)
selfe = -0.2*1j*hs.intra # selfenergy on edge, beware of minus sign


# create an embedding object with that selfenergy
eb = embedding.embed_hamiltonian(h,selfenergy=selfe) # create embedding object
# eb.H is the selfconsistent Hamiltonian object
eb.get_kdos(delta=1e-2)



