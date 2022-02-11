# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")





# zigzag ribbon
from pyqula import geometry
import numpy as np
from pyqula import embedding
g = geometry.honeycomb_lattice() # create geometry of a chain
#g = geometry.square_lattice() # create geometry of a chain
h = g.get_hamiltonian(has_spin=False) # get the Hamiltonian,spinless
# create a new intraterm, vacancy is modeled as a large onsite potential
h.shift_fermi(2.7)
vintra = h.intra.copy() ; vintra[0,0] = 1000.0

eb = embedding.Embedding(h,m=vintra)
(x,y,d) = eb.ldos(nsuper=5,e=2.)
np.savetxt("LDOS.OUT",np.array([x,y,d]).T)







