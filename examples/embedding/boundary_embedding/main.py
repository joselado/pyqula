# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")


from pyqula import geometry
from pyqula import embedding
import numpy as np
g = geometry.triangular_lattice() # create geometry of a chain
g = g.get_supercell(20)
from pyqula.supercell import turn_orthorhombic
g = turn_orthorhombic(g)
h = g.get_hamiltonian(has_spin=False) # get the Hamiltonian,spinless
h.add_onsite(-5.5)
h.add_onsite(lambda r: (np.random.random()-0.5)*1.)
eb = embedding.Embedding(h,selfenergy=2.) # create the embedding object
(x,y,d) = eb.get_ldos(energy=0.,delta=1e-3) 
print(d.shape)

import matplotlib.pyplot as plt
plt.scatter(x,y,c=d,cmap="inferno",s=100) ; plt.axis("equal")
plt.show()
