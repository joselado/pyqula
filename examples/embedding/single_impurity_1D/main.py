# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")

from pyqula import geometry
from pyqula import embedding
import numpy as np
import matplotlib.pyplot as plt
g = geometry.chain() # create geometry of a chain
#g = g.get_supercell(2)
h = g.get_hamiltonian(has_spin=False) # get the Hamiltonian,spinless
h.add_onsite(1.9)
hv = h.copy() # copy Hamiltonian to create a defective one
W = 0.3 # strength of the disorder
hv.add_onsite(lambda r: (np.sum((r - g.r[0])**2)<1e-2)*W) # add a defect
eb = embedding.Embedding(h,m=hv) # create an embedding object
(x,y,d) = eb.get_ldos(nsuper=200,energy=0.,delta=1e-2,nk=400) # compute LDOS
plt.plot(x,d) ;  plt.scatter(x,d)
plt.show()
