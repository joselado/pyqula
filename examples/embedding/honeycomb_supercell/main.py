# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")



from pyqula import geometry
import numpy as np
from pyqula import embedding
g = geometry.honeycomb_lattice() # create geometry of a chain
h = g.get_hamiltonian(has_spin=False) # get the Hamiltonian,spinless
hv = h.copy() # copy Hamiltonian to create a defective one
hv.add_onsite(lambda r: (np.sum((r - g.r[0])**2)<1e-2)*100) # add a defect
eb = embedding.Embedding(h,m=hv) # create an embedding object
(x,y,d) = eb.get_ldos(nsuper=20,energy=0.,delta=1e-2) # compute LDOS
#(x,y,d) = eb.get_didv(nsuper=10,T=0.1,energy=0.,delta=1e-2) # compute LDOS

#np.savetxt("LDOS.OUT",np.array([x,y,d]).T)







