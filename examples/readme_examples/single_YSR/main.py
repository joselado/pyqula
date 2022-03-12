# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")



from pyqula import geometry
import numpy as np
from pyqula import embedding
g = geometry.square_lattice() # create geometry of a chain
h = g.get_hamiltonian() # get the Hamiltonian,spinless
h.add_swave(0.1) # add s-wave superconductivity
h.add_onsite(3.0) # shift chemical potential
hv = h.copy() # copy Hamiltonian to create a defective one
hv.add_exchange(lambda r: [0.,0.,(np.sum((r - g.r[0])**2)<1e-2)*6.]) # add magnetic site
eb = embedding.Embedding(h,m=hv) # create an embedding object
ei = eb.get_energy_ingap_state() # get energy of the impurity state
(x,y,d) = eb.ldos(nsuper=19,e=ei,delta=1e-3) # compute LDOS

np.savetxt("LDOS.OUT",np.array([x,y,d]).T)







