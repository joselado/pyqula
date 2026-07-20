# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")



from pyqula import geometry
import numpy as np
from pyqula import embedding
g = geometry.honeycomb_lattice() # create geometry of a chain
#g = geometry.square_lattice() # create geometry of a chain
h = g.get_hamiltonian(has_spin=False) # get the Hamiltonian,spinless
h.add_haldane(0.1)
vintra = h.intra.copy() ; vintra[0,0] = 1000.0
h.add_onsite(0.2)
eb = embedding.Embedding(h,m=vintra) # create the embedding object
ei = eb.get_energy_ingap_state() # get energy of the impurity state
(es,ds) = eb.multidos(es=np.linspace(-1.0,1.0,200),delta=1e-2)
np.savetxt("DOS.OUT",np.array([es,ds]).T)
print("Energy of the impurity state",ei) # energy of the impurity state
(x,y,d) = eb.ldos(nsuper=5,e=ei) # get data
np.savetxt("LDOS.OUT",np.array([x,y,d]).T) # save data

import matplotlib.pyplot as plt

plt.subplot(1,2,1)
plt.plot(es,ds)
plt.xlabel("Energy")
plt.ylabel("DOS")
plt.subplot(1,2,2)
plt.scatter(x,y,c=d,cmap="inferno")
plt.colorbar(label="LDOS")
plt.axis("equal")
plt.show()

