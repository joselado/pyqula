# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")



import numpy as np
from pyqula import geometry
g = geometry.honeycomb_lattice()
g = geometry.square_lattice()
#h = g.get_hamiltonian()
h = g.get_hamiltonian(tij=[1.,0.3,0.1])
h.add_onsite(3.3)
#h.add_haldane(0.05)
(k,e,ds,db) = h.get_kdos(energies=np.linspace(-1.0,1.0,40),delta=3e-2)

import matplotlib.pyplot as plt

plt.subplot(1,2,1)
plt.scatter(k,e,c=ds/np.max(ds),vmax=0.4,cmap="inferno")
plt.subplot(1,2,2)
plt.scatter(k,e,c=db/np.max(db),vmax=0.4,cmap="inferno")

plt.show()






