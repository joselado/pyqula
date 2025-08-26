# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")

# plot the Chern number as a function of the energy

from pyqula import geometry
from pyqula import topology
g = geometry.honeycomb_lattice()
h = g.get_hamiltonian(has_spin=False)
h.add_haldane(0.4)
delta = 0.1
nk = 80
import numpy as np
es=np.linspace(-4.0,4.0,40)
(e,dc,c) = topology.chern_density(h,delta=delta,nk=nk,es=es)

import matplotlib.pyplot as plt
plt.subplot(1,2,1)
plt.plot(e,dc)
plt.xlabel("Energy") ; plt.ylabel("dC/dE")
plt.subplot(1,2,2)
plt.plot(e,c)
plt.xlabel("Energy") ; plt.ylabel("Chern number")
plt.tight_layout()

plt.show()



