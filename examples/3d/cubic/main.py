# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")





import numpy as np
from pyqula import geometry
from pyqula import dos
g = geometry.cubic_lattice()
g.write()
h = g.get_hamiltonian()
# A well converged DOS requires more k-points
h.turn_dense()
(k,e) = h.get_bands()
(xdos,ydos) = dos.autodos(h,nk=100,auto=True,delta=0.1,energies=np.linspace(-6.0,6.0,1000))

import matplotlib.pyplot as plt

plt.subplot(1,2,1)
plt.scatter(k,e)
plt.xlabel("k-path")
plt.ylabel("Energy")
plt.subplot(1,2,2)
plt.plot(xdos,ydos)
plt.xlabel("Energy")
plt.ylabel("DOS")
plt.show()







