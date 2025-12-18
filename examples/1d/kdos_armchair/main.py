# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")



from pyqula import geometry
import numpy as np
from pyqula import parallel # try with parallel
parallel.cores = 4 # number of cores

g = geometry.honeycomb_armchair_ribbon(1000) # create geometry of armchair ribbon
h = g.get_hamiltonian(has_spin=False,is_sparse=True) # create hamiltonian 

energies = np.linspace(-4,4,500) # energies

import time

t0 = time.time() # initial time

(k,e,d) = h.get_kdos_bands(
                mode="KPM", # use the KPM mode
                delta=1e-2, # energy smearing
                energies=energies, # energies
                nk=100, # kpoints
                ntries = 1, # number of KPM vectors used
                )

t1 = time.time() # final time
print("Total time",t1-t0)

import matplotlib.pyplot as plt

nk = len(np.unique(k)) # number of kpoints
ne = len(np.unique(e)) # number of energies

d2d = d.reshape((nk,ne)).T

plt.contourf(np.unique(k),np.unique(e),d2d,cmap="inferno",levels=100)
plt.xlabel("Momentum") ; plt.xticks([])
plt.ylabel("Energy") 

plt.tight_layout()

plt.show()



