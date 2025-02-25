# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")



from pyqula import geometry
import numpy as np

g = geometry.honeycomb_lattice()
h = g.get_hamiltonian()
h.add_sublattice_imbalance(0.1)
es = np.linspace(-0.4,0.4,100)
(es,ds) = h.get_dos(energies=es,delta=1e-3,mode="Green") # compute DOS

import matplotlib.pyplot as plt

plt.plot(es,ds)
plt.xlabel("Energy") ; plt.ylabel("DOS")
plt.ylim([0,max(ds)])

plt.show()







