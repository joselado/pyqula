# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")


import matplotlib.pyplot as plt


from pyqula import geometry
from pyqula import heterostructures
import numpy as np

g = geometry.chain() # create the geometry
h = g.get_hamiltonian() # create teh Hamiltonian
h1 = h.copy() # first lead
h2 = h.copy() # second lead
#h1.add_sublattice_imbalance(0.3)
#h2.add_sublattice_imbalance(-0.3)
central = [h1 for i in range(10)]
HT = heterostructures.build(h1,h2,central=central) # create the junction
es = np.linspace(-3.0,3.0,100)
es,ds = HT.get_dos(delta=1e-1,energies=es)

import matplotlib.pyplot as plt

plt.plot(es,ds)

plt.show()

