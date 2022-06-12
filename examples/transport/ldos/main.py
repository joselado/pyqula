# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")


import matplotlib.pyplot as plt


from pyqula import geometry
from pyqula import heterostructures
import numpy as np

g = geometry.honeycomb_lattice() # create the geometry
h = g.get_hamiltonian() # create the Hamiltonian
h.remove_spin()
h1 = h.copy() # first lead
h2 = h.copy() # second lead
h1.add_sublattice_imbalance(0.3)
h2.add_sublattice_imbalance(-0.3)
central = [h2.copy() for i in range(10)]
central += [h1.copy() for i in range(10)]
HT = heterostructures.build(h1,h2,central=central) # create the junction
#HT.get_dos(ic=10,delta=1e-1)
from pyqula import parallel
parallel.cores = 4
#HT.get_kdos(delta=3e-1,ic=10)
HT.get_ldos(energy=0.,delta=3e-1)
