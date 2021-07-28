# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")





from pyqula import geometry
from pyqula import heterostructures
import numpy as np
import matplotlib.pyplot as plt
g = geometry.single_square_lattice()
h = g.get_hamiltonian()
h1 = h.copy() # copy
h2 = h.copy() # copy
h1.add_onsite(1.0) # add electron hole symmetry
h1.add_swave(.0) # add electron hole symmetry

# create a topological superconductor
h2.add_rashba(1.0)
h2.add_zeeman(0.5)
h2.add_onsite(4.0)
#exit()
h2.add_swave(.2) # pairing gap of 0.1
h2.get_bands()

ht = heterostructures.build(h1,h2,lc=0.2) # create the junction
ht.delta = 1e-8 # imaginary part of the energy
es = np.linspace(-0.5,0.5,100,endpoint=False) # grid of energies
ts = [ht.didv(e,nk=20) for e in es] # calculate transmission
plt.plot(es,ts,marker="o")
plt.ylim([0,4.1])
plt.show()







