# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")





from pyqula import geometry
from pyqula import heterostructures
import numpy as np
import matplotlib.pyplot as plt
g = geometry.chain()
h = g.get_hamiltonian()
h1 = h.copy() # copy
h2 = h.copy() # copy
h1.add_swave(.0) # add electron hole symmetry
h2.add_swave(.01) # pairing gap of 0.01
ht = heterostructures.create_leads_and_central(h1,h2,h1) # create the junction
ht.delta = 1e-6 # imaginary part of the energy
es = np.linspace(-.05,.05,100) # grid of energies
ts = [ht.didv(energy=e) for e in es] # calculate transmission
plt.plot(es,ts,marker="o")
plt.ylim([0,4.1])
plt.show()







