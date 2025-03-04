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

# create a topological superconductor
h2.add_rashba(1.0)
h2.add_zeeman(0.5)
h2.add_onsite(2.0)
h2.get_bands()
h2.add_swave(.1) # pairing gap of 0.1
hc = [h2,h2] # dimer 
hc = [h2 for i in range(10)] # dimer 
ht = heterostructures.build(h1,h1,central=hc) # create the junction
ht0 = heterostructures.build(h1,h2) # create the junction
ht.scale_lc = 0.3 # scale the left coupling
ht.scale_rc = 0.3 # scale the left coupling
ht.delta = 1e-6 # imaginary part of the energy
ht0.scale_lc = 0.3 # scale the left coupling
ht0.delta = 1e-6 # imaginary part of the energy
es = np.linspace(-.5,.5,200) # grid of energies
ts = [ht.didv(energy=e) for e in es] # calculate transmission
ts0 = [ht0.didv(energy=e) for e in es] # calculate transmission
plt.plot(es,ts,marker="o",label="finite")
plt.plot(es,ts0,marker="o",label="infinite")
plt.legend()
plt.show()







