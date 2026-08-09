# Add the root path of the pyqula library
import os ; import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")


from pyqula import geometry
from pyqula import latticeising
import numpy as np

g = geometry.square_lattice() # bipartite lattice, ferromagnetic order not frustrated
g = g.get_supercell(12)
g.dimensionality = 0

li = latticeising.LatticeIsing(g,m=0.0) # random +-1 spins, zero net magnetization
li.add_interaction(Jij=[1.]) # first-neighbor ferromagnetic coupling

es,ms = li.anneal(temps=np.geomspace(4.,0.05,15),ntries=2e4)

print("Final magnetization per site",li.get_magnetization())

import matplotlib.pyplot as plt

plt.subplot(1,2,1)
plt.plot(range(len(es)),es)
plt.xlabel("Iteration")
plt.ylabel("Energy")

plt.subplot(1,2,2)
plt.plot(range(len(ms)),ms/li.nsites)
plt.xlabel("Iteration")
plt.ylabel("Magnetization per site")

plt.tight_layout()
plt.show()
