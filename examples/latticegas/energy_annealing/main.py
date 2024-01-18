# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")


from pyqula import geometry
from pyqula import supercell
from pyqula import latticegas
import numpy as np
g = geometry.triangular_lattice() # generate the geometry
g = supercell.turn_orthorhombic(g) # make a orthorombic cell
g = g.get_supercell(10)
g.dimensionality = 0
lg = latticegas.LatticeGas(g,filling=1./3.)
lg.add_interaction(Jij=[1.,1.,1.])


N = 100
es = []
#for i in range(N):
#    e = lg.optimize_energy(temp=0.1,ntries=100)
#    es.append(e[-1])

es = lg.optimize_energy(temp=2.,ntries=1e4)


import matplotlib.pyplot as plt

plt.plot(range((len(es))),es)
plt.xlabel("Iteration")
plt.ylabel("Energy")
plt.show()


