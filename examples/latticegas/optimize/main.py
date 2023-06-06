# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")


from pyqula import geometry
from pyqula import supercell
from pyqula import latticegas
import numpy as np
g = geometry.triangular_lattice() # generate the geometry
g = supercell.turn_orthorhombic(g) # make a orthorombic cell
g = g.get_supercell(8)
lg = latticegas.LatticeGas(g,filling=0.5)
lg.add_interaction(Jij=[1.,1.,0.5])
lg.optimize_energy(temp=0.1)
z = lg.den

import matplotlib.pyplot as plt

plt.scatter(g.x,g.y,c=z,marker="o",cmap="bwr",s=200)
plt.axis("equal")
plt.show()
