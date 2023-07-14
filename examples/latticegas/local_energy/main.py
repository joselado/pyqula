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
lg.optimize_energy(temp=0.1,ntries=1e3)

import matplotlib.pyplot as plt

z = lg.den
plt.subplot(1,2,1)
plt.scatter(g.x,g.y,c=z,marker="o",cmap="bwr",s=30)
plt.colorbar(location="bottom")
plt.axis("equal")
plt.axis("off")

z = lg.get_local_energy()
plt.subplot(1,2,2)
plt.scatter(g.x,g.y,c=z,marker="o",cmap="rainbow",s=30)
plt.colorbar(location="bottom")
plt.axis("equal") 
plt.axis("off")

plt.tight_layout()


plt.show()
