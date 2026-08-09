# Add the root path of the pyqula library
import os ; import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")


from pyqula import geometry
from pyqula import latticeising
import numpy as np

g = geometry.triangular_lattice() # geometrically frustrated for AFM coupling
g = g.get_supercell(10)
g.dimensionality = 0

li = latticeising.LatticeIsing(g,m=0.0)
li.add_interaction(Jij=[-1.]) # first-neighbor antiferromagnetic coupling
li.optimize_conserved(temp=0.1,ntries=2e4) # fixed-magnetization Kawasaki dynamics

import matplotlib.pyplot as plt

z = li.s
plt.subplot(1,3,1)
plt.scatter(g.x,g.y,c=z,marker="o",cmap="bwr",s=30)
plt.colorbar(location="bottom")
plt.axis("equal")
plt.axis("off")
plt.title("Spin")

z = li.get_local_energy()
plt.subplot(1,3,2)
plt.scatter(g.x,g.y,c=z,marker="o",cmap="rainbow",s=30)
plt.colorbar(location="bottom")
plt.axis("equal")
plt.axis("off")
plt.title("Local energy")

z = li.get_local_field()
plt.subplot(1,3,3)
plt.scatter(g.x,g.y,c=z,marker="o",cmap="rainbow",s=30)
plt.colorbar(location="bottom")
plt.axis("equal")
plt.axis("off")
plt.title("Local field")

plt.tight_layout()
plt.show()
