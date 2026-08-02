# Add the root path of the pyqula library
import os ; import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")


from pyqula import geometry
from pyqula import supercell
from pyqula import latticegas
import numpy as np

g = geometry.triangular_lattice() # generate the geometry
g = supercell.turn_orthorhombic(g) # make a orthorombic cell
g = g.get_supercell(6)
g.dimensionality = 0

lg = latticegas.LatticeGas(g,filling=0.0)
lg.den[:] = 0. # start from an empty lattice
lg.add_interaction(Jij=[1.,0.,0.]) # first-neighbor repulsion only

# scan the chemical potential in the grand-canonical ensemble and
# track how the filling responds -- unlike optimize_energy (fixed
# filling, swap moves), optimize_grand_canonical lets the filling
# itself fluctuate under mu
mus = np.linspace(-3.,1.,15)
fillings = []
for mu in mus:
    lg.mu[:] = mu
    lg.optimize_grand_canonical(temp=0.5,ntries=2e4)
    fillings.append(lg.den.mean())

# structure factor of the last (highest-mu) configuration: a peak away
# from q=0 signals the repulsion has driven the density into an
# ordered (e.g. striped or honeycomb-vacancy) pattern
qpath,sq = lg.get_structure_factor(nq=40)

import matplotlib.pyplot as plt

plt.subplot(1,2,1)
plt.plot(mus,fillings,marker="o")
plt.xlabel("Chemical potential")
plt.ylabel("Filling")

plt.subplot(1,2,2)
sc = plt.scatter(qpath[:,0],qpath[:,1],c=sq,cmap="inferno",s=30)
plt.colorbar(sc,location="bottom")
plt.xlabel("$q_x$")
plt.ylabel("$q_y$")
plt.axis("equal")

plt.tight_layout()
plt.show()
