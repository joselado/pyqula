# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")





from pyqula import geometry
from pyqula import topology
from pyqula import spectrum
g = geometry.honeycomb_lattice()
g = g.get_supercell(3)
h = g.get_hamiltonian(has_spin=False)
h.turn_dense()
h.add_onsite(0.9)

from pyqula.fermisurfacetk.swarmfs import fermi_surface
ks = fermi_surface(h,nk=100,nrep=3)

import matplotlib.pyplot as plt

plt.scatter(ks[:,0],ks[:,1],c=range(len(ks)),cmap="rainbow") 
plt.axis("equal") ; plt.colorbar()
plt.show()





