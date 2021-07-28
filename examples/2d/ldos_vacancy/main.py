# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")





from pyqula import geometry
from pyqula import topology
from pyqula import ldos
import matplotlib.pyplot as plt

g = geometry.honeycomb_lattice()
g = g.supercell(30)

r0 = [0.,0.,0.]
g = g.remove(g.closest_index(r0)) # remove this site

h = g.get_hamiltonian(has_spin=False)
(es,ds) = ldos.dos_site(h,nk=5,mode="KPM",i=g.closest_index(r0))
plt.plot(es,ds)
plt.show()







