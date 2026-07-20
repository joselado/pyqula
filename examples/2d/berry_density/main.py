# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")





from pyqula import geometry
from pyqula import topology
g = geometry.honeycomb_lattice()
h = g.get_hamiltonian(has_spin=False)
h.add_haldane(0.1)
(es,cs,csi) = topology.chern_density(h,write=True)
#h.get_bands()
#dos.dos(h,nk=100,use_kpm=True)

import matplotlib.pyplot as plt

plt.subplot(1,2,1)
plt.plot(es,cs)
plt.xlabel("Energy")
plt.ylabel("Berry curvature density")
plt.subplot(1,2,2)
plt.plot(es,csi)
plt.xlabel("Energy")
plt.ylabel("Integrated Chern density")
plt.show()







