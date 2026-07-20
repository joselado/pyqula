# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")





from pyqula import geometry
from pyqula import topology
from pyqula import spectrum
g = geometry.honeycomb_lattice()
h = g.get_hamiltonian(has_spin=True)
h.add_onsite(0.6)
(k,e) = h.get_bands()
nk = 100
(kx,ky,fs) = h.get_fermi_surface(nk=nk,operator="valley",num_waves=10,mode="lowest")

import matplotlib.pyplot as plt

plt.figure()
plt.scatter(k,e)
plt.xlabel("k-path")
plt.ylabel("Energy")

plt.figure()
plt.imshow(fs.reshape((nk,nk)),cmap="bwr")
plt.xticks([]) ; plt.yticks([])
plt.colorbar(label="Valley")
plt.show()







