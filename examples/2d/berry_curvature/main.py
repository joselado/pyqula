# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")

from pyqula import geometry
from pyqula import topology
from pyqula import dos
g = geometry.honeycomb_lattice()
h = g.get_hamiltonian(has_spin=False)
h.add_sublattice_imbalance(0.5)
kpath = ["G","K","M","K'","G"]
(ks,es) = h.get_bands(kpath=kpath)
(kx,ky,bs) = h.get_berry_curvature(kpath=kpath,reciprocal=False)
from pyqula.topology import write_berry
write_berry(h,kpath=kpath)

import matplotlib.pyplot as plt

plt.subplot(2,1,1)
plt.scatter(ks,es) 
plt.xticks([]) ; plt.xlabel("Momentum") ; plt.ylabel("Energy")
plt.subplot(2,1,2)
plt.scatter(kx,bs)
plt.xticks([]) ; plt.xlabel("Momentum") ; plt.ylabel("Berry curvature")

plt.show()


