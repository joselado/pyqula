# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")





from pyqula import geometry
from pyqula import topology
from pyqula import dos
g = geometry.honeycomb_lattice()
h = g.get_hamiltonian(has_spin=True)
h.add_haldane(0.1)
h.add_sublattice_imbalance(0.1)
from pyqula.topology import z2_vanderbilt
wc = z2_vanderbilt(h,full=True) # wc[0] is t, wc[1:] are Wannier center angles
(k,e) = h.get_bands()

import matplotlib.pyplot as plt

plt.subplot(1,2,1)
plt.title("Wannier centers")
for row in wc[1:]:
    plt.scatter(wc[0],row,c="black",s=5)
plt.xlabel("t") ; plt.ylabel("Wannier center angle")
plt.subplot(1,2,2)
plt.title("Bands")
plt.scatter(k,e)
plt.xlabel("k-path") ; plt.ylabel("Energy")
plt.tight_layout()
plt.show()







