# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")



from pyqula import geometry
from pyqula.specialhamiltonian import H2HFH
g = geometry.honeycomb_lattice()
h = g.get_hamiltonian()
h.add_onsite(1.0)
h = H2HFH(h,JK=0.2)
(k,e,c) = h.get_bands(operator="dispersive_electrons")
(x,y,d) = h.get_ldos(operator="electron")

import matplotlib.pyplot as plt

plt.subplot(1,2,1)
plt.scatter(k,e,c=c,cmap="inferno")
plt.colorbar(label="Dispersive electron character")
plt.xlabel("k-path") ; plt.ylabel("Energy")

plt.subplot(1,2,2)
plt.scatter(x,y,c=d,cmap="inferno")
plt.colorbar(label="LDOS")
plt.xlabel("x") ; plt.ylabel("y") ; plt.axis("equal")

plt.show()






