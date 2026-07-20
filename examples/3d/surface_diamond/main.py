# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")





from pyqula import geometry
from pyqula import dos
g = geometry.diamond_lattice_minimal()
h = g.get_hamiltonian()
#h.add_antiferromagnetism(1.)
(energies,dosbulk,dossurf) = dos.bulkandsurface(h,nk=300,delta=0.01)
#h.get_bands()

import matplotlib.pyplot as plt

plt.plot(energies,dosbulk,label="Bulk")
plt.plot(energies,dossurf,label="Surface")
plt.xlabel("Energy")
plt.ylabel("DOS")
plt.legend()
plt.show()







