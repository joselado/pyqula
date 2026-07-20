# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")





import numpy as np
from pyqula import geometry
g = geometry.diamond_lattice_minimal()
from pyqula import films
g = films.geometry_film(g,nz=2)
h = g.get_hamiltonian(has_spin=False)
h = h.get_multicell()
h.add_inplane_bfield(b=0.05)
(k,e,c) = h.get_bands(operator="zposition")
(energies,dos) = h.get_dos(nk=50)

import matplotlib.pyplot as plt

plt.subplot(1,2,1)
plt.scatter(k,e,c=c,cmap="rainbow")
plt.colorbar(label="z position")
plt.xlabel("k-path") ; plt.ylabel("Energy")
plt.subplot(1,2,2)
plt.plot(energies,dos)
plt.xlabel("Energy") ; plt.ylabel("DOS")
plt.show()







