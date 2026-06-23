# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")


import numpy as np

from pyqula import geometry
# generate square lattice Hamiltonian
g = geometry.square_lattice()
h = g.get_hamiltonian()


# imaginary amplitude for nodal d-wave, cos(kx) - cos(ky)
h.add_pairing(mode="nodal_dwave",delta=.3j)
# real amplitude for swave
h.add_pairing(mode="swave",delta=0.5)

(kx,ky,ds) = h.extract("deltak",mode="all")
import matplotlib.pyplot as plt

kxu = np.unique(kx)
kyu = np.unique(ky)
ds = ds.reshape((len(kxu),len(kyu)))

plt.contourf(kxu,kyu,ds,levels=40)
plt.xlabel("$k_x$")
plt.ylabel("$k_y$")
plt.colorbar(label="$|\\Delta_k|$")
plt.axis("equal")
plt.tight_layout()

plt.show()






