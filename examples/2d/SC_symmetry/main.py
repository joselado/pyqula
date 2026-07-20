# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")


import numpy as np

from pyqula import geometry
g = geometry.triangular_lattice()
g = g.get_supercell(3)
h = g.get_hamiltonian()
h.add_pairing(mode="nodal_fwave",delta=1.,d=[1,0,0])
h.add_pairing(mode="swave",delta=0.5)

# to get singlet/triplet order parameters
print("d-vector",h.get_average_dvector())
print("absolute triplet",h.extract("absolute_delta",mode="triplet",nk=20))
print("absolute singlet",h.extract("absolute_delta",mode="singlet",nk=20))


(kx,ky,d) = h.extract("deltak",mode="all")
(kxs,kys,ds) = h.extract("deltak",mode="singlet")
(kxt,kyt,dt) = h.extract("deltak",mode="triplet")
np.savetxt("DELTA.OUT",np.array([kx,ky,d]).real.T)
np.savetxt("DELTA_SINGLET.OUT",np.array([kxs,kys,ds]).real.T)
np.savetxt("DELTA_TRIPLET.OUT",np.array([kxt,kyt,dt]).real.T)

import matplotlib.pyplot as plt

plt.subplot(1,3,1)
plt.title("all")
plt.scatter(kx,ky,c=d.real,cmap="inferno")
plt.xlabel("kx") ; plt.ylabel("ky")
plt.subplot(1,3,2)
plt.title("singlet")
plt.scatter(kxs,kys,c=ds.real,cmap="inferno")
plt.xlabel("kx") ; plt.ylabel("ky")
plt.subplot(1,3,3)
plt.title("triplet")
plt.scatter(kxt,kyt,c=dt.real,cmap="inferno")
plt.xlabel("kx") ; plt.ylabel("ky")
plt.tight_layout()
plt.show()






