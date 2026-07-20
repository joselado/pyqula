# Add the root path of the pyqula library
import os ; import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")




from pyqula import geometry
g = geometry.honeycomb_lattice()
h = g.get_hamiltonian()
h.add_onsite(0.6)

# Default: evaluate the spectral weight at every point of the nk x nk mesh
(kx1,ky1,fs1) = h.get_fermi_surface(nk=128,delta=0.3,write=False)
print("Grid backend: ",len(fs1),"k-points evaluated")

# Alternative: reconstruct the mesh with qutecipy's quantics tensor cross
# interpolation (QTCI), diagonalizing only a fraction of the mesh's k-points
(kx2,ky2,fs2) = h.get_fermi_surface(nk=128,delta=0.3,write=False,
        backend="qtci",tolerance=1e-3)
print("qtci backend: ",len(fs2),"k-points in the reconstructed mesh")

import numpy as np
print("Max difference between the two maps: ",np.max(np.abs(fs1-fs2)))

import matplotlib.pyplot as plt

plt.subplot(1,2,1)
plt.title("grid backend")
plt.scatter(kx1,ky1,c=fs1,cmap="inferno")
plt.xlabel("kx") ; plt.ylabel("ky")
plt.subplot(1,2,2)
plt.title("qtci backend")
plt.scatter(kx2,ky2,c=fs2,cmap="inferno")
plt.xlabel("kx") ; plt.ylabel("ky")
plt.tight_layout()
plt.show()
