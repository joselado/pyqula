# Add the root path of the pyqula library
import os ; import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")


from pyqula import geometry
import numpy as np
g = geometry.triangular_lattice() # create geometry of the system
g = g.get_supercell(2) # create a supercell
h = g.get_hamiltonian() # create hamiltonian of the system
out = h.get_multi_fermi_surface(energies=np.linspace(-4,4,100),delta=1e-1)

import matplotlib.pyplot as plt

ie = len(out)//2 # pick a representative energy
kx,ky,d = out[ie]
plt.scatter(kx,ky,c=d,cmap="inferno")
plt.colorbar(label="Fermi surface weight")
plt.xlabel("kx") ; plt.ylabel("ky")
plt.show()

