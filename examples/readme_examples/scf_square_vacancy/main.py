# Add the root path of the pyqula library
import os ; import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")

from pyqula import geometry
g = geometry.square_lattice() # geometry of a square lattice
g = g.get_supercell([7,7]) # generate a 5x5 supercell
g = g.remove(i=g.get_central()[0]) # remove the central site
h = g.get_hamiltonian() # create hamiltonian of the system
h.add_rashba(.4) # add Rashba spin-orbit coupling
h = h.get_mean_field_hamiltonian(U=2.0,filling=0.5,mf="random") # perform SCF
(k,e,c) = h.get_bands(operator="sz") # calculate band structure
m = h.get_magnetization() # get the magnetization

import matplotlib.pyplot as plt

plt.subplot(1,2,1)
plt.scatter(k,e,c=c,cmap="bwr")
plt.colorbar(label="Sz")
plt.xlabel("k-path") ; plt.ylabel("Energy")

plt.subplot(1,2,2)
x,y = h.geometry.r[:,0],h.geometry.r[:,1]
plt.quiver(x,y,m[:,0],m[:,1])
plt.scatter(x,y,c=m[:,2],cmap="bwr")
plt.colorbar(label="Mz")
plt.axis("equal")
plt.xlabel("x") ; plt.ylabel("y")

plt.tight_layout()
plt.show()



