# Add the root path of the pyqula library
import os ; import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")

from pyqula import islands
g = islands.get_geometry(name="honeycomb",n=3,nedges=3) # get an island
h = g.get_hamiltonian() # get the Hamiltonian
h = h.get_mean_field_hamiltonian(U=1.0,filling=0.5,mf="ferro") # perform SCF
m = h.get_magnetization() # get the magnetization in each site
h.write_magnetization() # get the magnetization in each site

import matplotlib.pyplot as plt

x,y = h.geometry.r[:,0],h.geometry.r[:,1]
plt.quiver(x,y,m[:,0],m[:,1])
plt.scatter(x,y,c=m[:,2],cmap="bwr")
plt.colorbar(label="Mz")
plt.axis("equal")
plt.xlabel("x") ; plt.ylabel("y")
plt.show()



