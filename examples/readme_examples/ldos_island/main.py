# Add the root path of the pyqula library
import os ; import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")

from pyqula import islands
import numpy as np
g = islands.get_geometry(name="honeycomb",n=3,nedges=3) # get an island
h = g.get_hamiltonian() # get the Hamiltonian
h.get_multildos(projection="atomic") # get the LDOS, written to MULTILDOS/

import matplotlib.pyplot as plt

# pick a representative energy among the ones written to MULTILDOS/
names = open("MULTILDOS/MULTILDOS.TXT").read().split()
name = names[len(names)//2]
x,y,d = np.genfromtxt("MULTILDOS/"+name).T
plt.scatter(x,y,c=d,cmap="inferno")
plt.colorbar(label="LDOS")
plt.axis("equal")
plt.xlabel("x") ; plt.ylabel("y")
plt.show()



