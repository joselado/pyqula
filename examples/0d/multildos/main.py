# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")





import numpy as np
from pyqula import islands
g = islands.get_geometry(name="honeycomb",n=3,nedges=3) # get an island
h = g.get_hamiltonian() # get the Hamiltonian
es = np.linspace(-2.0,2.0,100)
h.get_multildos(projection="atomic",energies=es) # get the LDOS

import matplotlib.pyplot as plt
e0 = es[len(es)//2]
data = np.genfromtxt("MULTILDOS/LDOS_"+str(e0)+"_.OUT")
plt.scatter(data[:,0],data[:,1],c=data[:,2])
plt.colorbar(label="LDOS")
plt.xlabel("x")
plt.ylabel("y")
plt.title("LDOS at E="+str(e0))
plt.show()






