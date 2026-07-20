# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")





from pyqula import geometry
from pyqula import topology
from pyqula import operators
g = geometry.honeycomb_lattice()
#g = geometry.honeycomb_lattice_C6()
#g = geometry.chain()
g = g.supercell(6)
g.write()
g.dimensionality = 0
h = g.get_hamiltonian(has_spin=True)
h = h.get_multicell()
from pyqula import groundstate
import numpy as np
groundstate.hopping(h,nrep=2) # write three replicas

import matplotlib.pyplot as plt
data = np.genfromtxt("HOPPING.OUT")
x1,y1,x2,y2,t = data[:,0],data[:,1],data[:,2],data[:,3],data[:,4]
for i in range(len(t)):
    plt.plot([x1[i],x2[i]],[y1[i],y2[i]],color=plt.cm.viridis(t[i]/np.max(t)))
plt.xlabel("x")
plt.ylabel("y")
plt.gca().set_aspect("equal")
plt.show()







