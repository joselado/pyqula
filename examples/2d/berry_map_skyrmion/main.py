# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")





from pyqula import geometry
from pyqula import hamiltonians
import numpy as np
g = geometry.square_lattice()
n = 6
g = g.supercell(n)
g.center()
h = g.get_hamiltonian()
def fm(r):
  r2 = r.dot(r)  
  theta = np.tanh(np.sqrt(r2)/n)*np.pi
  mz = np.cos(theta) # mz
  ur = r/np.sqrt(r2)*np.sin(theta)
  ur[2] = mz
  return ur*3.0
g.write()
h.add_magnetism(fm)
h.write_magnetization()
#h.get_bands()
h.shift_fermi(4.0)
#exit()
from pyqula import topology
topology.Omega_rmap(h,k=[0.0,0.0,0.0],nrep=1,integral=True)
#topology.Omega_rmap(h,k=[0.5,0.0,0.0],nrep=3,integral=False)
#topology.Omega_rmap(h,k=[0.0,-0.0,0.0],nrep=3,integral=False)

(x,y,d,z) = np.genfromtxt("BERRY_RMAP.OUT").T

import matplotlib.pyplot as plt

plt.scatter(x,y,c=d,cmap="bwr")
plt.colorbar(label="Berry density")
plt.xlabel("x")
plt.ylabel("y")
plt.show()







