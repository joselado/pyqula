# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")



import numpy as np
from pyqula import geometry

g = geometry.square_lattice()
g0 = geometry.triangular_lattice()
g = g0.get_supercell(3)
r0 = g.r[0] + (g0.a1 + g0.a2*2)/3. # center of the skyrmion
h = g.get_hamiltonian()
from pyqula import potentials

f = potentials.commensurate_skyrmion(g,r0=r0)
h.add_exchange(f)
h.write_magnetization(nrep=2)


m = np.genfromtxt("MAGNETISM.OUT").T
x,y,mx,my,mz = m[0],m[1],m[3],m[4],m[5]
print(x.shape,mx.shape)
import matplotlib.pyplot as plt

plt.quiver(x,y,mx,my)
plt.scatter(x,y,c=mz,cmap="bwr")
plt.show()

