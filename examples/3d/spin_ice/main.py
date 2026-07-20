# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")





import numpy as np
from pyqula import geometry
from pyqula import sculpt
g = geometry.pyrochlore_lattice()
h = g.get_hamiltonian()
h.add_antiferromagnetism(1.0)
h.write_magnetization()

m = np.genfromtxt("MAGNETISM.OUT").T
x,y,mx,my,mz = m[0],m[1],m[3],m[4],m[5]
import matplotlib.pyplot as plt

plt.quiver(x,y,mx,my)
plt.scatter(x,y,c=mz,cmap="bwr")
plt.show()







