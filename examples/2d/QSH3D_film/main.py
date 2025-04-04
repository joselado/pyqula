# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")


from pyqula import specialhamiltonian
from pyqula import geometry,films
import numpy as np
# generate a film of a 3D QSH
W=20
g0 = geometry.diamond_lattice_minimal()
g = films.geometry_film(g0,nz=W)
zmax = np.max(g.r[:,2])
zmin = np.min(g.r[:,2])
def rem(r):
    if abs(r[2]-zmax)<0.01: return True
    if abs(r[2]-zmin)<0.01: return True
    return False
g = g.remove(rem)

h = specialhamiltonian.QSH3D_film(g=g,soc=0.1) 
(k,e,c) = h.get_bands(operator="surface")

import matplotlib.pyplot as plt

plt.scatter(k,e,c=c)
plt.xticks([]) ; plt.xlabel("momentum") ; plt.ylabel("Energy")
plt.ylim([min(e),max(e)]) ; plt.xlim([min(k),max(k)])

plt.show()








