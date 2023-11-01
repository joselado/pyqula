# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")

from pyqula import geometry
from pyqula import classicalspin
import numpy as np
g = geometry.triangular_lattice() # generate the geometry
g = g.get_supercell(3)


from pyqula.classicalspintk.align import most_perpendicular_vector
from pyqula.classicalspintk.align import most_perp_basis

sm = classicalspin.SpinModel(g) # generate a spin model
sm.add_heisenberg(Jij=[1.0],Jm=[1.,1.,1.]) # add heisenberg exchange
sm.minimize_energy(tries=1) # minimize Hamiltonian

ms = sm.magnetization
ms = most_perp_basis(ms) # rotate to the the most xy basis
mx = ms[:,0] 
my = ms[:,1]
mz = ms[:,2]
print(mz)
x = g.x
y = g.y
import matplotlib.pyplot as plt
plt.quiver(x,y,mx,my,cmap="bwr")
#plt.scatter(x,y,c=mz,s=200,cmap="bwr")
plt.show()




