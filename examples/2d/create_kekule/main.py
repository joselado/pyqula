# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")





from pyqula import islands
import numpy as np
from pyqula import kekule
# geometry of a graphene island
g = islands.get_geometry(name="honeycomb",n=6,nedges=6)
rk = kekule.kekule_positions(g.r) # return centers of the jkekule ordering


# now plot the different positions

import matplotlib.pyplot as plt
m = np.array(g.r).T # positions of the honeycomb lattice
mc = np.array(rk).T # positions of the Kekule ordering
print(m.shape)
print(mc.shape)
plt.scatter(m[0],m[1],c="red",s=60,label="lattice")
plt.scatter(mc[0],mc[1],c="blue",s=200,label="Kekule center")
plt.legend()

plt.show()







