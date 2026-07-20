# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")





from pyqula import geometry
g = geometry.triangular_lattice(n=13)
g.write(nrep=1)

import matplotlib.pyplot as plt

plt.scatter(g.x,g.y)
plt.xlabel("x") ; plt.ylabel("y") ; plt.axis("equal")
plt.show()







