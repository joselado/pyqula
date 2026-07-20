# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")




from pyqula import specialgeometry # special Hamiltonians library
g = specialgeometry.twisted_bilayer(6) # TBG Hamiltonian

from pyqula import stacking
v = stacking.stacking(g)
g.write_profile(v)

import matplotlib.pyplot as plt

plt.scatter(g.x,g.y,c=v)
plt.colorbar(label="Stacking")
plt.xlabel("x") ; plt.ylabel("y")
plt.show()







