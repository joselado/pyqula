# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")




from pyqula import specialgeometry # special Hamiltonians library
g = specialgeometry.tbg(6) # TBG Hamiltonian
from pyqula import potentials
f = potentials.tbgAA(g) # get a profile distinguishing AA from AB
g.write_profile(f) # write the profile in a file
# write_profile does not return arrays, it writes PROFILE.OUT (x,y,profile,z)
import numpy as np
m = np.genfromtxt("PROFILE.OUT").T
x,y,profile = m[0],m[1],m[2]

import matplotlib.pyplot as plt

plt.scatter(x,y,c=profile,cmap="bwr")
plt.colorbar(label="AA/AB stacking profile")
plt.xlabel("x")
plt.ylabel("y")
plt.show()







