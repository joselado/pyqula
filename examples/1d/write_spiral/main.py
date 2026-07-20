# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")




from pyqula import geometry
from pyqula import scftypes
import numpy as np
# create the hamiltonian
g = geometry.chain()
g = g.supercell(20)
h = g.get_hamiltonian(has_spin=True) # create hamiltonian of the system
###################

# This is the direction around which we rotate the magnetization
vector = [1.,0.,0.]
q = np.array([0.5,0.0,0.0])
# rotate the Hamiltonian
h.add_zeeman([0.,.0,0.5])
h.generate_spin_spiral(vector=vector,qspiral=-q,fractional=True)
#h = h.supercell(4)
h.write_magnetization(nrep=1)

import matplotlib.pyplot as plt

m = np.genfromtxt("MAGNETISM.OUT").T
x,mx,my,mz = m[0],m[3],m[4],m[5]
plt.plot(x,mx,label="Mx")
plt.plot(x,my,label="My")
plt.plot(x,mz,label="Mz")
plt.xlabel("x") ; plt.ylabel("Magnetization")
plt.legend()
plt.show()








