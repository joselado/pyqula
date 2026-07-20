# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")





import numpy as np
from pyqula import geometry
g = geometry.triangular_lattice()
#g = geometry.honeycomb_lattice()
h = g.get_hamiltonian(has_spin=True)
h.turn_dense()
#h.shift_fermi(1.5)
h.set_filling(0.5,nk=20)
from pyqula import response
from pyqula import parallel
parallel.cores = 6
response.magnetic_response_map(h,nk=20,nq=40,j=[0.5,0.,0.])
#h.get_bands()
#dos.dos(h,nk=100,use_kpm=True)
# magnetic_response_map does not return arrays, it writes SUSCEPTIBILITY.OUT
(qx,qy,en) = np.genfromtxt("SUSCEPTIBILITY.OUT").T
nq = int(round(np.sqrt(len(qx))))

import matplotlib.pyplot as plt

plt.contourf(qx.reshape(nq,nq),qy.reshape(nq,nq),en.reshape(nq,nq),
        levels=100,cmap="inferno")
plt.colorbar(label="Energy")
plt.xlabel("qx")
plt.ylabel("qy")
plt.show()







