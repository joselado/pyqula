# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")



import numpy as np
from pyqula import geometry
g = geometry.triangular_lattice()
h = g.get_hamiltonian(has_spin=False)
h.get_qpi(delta=1e-1,mode="pm",info=True,nk=50,
        energies=np.linspace(-6.,6.,100))
# get_qpi does not return arrays, it writes DOS.OUT (among other files)
(es,dos) = np.loadtxt("DOS.OUT").T

import matplotlib.pyplot as plt

plt.plot(es,dos)
plt.xlabel("Energy")
plt.ylabel("DOS")
plt.show()







