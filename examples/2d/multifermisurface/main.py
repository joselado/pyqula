# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")





import numpy as np
from pyqula import geometry
g = geometry.square_lattice()
h = g.get_hamiltonian()
h.get_qpi(delta=0.1,nk=80,energies=np.linspace(-4,4,101))
# get_qpi does not return arrays, it writes DOS.OUT (among other files)
(es,dos) = np.loadtxt("DOS.OUT").T

import matplotlib.pyplot as plt

plt.plot(es,dos)
plt.xlabel("Energy")
plt.ylabel("DOS")
plt.show()









