# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")

from pyqula import geometry
import numpy as np
g = geometry.chain()
h = g.get_hamiltonian(tij=[0.5,0.,0.,0.5],has_spin=True)
h.add_rashba(0.7)
energies = np.linspace(-4.,4.0,60)

(e1,d1) = h.get_dos(energies=energies,delta=1e-2,mode="ED",nk=1000)
(e2,d2) = h.get_dos(energies=energies,delta=1e-2,mode="Green")

import matplotlib.pyplot as plt

plt.plot(e1,d1,label="ED")
plt.plot(e2,np.array(d2)/np.pi,label="Green")
plt.legend()
plt.show()
