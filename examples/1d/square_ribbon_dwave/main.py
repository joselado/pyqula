# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")

from pyqula import geometry
g = geometry.square_ribbon(20) # create geometry 
h = g.get_hamiltonian() # create hamiltonian of the system
h.add_onsite(2.0) # shift the chemical potential
h.add_pairing(delta=0.5,mode="dx2y2") # d-wave superconductivity
Op = h.get_operator("electron") # electron
(ks,es,ps) = h.get_bands(operator=Op)

import matplotlib.pyplot as plt
plt.scatter(ks,es,s=20*ps)
plt.xticks([]) ; plt.xlim([min(ks),max(ks)])
plt.xlabel("Momentum") ; plt.ylabel("Energy")
plt.show()

