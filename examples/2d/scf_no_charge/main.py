# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")





# zigzag ribbon
import numpy as np
from pyqula import geometry
from pyqula import meanfield

g = geometry.honeycomb_lattice()
g = g.supercell(1)
h = g.get_hamiltonian(has_spin=False) # create hamiltonian of the system
U = 3.0
filling = 0.5
hscf = h.get_mean_field_hamiltonian(Vr=lambda r1,r2: 0.,filling=filling,constrains=["no_charge"])
(k,e) = h.get_bands() # calculate band structure
(kscf,escf) = hscf.get_bands()

import matplotlib.pyplot as plt

plt.scatter(k,e,label="no SCF")
plt.scatter(kscf,escf,label="SCF")
plt.legend()
plt.xlabel("k-path") ; plt.ylabel("Energy")
plt.show()







