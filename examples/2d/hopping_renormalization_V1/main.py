# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")





# zigzag ribbon
import numpy as np
from pyqula import geometry
from scipy.sparse import csc_matrix
from pyqula import meanfield
g = geometry.triangular_lattice()
h = g.get_hamiltonian() # create hamiltonian of the system
h.remove_spin() # spinless Hamiltonian
hscf = h.get_mean_field_hamiltonian(filling=0.5,nk=10,
                        V1 = 1.0, # first neighbor interaction
                        V2 = 1.0, # second neighbor interaction
                        verbose=1,
                        mix=0.9, # mixing
                        mf = "random" # random initial guess
                        )
(k,e) = h.get_bands()
(kscf,escf) = hscf.get_bands()

import matplotlib.pyplot as plt

plt.scatter(k,e,label="no SCF")
plt.scatter(kscf,escf,label="SCF")
plt.legend()
plt.show()


