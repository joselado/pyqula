# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")


import numpy as np
from pyqula import geometry

g = geometry.honeycomb_lattice()
h = g.get_hamiltonian() # create hamiltonian of the system
h = h.get_mean_field_hamiltonian(U=3.0,mf="antiferro",
              maxite=3, # stop after this many iterations
              verbose=1 # print info as a reference
              )

# if the SCF does not converge, a None object is returned
print("Output Hamiltonian",h)
