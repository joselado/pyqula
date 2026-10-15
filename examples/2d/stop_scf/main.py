# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")


import numpy as np
from pyqula import geometry
from pyqula.meanfield import MeanFieldConvergenceError

g = geometry.honeycomb_lattice()
h = g.get_hamiltonian() # create hamiltonian of the system
try:
    h = h.get_mean_field_hamiltonian(U=3.0,mf="antiferro",
              maxite=3, # stop after this many iterations
              verbose=1 # print info as a reference
              )
except MeanFieldConvergenceError as error:
    # if the SCF does not converge, an error is raised
    print(error) ; h = None
print("Output Hamiltonian",h)

if h is not None: # only plot if the SCF converged
    (k,e) = h.get_bands()

    import matplotlib.pyplot as plt

    plt.scatter(k,e)
    plt.xlabel("k-path") ; plt.ylabel("Energy")
    plt.show()
