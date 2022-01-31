# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")


import numpy as np
from pyqula import specialhamiltonian # special Hamiltonians library
h = specialhamiltonian.twisted_bilayer(n=1,  # twisted bilayer Hamiltonian
                                        ti=0.3, # interlayer hopping
                                        g0="triangular" # initial geometry
                                        )

h.geometry.write(nrep=3)  # write the geometry in a file
h.add_rashba(lambda r: 0.2*np.sign(r[2])) # stagger Rashba SOC
Op = h.get_operator("zposition")*h.get_operator("sx") # layer times Sx
(k,e,c) = h.get_bands(operator=Op) # compute band structure








