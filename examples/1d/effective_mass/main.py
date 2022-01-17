# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")



import numpy as np
from pyqula import geometry
g = geometry.chain()
h = g.get_hamiltonian(has_spin=False) # create hamiltonian of the system
#h.add_rashba(1.)
#h.add_exchange([0.,0.,1.])
from pyqula import mass
h.get_bands(operator="mass")



