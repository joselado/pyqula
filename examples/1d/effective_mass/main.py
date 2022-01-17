# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")



import numpy as np
from pyqula import geometry
g = geometry.chain()
#g = g.get_supercell(4)
h = g.get_hamiltonian() # create hamiltonian of the system
h.add_rashba(1.)
h.add_exchange([0.,0.,1.])
from pyqula import mass
print(mass.effective_mass(h,[0.,0.,0.]))
h.get_bands(operator="mass")





