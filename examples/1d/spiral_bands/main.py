# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")




from pyqula import geometry
from pyqula import scftypes
import numpy as np
# create the hamiltonian
g = geometry.chain()
#g = g.get_supercell(4)
h = g.get_hamiltonian(has_spin=True) # create hamiltonian of the system
###################

# This is the direction around which we rotate the magnetization
vector = [1.,0.,0.]
q = np.array([0.2,0.0,0.0])
# rotate the Hamiltonian
h.generate_spin_spiral(vector=vector,qspiral=q)
h.add_zeeman([0.,.0,0.4])
#h.generate_spin_spiral(vector=vector,qspiral=-q,fractional=True)
h.get_bands(operator="sx")
h.write_magnetization(nrep=1)








