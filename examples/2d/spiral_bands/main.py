# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")





from pyqula import geometry
from pyqula import scftypes
import numpy as np
# create the hamiltonian
g = geometry.honeycomb_lattice() # triangular lattice geometry
#g = geometry.chain()
#g = g.supercell(2)
h = g.get_hamiltonian(has_spin=True) # create hamiltonian of the system
q = [1./3.,1./3.,0.]
q = [0.,0.,0.]
q = np.array(q) + 0.05
vector = [0.,0.,1.]
h.generate_spin_spiral(vector=[0.,0.,1.],qspiral=q,
        fractional=True)
h.add_zeeman([.0,0.2,0.0])
#h.add_zeeman([[.0,0.1,0.0],[0.,-0.1,0.]])
h.get_bands(operator="sz")








