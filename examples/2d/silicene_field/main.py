# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")





from pyqula import geometry
from pyqula import multicell
import numpy as np

g = geometry.buckled_honeycomb_lattice() # create the geometry
g = geometry.bulk2ribbon(g,n=20) # create a ribbon from this 2D Hamiltonian
h = g.get_hamiltonian(has_spin=True) # create first neighbor Hamiltonian
h.add_onsite(lambda r: 0.2*np.sign(r[2])) # add a perpendicular electric field
h.add_kane_mele(0.1) # add Kane Mele SOC

h.get_bands(operator="sz") # compute band structure







