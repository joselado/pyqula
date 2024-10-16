# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")





# zigzag ribbon
import numpy as np
from pyqula import geometry
from pyqula import parallel
g = geometry.honeycomb_lattice()
Us = np.linspace(0.,4.,10) # different Us
h = g.get_hamiltonian() # create hamiltonian of the system
parallel.cores = 7
h.get_dos(nk=300)
#hmf = h.get_mean_field_hamiltonian(nk=100,U=3.) # mean field Hamiltonian







