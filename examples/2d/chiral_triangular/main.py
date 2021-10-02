# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")





from pyqula import geometry
from pyqula import specialhopping
import numpy as np
g = geometry.triangular_lattice()
tij = specialhopping.phase_C3(g,phi=0.3)
h = g.get_hamiltonian(has_spin=False,tij=tij)
h.turn_spinful(enforce_tr=True)
h.turn_dense()
h.get_bands(operator="sz")







