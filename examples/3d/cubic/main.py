# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")





import numpy as np
from pyqula import geometry
from pyqula import dos
g = geometry.cubic_lattice()
g.write()
h = g.get_hamiltonian()
# A well converged DOS requires more k-points
h.turn_dense()
h.get_bands()
dos.autodos(h,nk=100,auto=True,delta=0.1,energies=np.linspace(-6.0,6.0,1000))







