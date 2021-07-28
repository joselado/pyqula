# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")





from pyqula import geometry
from pyqula import hamiltonians
import numpy as np
from pyqula import specialgeometry
g = specialgeometry.twisted_multilayer(6,rot=[0,1,0])
g.write()
from pyqula.specialhopping import twisted_matrix
h = g.get_hamiltonian(is_sparse=True,has_spin=False,is_multicell=False,
     mgenerator=twisted_matrix(ti=0.4,lambi=7.0))
#def ff(r): return 0.05*r[2]
h.get_bands(num_bands=40)







