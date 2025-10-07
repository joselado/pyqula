# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")





from pyqula import geometry
from pyqula import hamiltonians
import numpy as np
from pyqula import specialgeometry
from pyqula.specialhopping import twisted_matrix
from pyqula import topology

g = specialgeometry.twisted_bilayer(9)
h = g.get_hamiltonian(is_sparse=True,has_spin=False,is_multicell=False,
     mgenerator=twisted_matrix(ti=0.4,lambi=7.0))
h.turn_dense()
def ff(r): return r[2]*0.05
h.add_onsite(ff) # interlayer bias
h.set_filling(0.5,nk=3) # set to half filling 
topology.spatial_berry_density(h,k=[-0.333333,0.33333,0.0],nrep=3,
                               operator="valley")







