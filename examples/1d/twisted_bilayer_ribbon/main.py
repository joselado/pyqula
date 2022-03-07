# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")





from pyqula import geometry
from pyqula import hamiltonians
import numpy as np
import klist
import sculpt
from specialhopping import twisted,twisted_matrix
import specialgeometry
g = specialgeometry.twisted_bilayer(2) # get a small unit cell
import ribbon
g = ribbon.bulk2ribbon(g,n=4) # create a ribbon geometry
h = g.get_hamiltonian(is_sparse=True,has_spin=False,
     mgenerator=twisted_matrix(ti=0.4,lambi=7.0))
g.write() # write structure
import ldos
ldos.ldos(h,mode="arpack",nk=10,delta=0.1,nrep=3) # write the LDOS
h.get_bands(num_bands=20,operator="yposition") # write the bands







