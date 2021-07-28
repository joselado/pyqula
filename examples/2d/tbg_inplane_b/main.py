# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")





import numpy as np
from pyqula import parallel
parallel.cores = 7 # parallelize in 7 cores

#### create the geometry ####
from pyqula import specialgeometry
g = specialgeometry.twisted_bilayer(6) # geometry of 2D TBG
# the reciprocal lattice vectors are g.a1 and g.a2

### create the Hamiltonian ####
from pyqula.specialhopping import twisted_matrix
ti = 0.0 # interlayer hopping
#ti = 0.2 # interlayer hopping
h = g.get_hamiltonian(is_sparse=True,has_spin=False,
     mgenerator=twisted_matrix(ti=ti))
b = 0.02 # magnetic field
#b = 0.035 # magnetic field
bphi = 0.828 # angle of the magnetic field (find the proper one)
bias = 0.0 # interlayer bias
#bias = 0.05 # interlayer bias
h.add_inplane_bfield(b=b,phi=bphi) # add inplane magnetic field
h.add_onsite(lambda r: np.sign(r[2])*bias) # add interlayer bias
# incase you want to do a ribbon, uncomment the following lines
#from pyqula import multicell
#h = multicell.bulk2ribbon(h,n=20)


### compute band structure ####
h.get_bands(num_bands=20,operator="zposition")







