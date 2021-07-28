# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")





# zigzag ribbon
import numpy as np
from pyqula import geometry
from pyqula import scftypes
from pyqula import operators
from scipy.sparse import csc_matrix
from pyqula import parallel

#parallel.cores = 4
g = geometry.triangular_lattice()
#g = geometry.bichain()
g = g.supercell(6)
#g = geometry.triangular_lattice()
#g = g.supercell(3)
h = g.get_hamiltonian(has_spin=True) # create hamiltonian of the system
h = h.get_multicell()
mf = scftypes.guess(h,mode="ferro",fun=1.0) 
def vfun(r):
    if r<1e-2: return 0.0
    else: return 2.0*np.exp(-r)
scf = scftypes.selfconsistency(h,nkp=10,filling=0.5,g=3.0,
                mix=0.9,mf=mf,mode="fastCoulomb",vfun=vfun)
h = scf.hamiltonian
h.get_bands(operator="sz")
#print(h.extract("density"))







