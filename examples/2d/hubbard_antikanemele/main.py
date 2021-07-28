# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")





# zigzag ribbon
import numpy as np
from pyqula import geometry
from pyqula import scftypes
from pyqula import operators
from scipy.sparse import csc_matrix
g = geometry.honeycomb_lattice()
h = g.get_hamiltonian() # create hamiltonian of the system
h.add_anti_kane_mele(0.1)
mf = scftypes.guess(h,mode="random") # antiferro initialization
# perform SCF with specialized routine for Hubbard
U = 3.0
scf = scftypes.hubbardscf(h,nkp=10,filling=0.5,g=U,
              mix=0.9,mf=mf)
scf.hamiltonian.get_bands(operator="sz")







