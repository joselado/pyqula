# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")





# zigzag ribbon
import numpy as np
from pyqula import geometry
from pyqula import scftypes
from pyqula import parallel
parallel.cores = 7
from scipy.sparse import csc_matrix
g = geometry.honeycomb_lattice()
g.write()
Us = np.linspace(0.,4.,10) # different Us
#f = open("EVOLUTION.OUT","w") # file with the results
h = g.get_hamiltonian() # create hamiltonian of the system
mf = scftypes.guess(h,mode="antiferro") # antiferro initialization
# perform SCF with specialized routine for Hubbard
U = 3.0
scf = scftypes.hubbardscf(h,nkp=20,filling=0.5,g=U,
              mix=0.001,mf=mf)







