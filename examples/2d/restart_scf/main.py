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


U = 5.0

mf = scftypes.guess(h,mode="antiferro") # antiferro initialization
# perform SCF with specialized routine for Hubbard
# save = True will write the mean field in a file at every iteration
scf = scftypes.hubbardscf(h,nkp=20,filling=0.5,g=U,
              mix=0.9,mf=mf,save=True,silent=True)

# perform it again without giving a MF (it will be read from a file)
# mf = None forces the code to read the mean field from a file
scf = scftypes.hubbardscf(h,nkp=20,filling=0.5,g=U,
              mix=0.9,mf=None,save=True)









