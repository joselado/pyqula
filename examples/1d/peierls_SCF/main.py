# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")





import numpy as np
from pyqula import geometry
from pyqula import groundstate
from pyqula import meanfield
from scipy.sparse import csc_matrix
g = geometry.chain()
g = g.supercell(4)
h = g.get_hamiltonian(has_spin=False) # create hamiltonian of the system
nk = 10
filling = 0.5
mf = meanfield.guess(h,"dimerization")
scf = meanfield.Vinteraction(h,V1=2.0,nk=nk,filling=filling,mf=mf)
h = scf.hamiltonian # get the Hamiltonian
h.get_bands() # calculate band structure
from pyqula import topology
groundstate.hopping(h)
print(np.round(h.intra,3).real)







