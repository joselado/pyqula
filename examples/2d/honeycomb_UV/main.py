# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")





# zigzag ribbon
import numpy as np
from pyqula import geometry
from scipy.sparse import csc_matrix
from pyqula import meanfield
g = geometry.honeycomb_lattice()
#g = geometry.kagome_lattice()
filling = 0.5
nk = 10
h = g.get_hamiltonian(has_spin=True) # create hamiltonian of the system
mf = meanfield.guess(h,"random")
scf = meanfield.Vinteraction(h,U=0.0,V1=4.0,nk=nk,filling=filling,mf=mf)
from pyqula import scftypes
print("Symmetry breaking",scf.identify_symmetry_breaking()) 
scf.hamiltonian.get_bands() # get the Hamiltonian







