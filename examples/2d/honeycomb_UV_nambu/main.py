# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")





# zigzag ribbon
import numpy as np
from pyqula import geometry
from scipy.sparse import csc_matrix
from pyqula import meanfield
g = geometry.honeycomb_lattice()
nk = 10
h = g.get_hamiltonian(has_spin=True) # create hamiltonian of the system
filling = 0.7 # filling of the system
h.turn_nambu() # setup nambu degree of freedom
mf = meanfield.guess(h,"random")
scf = meanfield.Vinteraction(h,U=-3.0,V1=0.0,nk=nk,filling=filling,mf=mf,
        verbose=1) # perform the SCF calculation
from pyqula import scftypes
print("Symmetry breaking",scf.identify_symmetry_breaking()) 
scf.hamiltonian.get_bands(operator="electron",nk=2000) # get the Hamiltonian







