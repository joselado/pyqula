# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")





# zigzag ribbon
import numpy as np
from pyqula import geometry
from scipy.sparse import csc_matrix
from pyqula import meanfield
g = geometry.chain()
#g = g.supercell(2)
h = g.get_hamiltonian() # create hamiltonian of the system

# We will take a ferromagnetic triangular lattice, whose
# superconducting state is a p-wave superfluid state with
# odd superconductivity

h.add_zeeman([0.,0.,20.0]) # add ferromagnetism
h.turn_nambu() # setup a Nambu hamiltonian


# the interaction terms are controlled by U, V1 and V2
# U is the onsite Hubbard interaction
# V1 is the first neighbor charge-charge interaction
# V2 is the second neighbor charge-charge interaction
mf = meanfield.guess(h,"random") # random intialization
scf = meanfield.Vinteraction(h,U=0.0,V1=-6.0,V2=0.0,
        nk=20,filling=0.2,mf=mf,mix=0.3)
from pyqula import scftypes



print("##########################################")
print("Symmetry breaking created by interactions")
print(scf.identify_symmetry_breaking())
print("##########################################")



# now extract the Hamiltonian and compute the bands
h = scf.hamiltonian # get the Hamiltonian
h.get_bands(operator="electron") # calculate band structure







