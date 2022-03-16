# Add the root path of the pyqula library
import os ; import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")

from pyqula import geometry
import numpy as np
from pyqula import geometry
g = geometry.triangular_lattice() # generate the geometry
h = g.get_hamiltonian() # create Hamiltonian of the system
h.add_exchange([0.,0.,1.]) # add exchange field
h.setup_nambu_spinor() # initialize the Nambu basis
# perform a superconducting non-collinear mean-field calculation
h = h.get_mean_field_hamiltonian(V1=-1.0,filling=0.3,mf="random")
# compute the non-unitarity of the spin-triplet superconducting d-vector
d = h.get_dvector_non_unitarity() # non-unitarity of spin-triplet
# electron spectral-funtion
h.get_kdos_bands(operator="electron",nk=400,energies=np.linspace(-2.0,2.0,400)) 


