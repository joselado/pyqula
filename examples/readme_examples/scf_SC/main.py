# Add the root path of the pyqula library
import os ; import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")

from pyqula import geometry
import numpy as np
g = geometry.triangular_lattice() # geometry of a Lieb lattice
h = g.get_hamiltonian()  # get the Hamiltonian 
h.turn_nambu() # setup the Nambu form of the Hamiltonian
h = h.get_mean_field_hamiltonian(U=-1.0,filling=0.05,mf="swave") # perform SCF
# electron spectral-funtion
h.get_kdos_bands(operator="electron",nk=400,energies=np.linspace(-1.0,1.0,100)) 


