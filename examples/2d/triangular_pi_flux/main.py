# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")





from pyqula import geometry,specialhamiltonian
h = specialhamiltonian.triangular_pi_flux(has_spin=False) # get the pi-flux hamiltonian
h.get_bands()
h.write_hopping()







