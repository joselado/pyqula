# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")





# zigzag ribbon
from surface_TI import hamiltonian # this function will yield the Hamiltonian
h = hamiltonian(mw=1.0) # get the Hamiltonian, mw is the Wilson mass
h.get_bands() # get the bandstructure







