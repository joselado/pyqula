# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")





from pyqula import geometry
g = geometry.honeycomb_lattice() # create a honeycomb lattice
h = g.get_hamiltonian() # get the Hamiltonian
h.add_sublattice_imbalance(1.0) # add sublattice imbalance
h.get_bands() # compute bands







