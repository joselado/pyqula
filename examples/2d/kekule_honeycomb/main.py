# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")


from pyqula import geometry
g = geometry.honeycomb_lattice_C6()
g = geometry.honeycomb_lattice()
g = g.get_supercell(3)
h = g.get_hamiltonian(has_spin=False) # create hamiltonian of the system
h.add_kekule(0.2) # add a Kekule distortion
#h.add_sublattice_imbalance(0.2) # add sublattice imbalance
h.write_hopping() # you can plot this with ql-network
h.get_bands() # calculate band structure







