# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")

from pyqula import geometry
g = geometry.honeycomb_lattice()
g = g.get_supercell(4)
h = g.get_hamiltonian(has_spin=False)

from pyqula import potentials

f = potentials.commensurate_potential(g,minmax=[-1,1.])
f = f*0.3 # redefine the amplitude of the modulation
g.write_profile(f) # write the profile in a file 
h.add_sublattice_imbalance(f) # add a sublattice imbalance with this profile
h.get_ldos(e=0.0,num_bands=20) # compute LDOS
h.get_bands() # compute band structure







