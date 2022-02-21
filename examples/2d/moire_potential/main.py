# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")

from pyqula import geometry
g = geometry.honeycomb_lattice()
g = g.get_supercell(40)
h = g.get_hamiltonian(has_spin=False)

from pyqula import potentials

f = potentials.commensurate_potential(g,minmax=[-1,1.])*2 + 1

f = f*0.3

h.add_sublattice_imbalance(f)
h.turn_sparse()
h.get_ldos(e=0.0,num_bands=20) ;exit()

#g.write_profile(f) ; exit()

h.get_bands()







