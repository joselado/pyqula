# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")





from pyqula import geometry
from pyqula import topology
from pyqula import dos
g = geometry.honeycomb_lattice()
h = g.get_hamiltonian(has_spin=True)
h.add_haldane(0.1)
h.add_sublattice_imbalance(0.1)
from pyqula.topology import z2_vanderbilt
z2_vanderbilt(h,full=True)
h.get_bands()







