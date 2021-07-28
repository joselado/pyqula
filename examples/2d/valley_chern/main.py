# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")





from pyqula import geometry
from pyqula import topology
g = geometry.honeycomb_lattice()
h = g.get_hamiltonian(has_spin=False)
h.add_sublattice_imbalance(0.2)
op = h.get_operator("valley",projector=True) # valley operator
c = topology.chern(h,mode="Green",delta=0.0001,nk=20,operator=op)
print("")
print(c)







