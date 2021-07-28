# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")





from pyqula import geometry
from pyqula import topology
from pyqula import klist
g = geometry.honeycomb_lattice()
h = g.get_hamiltonian(has_spin=False)
#h.add_haldane(0.05)
#h.add_zeeman(0.3)
#h.add_rashba(0.3)
h.add_sublattice_imbalance(0.6)
from pyqula import dos
from pyqula import topology
#op = h.get_operator("valley",projector=True) # valley operator
op = None
#topology.write_berry(h,mode="Green",operator=op,delta=0.00001)
op = None
(x1,y1) = topology.write_berry(h,mode="Wilson",operator=op)
(x,y) = topology.write_berry(h,mode="Green",operator=op)
import matplotlib.pyplot as plt
plt.plot(x,y,c="blue",label="Wilson")
plt.scatter(x1,y1,c="red",label="Berry")
plt.legend()
plt.show()







