# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")





from pyqula import geometry
from pyqula import topology
g = geometry.honeycomb_lattice()
h = g.get_hamiltonian()
#h.add_sublattice_imbalance(0.6)
h.add_antiferromagnetism(0.6)

from pyqula.topologytk.occstates import states_generator,filter_state

focc = filter_state(h.get_operator("energy"),accept= lambda e: e<0.)
fsz = filter_state(h.get_operator("sz"),accept= lambda e: e>0.)
h.os_gen = states_generator(h,filt=focc*fsz)

(x,y) = topology.write_berry(h)
import matplotlib.pyplot as plt
plt.plot(x,y,c="blue",label="Berry")
plt.legend()
plt.show()







