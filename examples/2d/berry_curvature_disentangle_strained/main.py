# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")




import numpy as np
from pyqula import geometry
from pyqula import topology
g = geometry.honeycomb_lattice()
g = g.get_supercell(4)
h = g.get_hamiltonian()
h.remove_spin()

from pyqula import potentials
f3 = potentials.commensurate_potential(g,minmax=[0.6,1.0])
h.add_sublattice_imbalance(1./3.*f3)
h.add_strain(f3,mode="scalar")
#h.write_hopping()
h.get_bands(operator="valley") ; exit()

from pyqula.topologytk.occstates import states_generator,filter_state
from pyqula.topologytk.occstates import max_valence_states

focc = max_valence_states(h,n=2)
fsz = filter_state(h.get_operator("valley"),accept= lambda e: e<0.)
h.os_gen = states_generator(h,filt=focc)

#(x,y) = topology.write_berry(h,nk=60)
from pyqula import parallel
parallel.cores = 6
h.get_berry_curvature(nk=50) ; exit()
import matplotlib.pyplot as plt
plt.plot(x,y,c="blue",label="Berry")
plt.legend()
plt.show()







