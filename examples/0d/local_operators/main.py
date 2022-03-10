# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")





from pyqula import islands
from pyqula import spectrum
from pyqula import operators

import numpy as np
g = islands.get_geometry(name="honeycomb",n=5.5,nedges=6,rot=0.0) # get an island
i = g.get_central()[0]
#g = g.remove(i)

h = g.get_hamiltonian(has_spin = False)
h.add_peierls(0.1)
h.add_sublattice_imbalance(0.1)
#h.add_haldane(0.03)
x = np.zeros(h.intra.shape[0])
#x[i] = 0.3 # set onsite
h.shift_fermi(x) # shift the Fermi eenrgy
ops = operators.get_envelop(h,sites=range(h.intra.shape[0]),d=0.6)

fv = h.get_operator("sublattice")
fv = h.get_operator("valley")
ops = [fv*o for o in ops] # local times valley


ys = spectrum.ev(h,operator=ops).real

np.savetxt("EV.OUT",np.array([g.x,g.y,ys]).T)










