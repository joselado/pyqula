# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")





from pyqula import islands
from pyqula import spectrum
from pyqula import operators

import numpy as np
g = islands.get_geometry(name="honeycomb",n=8,nedges=6,rot=0.0) # get an island
#g = g.remove(g.get_central())

h = g.get_hamiltonian(has_spin = False)
h.add_peierls(0.05)
h.add_sublattice_imbalance(.2)
#h.add_sublattice_imbalance(0.1)
ops = operators.get_envelop(h,sites=range(h.intra.shape[0]),d=0.3)

fv = h.get_operator("valley") # valley function
#fv = operators.get_valley_taux(h,projector=True) # valley function
#ops = [fv(o) for o in ops] # local times valley

#ys = spectrum.ev(h,operator=ops).real
ys = spectrum.real_space_vev(h,operator=fv)

np.savetxt("EV.OUT",np.array([g.x,g.y,ys]).T)










