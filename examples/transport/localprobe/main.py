# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")


from pyqula import geometry
from pyqula.heterostructures import LocalProbe
import numpy as np
import matplotlib.pyplot as plt
g = geometry.chain()
#g = g.get_supercell(100) ; g.dimensionality = 0
#g = geometry.single_square_lattice()
h = g.get_hamiltonian()
#h.add_onsite(2.0)
#h.add_exchange([0.,0.,0.6])
#h.add_rashba(0.6)
h.add_swave(0.1)
#h.get_bands() ; exit()
lp = LocalProbe(h) # create a local probe object
lp.i = 0
lp.mode = "surface" # mode of the calculation
lp.delta = 1e-5
#lp.delta_bulk = 1e-3 # this is the selfenery of the bulk
es = np.linspace(-0.2,0.2,201)
lp.T = 1.0
ts = [lp.didv(energy=e) for e in es]

import matplotlib.pyplot as plt
plt.plot(es,ts)
plt.ylim([0.,max(ts)])
plt.show()

