# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")





from pyqula import geometry
from pyqula.heterostructures import LocalProbe
import numpy as np
import matplotlib.pyplot as plt
g = geometry.chain()
g = geometry.single_square_lattice()
h = g.get_hamiltonian()
h.add_onsite(1.0)
h.add_swave(0.2)
lp = LocalProbe(h) # create a local probe object
lp.delta = 1e-5
lp.delta_bulk = 1e-2
es = np.linspace(-0.4,0.4,10)
lp.T = 1e-2
ts = [lp.didv(energy=e) for e in es]

import matplotlib.pyplot as plt
plt.plot(es,ts)
plt.ylim([0.,max(ts)])
plt.show()

