# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")


from pyqula import geometry
from pyqula.heterostructures import LocalProbe
import numpy as np
import matplotlib.pyplot as plt
g = geometry.chain()
g = g.get_supercell(2)
#g = geometry.single_square_lattice()
h = g.get_hamiltonian()
#h.add_onsite(2.0)
#h.add_onsite(4.0)
#h.add_exchange([0.,0.,0.6])
#h.add_rashba(0.6)
#h.add_swave(0.1)

r0 = g.r[g.get_central()][0]
def f(r):
    dr = r-r0 ; dr = dr.dot(dr)
    if dr<1e-2: return 2.0 # remove site
    return 0.0

vh = h.copy() ; vh.add_onsite(f)

from pyqula import embedding

eb = embedding.Embedding(h,m=vh) # create the embedding object
es = np.linspace(-2.2,2.2,101)
T = 1e-1
from pyqula import parallel
parallel.cores = 5
ts = parallel.pcall(lambda e: eb.didv(energy=e,T=T),es)
#ts = [lp.didv(energy=e) for e in es]

import matplotlib.pyplot as plt
plt.plot(es,ts)
plt.ylim([0.,max(ts)])
plt.show()

