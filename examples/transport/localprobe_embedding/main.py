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
h = g.get_hamiltonian(tij=[1.0,0.2])
h.add_onsite(1.0)
#h.add_onsite(4.0)
#h.add_exchange([0.,0.,0.6])
#h.add_rashba(0.6)
delta = 0.02
h.add_swave(delta)

r0 = g.r[0]
def fJ(r):
    dr = r-r0 ; dr = dr.dot(dr)
    if dr<1e-2: return [0.,0.,1.] 
    return 0.0

r1 = g.r[1]
W = -2.0
def fW(r):
    dr = r-r1 ; dr = dr.dot(dr)
    if dr<1e-2: return W 
    return 0.0



vh = h.copy() ; vh.add_exchange(fJ) ; vh.add_onsite(fW)

from pyqula import embedding

eb = embedding.Embedding(h,m=vh) # create the embedding object
es = np.linspace(-3*delta,3*delta,601)
T = 1e-1
from pyqula import parallel
#parallel.cores = 5
#ts = parallel.pcall(lambda e: eb.didv(energy=e,T=T),es)
ts0 = [eb.didv(energy=e,T=0.4,i=0) for e in es]
ts1 = [eb.didv(energy=e,T=0.4,i=1) for e in es]

tsa = [ts0,ts1] # array with didv

import matplotlib.pyplot as plt
for ts in tsa:
  plt.plot(es,ts/np.max(ts))
plt.ylim([0.,1.])
plt.show()

