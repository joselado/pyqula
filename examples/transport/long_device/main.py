# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")





from pyqula import geometry
from pyqula import heterostructures
import numpy as np
g = geometry.honeycomb_armchair_ribbon(4)
h = g.get_hamiltonian(has_spin=True)
h.add_rashba(0.2)
h.add_exchange([0.3,0.,0.])
nc = 40
hc = [h.copy() for i in range(nc)]
ht = heterostructures.build(h,h,central=hc)
h.get_bands() # get bandstructure
es = np.linspace(-3.,3.,50)

import time

from pyqula import green
green.mode_block_inverse = "gauss"
t0 = time.time()
ts0 = [ht.landauer(e) for e in es]
t1 = time.time()
green.mode_block_inverse = "full"
ts1 = [ht.landauer(e) for e in es]
t2 = time.time()

print("Time with full inversion",t2-t1)
print("Time with Gauss inversion",t1-t0)

import matplotlib.pyplot as plt
plt.scatter(es,ts0,label="Gauss inversion (efficient)",c="red")
plt.plot(es,ts1,label="Full inversion (brute force)",c="blue")
plt.legend()
plt.show()








