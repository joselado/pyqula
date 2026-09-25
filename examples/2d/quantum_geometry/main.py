# Add the root path of the pyqula library
import os ; import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")

from pyqula import geometry
from pyqula import topology
g = geometry.honeycomb_lattice()
h = g.get_hamiltonian()
h.add_haldane(0.1) # gapped, with E=0 inside the gap

# Berry curvature and quantum metric of the occupied bands along a k-path,
# from the quantum geometric tensor (the orbitals sit at their positions,
# gauge="atomic", and k is in reduced coordinates)
(ks,g_metric,omega) = topology.quantum_geometric_tensor_path(h,nk=200)
trg = g_metric[:,0,0] + g_metric[:,1,1] # trace of the quantum metric
be = omega[:,0,1] # Berry curvature


import numpy as np


print("Tr g",np.mean(np.abs(trg)))
print("Berry",np.mean(np.abs(be)))

import matplotlib.pyplot as plt

plt.subplot(1,2,1)
plt.plot(ks,be)
plt.xlabel("kpath") ; plt.xticks([]) ; plt.ylabel("Berry curvature")
plt.subplot(1,2,2)
plt.plot(ks,trg)
plt.xlabel("kpath") ; plt.xticks([]) ; plt.ylabel("Tr quantum metric")

plt.tight_layout()
plt.show()
