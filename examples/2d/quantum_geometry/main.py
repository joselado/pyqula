# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")

from pyqula import geometry
from pyqula import topology
from pyqula import dos
g = geometry.honeycomb_lattice()
h = g.get_hamiltonian()
h.add_haldane(0.1)
h.shift_fermi(0.9)

# compute Berry curvature and quantum geometry
from pyqula.topologytk import quantumgeometry
(es,qg,be) = quantumgeometry.get_QG_kpath(h,delta=0.1)


import numpy as np


print("QG",np.mean(np.abs(qg)))
print("Berry",np.mean(np.abs(be)))

import matplotlib.pyplot as plt

plt.subplot(1,2,1)
plt.plot(es,be)
plt.xlabel("kpath") ; plt.xticks([]) ; plt.ylabel("Berry curvature")
plt.subplot(1,2,2)
plt.plot(es,qg)
plt.xlabel("kpath") ; plt.xticks([]) ; plt.ylabel("Quantum Geometry")

plt.tight_layout()
plt.show()


