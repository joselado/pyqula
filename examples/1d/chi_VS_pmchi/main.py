# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")

from pyqula import geometry
import numpy as np
g = geometry.triangular_lattice()
#g = geometry.honeycomb_armchair_ribbon(3)
h = g.get_hamiltonian()
h.set_filling(0.5)
from pyqula import chi

energies = np.linspace(-4.,4.0,60)

(es,cs) = chi.pmchi(h,energies=energies,delta=1e-1)

import matplotlib.pyplot as plt

plt.plot(es,cs)

plt.show()
