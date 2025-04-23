# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")

from pyqula import geometry
import numpy as np
g = geometry.triangular_lattice()
g = geometry.chain()
#g = geometry.honeycomb_armchair_ribbon(3)
h = g.get_hamiltonian()
h.set_filling(0.5)
from pyqula import chi

energies = np.linspace(-8.,8.0,200)

(es,cs) = chi.pmchi(h,energies=energies,delta=1e-1,nk=200)
cfull = h.get_chi(energies=energies,delta=2e-1,nk=200)

import matplotlib.pyplot as plt
plt.subplot(1,2,1) ; plt.title("Poor man")
plt.plot(es,cs.real,label="real")
plt.plot(es,cs.imag,label="imag")
plt.legend()
plt.subplot(1,2,2) ; plt.title("Full version")
plt.plot(energies,cfull.real,label="real")
plt.plot(energies,cfull.imag,label="imag")
plt.legend()

plt.tight_layout()
plt.show()
