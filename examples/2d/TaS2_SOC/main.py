# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")





from pyqula import specialhamiltonian
from pyqula import geometry
import numpy as np
h = specialhamiltonian.TaS2_SOC()
#h = geometry.chain().get_hamiltonian(tij=[1.])
#h.get_bands(operator="sz") 
(e,d) = h.get_dos(nk=300)

import matplotlib.pyplot as plt

plt.plot(e,d)
plt.xlabel("Energy") ; plt.ylabel("DOS")
plt.show()







